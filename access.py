"""Mod³ channel access control — sender gating and pairing flow.

access.json schema
------------------
{
  "policy": "self" | "allowlist" | "deny",
  "allow": ["<identifier>", ...],
  "pending": [
    {"code": "abcde", "identifier": "<uuid>", "expires_at": "<iso8601>"}
  ]
}

Policy semantics
----------------
  self       — only localhost connections (127.0.0.1 / ::1) are allowed;
               no remote clients permitted even if listed in allow[].
  allowlist  — localhost auto-allowed; remote clients allowed if their
               identifier appears in allow[].
  deny       — all connections refused (maintenance / lockdown mode).

Pairing flow (for non-localhost clients)
-----------------------------------------
1. Remote client connects with a stable device UUID (header or WS sub-protocol).
2. BrowserChannel calls is_allowed(identifier, host) — returns False for unknowns.
3. BrowserChannel calls add_pending(identifier) — generates a 5-letter code,
   stores it in pending[], emits a pairing_request channel notification.
4. Operator runs /mod3:access pair <code> in their Claude Code terminal.
5. The skill calls confirm_pairing(code) here, which moves the identifier
   from pending to allow[] and persists the file.
6. Subsequent connections from that identifier pass is_allowed().

Security notes
--------------
- Pairing approval MUST come from the operator's Claude Code terminal via the
  /mod3:access skill, never through the channel itself (prompt-injection risk).
- Trust only the actual TCP peername for localhost detection; X-Forwarded-For
  and similar headers are NOT consulted.
- Codes use the alphabet [a-km-z] (26 − 1 = 25 letters, 'l' excluded) matching
  Claude Code's own permission-relay ID format.  With 5-letter codes the
  collision space is 25^5 = 9,765,625 — adequate for session-scoped pairing.
- Codes expire after MOD3_PAIRING_EXPIRE_SECONDS (default: 600s / 10 min).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mod3.access")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAIRING_ALPHABET = "abcdefghijkmnopqrstuvwxyz"  # 'l' excluded (visually ambiguous)
_CODE_LENGTH = 5
_DEFAULT_EXPIRE_SECONDS = 600  # 10 minutes

_LOCALHOST_ADDRS = frozenset({"127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1"})


def _config_path() -> Path:
    return Path(os.environ.get("MOD3_ACCESS_CONFIG", "")).expanduser() or (
        Path.home() / ".claude" / "channels" / "mod3" / "access.json"
    )


# ---------------------------------------------------------------------------
# File I/O (with a module-level lock for thread safety)
# ---------------------------------------------------------------------------

_file_lock = threading.Lock()


def _load() -> dict:
    path = _config_path()
    if not path.exists():
        return {"policy": "self", "allow": [], "pending": []}
    try:
        with open(path) as f:
            data = json.load(f)
        # Normalise missing keys
        data.setdefault("policy", "self")
        data.setdefault("allow", [])
        data.setdefault("pending", [])
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read access.json: %s", exc)
        return {"policy": "self", "allow": [], "pending": []}


def _save(data: dict) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        tmp.replace(path)
    except OSError as exc:
        logger.error("Failed to write access.json: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_localhost(host: str) -> bool:
    """Return True if the host address is a loopback address."""
    return host.strip() in _LOCALHOST_ADDRS


def is_allowed(identifier: str, host: str = "") -> bool:
    """Return True if this connection is permitted to use the channel.

    Args:
        identifier: Stable device identifier from the client.
        host: Actual TCP peername (NOT proxy headers).  Used for localhost
              detection; empty string is treated as non-localhost.

    Returns:
        True if allowed, False if denied or pending approval.
    """
    localhost = is_localhost(host)

    with _file_lock:
        data = _load()

    policy = data.get("policy", "self")

    if policy == "deny":
        return False

    if localhost:
        # Localhost is always allowed under "self" and "allowlist" policies.
        return True

    # Non-localhost connections
    if policy == "self":
        logger.info("access denied (policy=self): identifier=%s host=%s", identifier, host)
        return False

    # policy == "allowlist"
    allowed = data.get("allow", [])
    if identifier in allowed:
        return True

    logger.info("access denied (not in allowlist): identifier=%s", identifier)
    return False


def _generate_code() -> str:
    return "".join(random.choices(_PAIRING_ALPHABET, k=_CODE_LENGTH))


def _expire_seconds() -> int:
    try:
        return int(os.environ.get("MOD3_PAIRING_EXPIRE_SECONDS", _DEFAULT_EXPIRE_SECONDS))
    except ValueError:
        return _DEFAULT_EXPIRE_SECONDS


def add_pending(identifier: str) -> str:
    """Register a pairing request for an unknown remote identifier.

    Generates a 5-letter code, stores it in pending[], and returns the code.
    The caller is responsible for emitting the pairing_request channel notification
    so Claude Code can surface the approval prompt.

    If the identifier already has a pending (non-expired) entry, returns the
    existing code rather than creating a duplicate.
    """
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(seconds=_expire_seconds())

    with _file_lock:
        data = _load()
        pending = data.get("pending", [])

        # Return existing non-expired code if present
        for entry in pending:
            if entry.get("identifier") == identifier:
                try:
                    exp = datetime.fromisoformat(entry["expires_at"])
                    if exp > now:
                        logger.debug("reusing existing pairing code for identifier=%s", identifier)
                        return entry["code"]
                except (KeyError, ValueError):
                    pass

        # Generate a fresh code (retry on collision)
        existing_codes = {e["code"] for e in pending}
        for _ in range(20):
            code = _generate_code()
            if code not in existing_codes:
                break

        # Prune expired entries and append new one
        data["pending"] = [
            e for e in pending
            if _entry_not_expired(e, now)
        ]
        data["pending"].append({
            "code": code,
            "identifier": identifier,
            "expires_at": expire_at.isoformat(),
        })
        _save(data)

    logger.info("pairing pending: code=%s identifier=%s expires=%s", code, identifier, expire_at.isoformat())
    return code


def _entry_not_expired(entry: dict, now: datetime) -> bool:
    try:
        return datetime.fromisoformat(entry["expires_at"]) > now
    except (KeyError, ValueError):
        return False


def confirm_pairing(code: str) -> Optional[str]:
    """Approve a pending pairing request by code.

    Moves the identifier from pending[] to allow[] and persists.

    Returns:
        The confirmed identifier on success, or None if the code is
        unknown / expired.
    """
    now = datetime.now(timezone.utc)
    with _file_lock:
        data = _load()
        pending = data.get("pending", [])
        matched = None
        remaining = []
        for entry in pending:
            if entry.get("code") == code and _entry_not_expired(entry, now):
                matched = entry.get("identifier")
            else:
                remaining.append(entry)

        if matched is None:
            logger.warning("confirm_pairing: code=%r not found or expired", code)
            return None

        data["pending"] = remaining
        allowed = data.get("allow", [])
        if matched not in allowed:
            allowed.append(matched)
        data["allow"] = allowed
        _save(data)

    logger.info("pairing confirmed: code=%s identifier=%s", code, matched)
    return matched


def set_policy(policy: str) -> None:
    """Set the access policy. policy must be 'self', 'allowlist', or 'deny'."""
    if policy not in ("self", "allowlist", "deny"):
        raise ValueError(f"Invalid policy {policy!r}; must be 'self', 'allowlist', or 'deny'")
    with _file_lock:
        data = _load()
        data["policy"] = policy
        _save(data)
    logger.info("access policy set to: %s", policy)


def add_identifier(identifier: str) -> None:
    """Manually add an identifier to the allowlist."""
    with _file_lock:
        data = _load()
        allowed = data.get("allow", [])
        if identifier not in allowed:
            allowed.append(identifier)
        data["allow"] = allowed
        _save(data)
    logger.info("identifier added to allowlist: %s", identifier)


def list_access() -> dict:
    """Return the current access.json contents (a copy)."""
    with _file_lock:
        return _load()
