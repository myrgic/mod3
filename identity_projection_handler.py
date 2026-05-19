"""SSE bridge handler for cogos identity-projection events (Primitive 3, Wave 6c).

Subscribes to the following event kinds emitted by cogos identity_provider.go:

    identity.projected         — new identity reconciled for the first time
    identity.expression.updated — existing identity's expression(s) changed

Cogos emits these events with this payload shape (identity_provider.go:658-666):

    {
        "iss":             "<issuer>",
        "sub":             "<subject slug, e.g. 'cog'>",
        "aud":             "<primary audience expression>",
        "action":          "create" | "update",
        "spec_hash":       "<sha256 of spec>",
        "projection_path": ".cog/id/<sub>.cog.md",
        "applied_at":      "<RFC3339>",
    }

NOTE — gap vs. directive assumption: the event payload does NOT embed the full
IdentityProjection struct inline. In particular, ``voice_profile`` is NOT present
in today's cogos event. This handler is written to handle BOTH cases:

  1. ``voice_profile`` present in payload (future-proofed — cogos may enrich the
     event in a follow-up). When present, the generative conditionals_ref URI is
     resolved immediately.
  2. ``voice_profile`` absent (today's cogos). The handler records the identity
     sub/iss/projection_path for future enrichment and logs a debug note. When
     cogos gains a GET /v1/identity/projections/{sub} endpoint, this handler
     should be updated to fetch-and-resolve on event receipt.

Follow-up gap: cogos needs ``GET /v1/identity/projections/{sub}`` so that mod3
can fetch the full IdentityProjection (including voice_profile) on event arrival
without requiring the event payload to carry the full struct.

Usage from bus_bridge_runner::

    from identity_projection_handler import IdentityVoiceCache, handle_identity_event

    cache = IdentityVoiceCache()

    async for env in subscriber.stream():
        if env.kind in IDENTITY_KINDS:
            handle_identity_event(env.payload, cache)
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

from voice_profile_schema import IdentityVoiceProfile, resolve_voices_uri

logger = logging.getLogger("mod3.identity_projection")

# Event kind strings from cogos identity_provider.go:215-217
IDENTITY_KIND_PROJECTED = "identity.projected"
IDENTITY_KIND_EXPRESSION_UPDATED = "identity.expression.updated"

# Both kinds carry the same payload shape and get the same handler logic.
IDENTITY_KINDS: frozenset[str] = frozenset({IDENTITY_KIND_PROJECTED, IDENTITY_KIND_EXPRESSION_UPDATED})


@dataclass
class ResolvedVoiceProfile:
    """Resolved voice state for one identity.

    ``profile`` is the structured voice config parsed from the cogos event.
    ``generative_path`` is the resolved filesystem path for the generative
    conditionals (e.g. ~/.mod3/voices/cog.safetensors) — may be None if the
    identity has no generative head, or if resolution failed.
    ``discriminative_path`` is the resolved path for the ECAPA embedding
    (schema-only in this wave).
    ``pending_fetch`` is True when the event arrived but voice_profile was
    absent — the sub/projection_path are recorded but no resolution ran yet.
    """

    sub: str
    iss: Optional[str] = None
    projection_path: Optional[str] = None
    profile: Optional[IdentityVoiceProfile] = None
    generative_path: Optional[pathlib.Path] = None
    discriminative_path: Optional[pathlib.Path] = None
    # True when voice_profile was absent from the event — awaiting follow-up
    # fetch once cogos exposes GET /v1/identity/projections/{sub}.
    pending_fetch: bool = False


class IdentityVoiceCache:
    """Thread-safe cache mapping identity sub → ResolvedVoiceProfile.

    Updated by ``handle_identity_event``. Read by the TTS path to find a
    pre-resolved conditionals path for a given identity subject slug.

    This cache sits alongside the session-level voice pool in session_registry.py
    — it operates at identity scope (persistent across sessions) rather than
    session scope (per-connection). A TTS caller that has both a session voice
    and an identity voice profile should prefer the identity profile's
    conditionals_ref for the generative engine, since it carries the cloned
    speaker persona rather than just a name token.
    """

    def __init__(self) -> None:
        self._by_sub: dict[str, ResolvedVoiceProfile] = {}

    def get(self, sub: str) -> Optional[ResolvedVoiceProfile]:
        """Return the cached ResolvedVoiceProfile for ``sub``, or None."""
        return self._by_sub.get(sub)

    def put(self, resolved: ResolvedVoiceProfile) -> None:
        """Store or update the cache entry for ``resolved.sub``."""
        self._by_sub[resolved.sub] = resolved

    def all_subs(self) -> list[str]:
        """Return all cached subject slugs."""
        return list(self._by_sub.keys())


def handle_identity_event(payload: dict, cache: IdentityVoiceCache) -> None:
    """Parse one identity-projection event and update the voice cache.

    Designed to be called from the SSE bridge loop for events whose
    ``kind`` is in ``IDENTITY_KINDS``. Safe to call from the async event
    loop — does no I/O beyond path construction (resolve_voices_uri is
    filesystem path arithmetic, not disk I/O).

    Behavior:
      - ``voice_profile`` present and non-None → parse, resolve URIs, cache.
      - ``voice_profile`` absent or None → record sub/projection_path as
        pending-fetch, no resolution. Does not raise.
      - Malformed payload (missing required fields) → logs a warning, returns
        without touching the cache, does not crash.
      - Idempotent: same event twice updates the cache entry without error.
    """
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        logger.warning(
            "identity-bridge: event missing 'sub' field — skipping. payload_keys=%s",
            sorted(payload.keys())[:10],
        )
        return

    iss = payload.get("iss")
    projection_path = payload.get("projection_path")
    vp_raw = payload.get("voice_profile")

    existing = cache.get(sub)
    is_first_time = existing is None

    if vp_raw is None:
        # voice_profile absent — this is today's cogos behavior.
        # Record the identity for future enrichment (see module docstring gap note).
        resolved = ResolvedVoiceProfile(
            sub=sub,
            iss=iss if isinstance(iss, str) else None,
            projection_path=projection_path if isinstance(projection_path, str) else None,
            pending_fetch=True,
        )
        cache.put(resolved)
        logger.debug(
            "identity-bridge: voice_profile absent for sub=%r — recorded as pending-fetch "
            "(awaiting cogos GET /v1/identity/projections/{sub} endpoint)",
            sub,
        )
        return

    # voice_profile is present (future cogos enriched event, or test fixture).
    if not isinstance(vp_raw, dict):
        logger.warning(
            "identity-bridge: voice_profile for sub=%r is not a dict (got %s) — skipping",
            sub,
            type(vp_raw).__name__,
        )
        return

    try:
        profile = IdentityVoiceProfile.from_dict(vp_raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "identity-bridge: failed to parse IdentityVoiceProfile for sub=%r: %s — skipping",
            sub,
            exc,
        )
        return

    generative_path: Optional[pathlib.Path] = None
    discriminative_path: Optional[pathlib.Path] = None

    if profile.generative is not None:
        uri = profile.generative.conditionals_ref
        try:
            generative_path = resolve_voices_uri(uri)
        except ValueError as exc:
            logger.warning(
                "identity-bridge: could not resolve generative URI %r for sub=%r: %s",
                uri,
                sub,
                exc,
            )

    if profile.discriminative is not None:
        uri = profile.discriminative.embedding_ref
        try:
            discriminative_path = resolve_voices_uri(uri)
        except ValueError as exc:
            logger.warning(
                "identity-bridge: could not resolve discriminative URI %r for sub=%r: %s",
                uri,
                sub,
                exc,
            )

    resolved = ResolvedVoiceProfile(
        sub=sub,
        iss=iss if isinstance(iss, str) else None,
        projection_path=projection_path if isinstance(projection_path, str) else None,
        profile=profile,
        generative_path=generative_path,
        discriminative_path=discriminative_path,
        pending_fetch=False,
    )
    cache.put(resolved)

    if is_first_time:
        logger.info(
            "identity-bridge: resolved voice profile for sub=%r engine=%s generative_path=%s",
            sub,
            profile.generative.engine if profile.generative else None,
            generative_path,
        )
    else:
        logger.debug(
            "identity-bridge: re-resolved voice profile for sub=%r (idempotent update)",
            sub,
        )
