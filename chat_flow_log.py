"""chat_flow_log.py — Structured chat-flow event logger for Mod³.

Maintains an in-memory ring buffer (deque, max 5000 events) and optionally
appends events as JSON-per-line to ~/.mod3/chat-flow.log (rotated at 50 MB).

Usage
-----
    from chat_flow_log import get_chat_flow_log

    log = get_chat_flow_log()
    log.emit(
        event_type="chat.message_received",
        session_id="cs-abc123",
        message_id="msg-xyz",
        from_seat="http",
        to_seats=[],
        content="hello world",
        direction="inbound",
    )
    events = log.query(session_id="cs-abc123", limit=20)

Event schema
------------
Each event is a dict with:
  ts             — ISO 8601 timestamp (UTC)
  event_type     — one of the chat.* constants below
  session_id     — mod3 session identifier
  message_id     — per-message UUID (short)
  from_seat      — originating seat_id or "http" / "ws" / "acp"
  to_seats       — list of destination seat_ids (empty = no fan-out)
  content_hash   — first 8 hex chars of sha256(content.encode())
  content_preview— first 80 chars of content (truncated)
  direction      — "inbound" | "outbound"
  error          — error string if event_type is chat.error, else absent

SSE live-tail
-------------
Call ``subscribe()`` to get a ``asyncio.Queue`` that receives all new events
(as dicts). Call ``unsubscribe(q)`` when the SSE stream closes.
Subscribers are per-process singletons; they survive handler restarts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger("mod3.chat_flow")

# ---------------------------------------------------------------------------
# Event-type constants
# ---------------------------------------------------------------------------

CHAT_MESSAGE_RECEIVED = "chat.message_received"
CHAT_FAN_OUT = "chat.fan_out"
CHAT_SEAT_DISPATCH = "chat.seat_dispatch"
CHAT_RESPONSE_GENERATED = "chat.response_generated"
CHAT_MESSAGE_SENT = "chat.message_sent"
CHAT_ECHO_SUPPRESSED = "chat.echo_suppressed"
CHAT_ERROR = "chat.error"

_VALID_EVENT_TYPES = frozenset(
    {
        CHAT_MESSAGE_RECEIVED,
        CHAT_FAN_OUT,
        CHAT_SEAT_DISPATCH,
        CHAT_RESPONSE_GENERATED,
        CHAT_MESSAGE_SENT,
        CHAT_ECHO_SUPPRESSED,
        CHAT_ERROR,
    }
)

# ---------------------------------------------------------------------------
# File rotation constants
# ---------------------------------------------------------------------------

_LOG_PATH = Path.home() / ".mod3" / "chat-flow.log"
_MAX_LOG_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Core logger class
# ---------------------------------------------------------------------------


class ChatFlowLog:
    """Thread-safe structured chat-flow event logger."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ring: deque[dict[str, Any]] = deque(maxlen=5000)
        # SSE subscribers — asyncio.Queue per active /v1/logs/chat-flow/stream
        self._subscribers: list[asyncio.Queue] = []
        self._subs_lock = threading.Lock()
        # File handle (lazy open)
        self._file = None
        self._file_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(
        self,
        event_type: str,
        session_id: str,
        message_id: str,
        from_seat: str,
        to_seats: list[str],
        content: str,
        direction: str,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build and record a chat-flow event.  Never raises."""
        try:
            return self._emit_inner(
                event_type=event_type,
                session_id=session_id,
                message_id=message_id,
                from_seat=from_seat,
                to_seats=to_seats,
                content=content,
                direction=direction,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.debug("chat_flow_log.emit failed: %s", exc)
            return {}

    def _emit_inner(
        self,
        event_type: str,
        session_id: str,
        message_id: str,
        from_seat: str,
        to_seats: list[str],
        content: str,
        direction: str,
        error: str | None,
    ) -> dict[str, Any]:
        content_str = content or ""
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:8]
        content_preview = content_str[:80]
        ts = datetime.now(timezone.utc).isoformat()

        event: dict[str, Any] = {
            "ts": ts,
            "event_type": event_type,
            "session_id": session_id or "",
            "message_id": message_id or "",
            "from_seat": from_seat or "",
            "to_seats": list(to_seats or []),
            "content_hash": content_hash,
            "content_preview": content_preview,
            "direction": direction or "",
        }
        if error is not None:
            event["error"] = error

        # Ring buffer
        with self._lock:
            self._ring.append(event)

        # Python logger (DEBUG)
        _logger.debug(
            "[%s] session=%s msg=%s from=%s dir=%s hash=%s preview=%r",
            event_type,
            session_id,
            message_id,
            from_seat,
            direction,
            content_hash,
            content_preview,
        )

        # File append (best-effort)
        self._append_to_file(event)

        # Notify SSE subscribers (thread-safe handoff to asyncio)
        self._notify_subscribers(event)

        return event

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,  # ISO timestamp
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return events from the ring buffer matching the given filters."""
        limit = max(1, min(limit, 1000))
        since_ts: str | None = None
        if since:
            try:
                # Normalise to comparable ISO string
                dt = datetime.fromisoformat(since)
                since_ts = dt.isoformat()
            except ValueError:
                pass  # bad since — ignore filter

        event_types: set[str] | None = None
        if event_type:
            event_types = {t.strip() for t in event_type.split(",") if t.strip()}

        with self._lock:
            snapshot = list(self._ring)

        results: list[dict[str, Any]] = []
        for ev in reversed(snapshot):  # newest first
            if session_id and ev.get("session_id") != session_id:
                continue
            if event_types and ev.get("event_type") not in event_types:
                continue
            if since_ts and ev.get("ts", "") < since_ts:
                continue
            results.append(ev)
            if len(results) >= limit:
                break

        # Return in chronological order
        results.reverse()
        return results

    # ------------------------------------------------------------------
    # SSE subscriber management
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber.  Returns a queue of event dicts."""
        q: asyncio.Queue = asyncio.Queue()
        with self._subs_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove an SSE subscriber queue."""
        with self._subs_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _notify_subscribers(self, event: dict[str, Any]) -> None:
        with self._subs_lock:
            subs = list(self._subscribers)
        for q in subs:
            # Non-blocking put from any thread
            try:
                q.put_nowait(event)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # File append
    # ------------------------------------------------------------------

    def _append_to_file(self, event: dict[str, Any]) -> None:
        """Append the event as JSON-per-line to ~/.mod3/chat-flow.log.

        Rotates (truncates) the file when it exceeds 50 MB.  All I/O errors
        are silently dropped — file logging is best-effort.
        """
        try:
            with self._file_lock:
                _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                # Check size for rotation
                try:
                    size = _LOG_PATH.stat().st_size
                except FileNotFoundError:
                    size = 0
                mode = "w" if size >= _MAX_LOG_BYTES else "a"
                with _LOG_PATH.open(mode, encoding="utf-8") as fh:
                    fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001
            _logger.debug("chat_flow_log file append failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: ChatFlowLog | None = None
_singleton_lock = threading.Lock()


def get_chat_flow_log() -> ChatFlowLog:
    """Return the process-singleton ChatFlowLog, creating it on first call."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ChatFlowLog()
    return _singleton
