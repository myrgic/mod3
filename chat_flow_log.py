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

    # Phase timing — wrap a call site with phase_timer:
    async with phase_timer("stt_transcribe", session_id, msg_id):
        result = await loop.run_in_executor(_STT_EXECUTOR, _transcribe)

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

Phase events (chat.phase.*) schema
-----------------------------------
  ts             — ISO 8601 timestamp (UTC, at phase-end)
  event_type     — "chat.phase.<name>"
  session_id     — mod3 session / channel identifier
  message_id     — per-message UUID (short) — may be empty on voice path
  phase_name     — short name (e.g. "stt_transcribe")
  duration_ms    — wall-time for the phase in milliseconds (int)
  ok             — true | false
  error          — error string if ok=false (absent otherwise)

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
import threading
import time
import types
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
# Phase-timing event-type constants
# ---------------------------------------------------------------------------
# Voice path phases
CHAT_PHASE_STT_CAPTURE = "chat.phase.stt_capture"
CHAT_PHASE_STT_TRANSCRIBE = "chat.phase.stt_transcribe"
CHAT_PHASE_TTS_SYNTHESIZE = "chat.phase.tts_synthesize"
CHAT_PHASE_TTS_PLAYBACK_START = "chat.phase.tts_playback_start"

# Both paths
CHAT_PHASE_AGENT_DISPATCH = "chat.phase.agent_dispatch"
CHAT_PHASE_PROVIDER_CALL = "chat.phase.provider_call"
CHAT_PHASE_TOOL_EXECUTE = "chat.phase.tool_execute"
CHAT_PHASE_TURN_TOTAL = "chat.phase.turn_total"

# Prefix used for wildcard matching in queries
CHAT_PHASE_PREFIX = "chat.phase."

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

    def emit_phase(
        self,
        phase_name: str,
        session_id: str,
        message_id: str,
        duration_ms: int,
        *,
        ok: bool = True,
        error: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a phase-timing event.  Never raises.

        Phase events are stored in the same ring buffer as chat-flow events and
        are served by the same /v1/logs/chat-flow endpoint.  They are queryable
        with event_type="chat.phase.<name>" or by prefix using the wildcard
        support in ChatFlowLog.query (pass a comma-separated list or prefix).

        Args:
            phase_name:   Short name such as "stt_transcribe" or "provider_call".
            session_id:   Channel / session identifier for the turn.
            message_id:   Per-message UUID (may be empty string on voice path).
            duration_ms:  Wall-time for the phase in milliseconds.
            ok:           False if the phase raised an exception.
            error:        Exception message when ok=False.
            trace_id:     W3C trace-id (32 hex chars) from the upstream provider
                          request (e.g. from CogOSProvider._make_traceparent).
                          When present, correlates this mod3 phase event with
                          the kernel's bus_traces kernel.chat.subspan.v1 events
                          that share the same trace_id.
        """
        try:
            return self._emit_phase_inner(
                phase_name=phase_name,
                session_id=session_id,
                message_id=message_id,
                duration_ms=duration_ms,
                ok=ok,
                error=error,
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.debug("chat_flow_log.emit_phase failed: %s", exc)
            return {}

    def _emit_phase_inner(
        self,
        phase_name: str,
        session_id: str,
        message_id: str,
        duration_ms: int,
        ok: bool,
        error: str | None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        event_type = f"{CHAT_PHASE_PREFIX}{phase_name}"
        event: dict[str, Any] = {
            "ts": ts,
            "event_type": event_type,
            "session_id": session_id or "",
            "message_id": message_id or "",
            "phase_name": phase_name,
            "duration_ms": int(duration_ms),
            "ok": ok,
        }
        if not ok and error is not None:
            event["error"] = error
        if trace_id:
            event["trace_id"] = trace_id

        with self._lock:
            self._ring.append(event)

        _logger.debug(
            "[%s] session=%s msg=%s duration_ms=%d ok=%s",
            event_type,
            session_id,
            message_id,
            duration_ms,
            ok,
        )

        self._append_to_file(event)
        self._notify_subscribers(event)
        return event

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

        # Parse event_type filter — supports exact names, comma-separated lists,
        # and wildcard suffixes (e.g. "chat.phase.*" matches all phase events).
        event_type_exact: set[str] | None = None
        event_type_prefixes: list[str] | None = None
        if event_type:
            raw_types = [t.strip() for t in event_type.split(",") if t.strip()]
            exact_set: set[str] = set()
            prefix_list: list[str] = []
            for t in raw_types:
                if t.endswith("*"):
                    prefix_list.append(t[:-1])  # strip the trailing '*'
                else:
                    exact_set.add(t)
            event_type_exact = exact_set or None
            event_type_prefixes = prefix_list or None

        with self._lock:
            snapshot = list(self._ring)

        results: list[dict[str, Any]] = []
        for ev in reversed(snapshot):  # newest first
            if session_id and ev.get("session_id") != session_id:
                continue
            if event_type_exact is not None or event_type_prefixes is not None:
                ev_type = ev.get("event_type", "")
                exact_match = event_type_exact is not None and ev_type in event_type_exact
                prefix_match = event_type_prefixes is not None and any(
                    ev_type.startswith(p) for p in event_type_prefixes
                )
                if not exact_match and not prefix_match:
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


# ---------------------------------------------------------------------------
# phase_timer — context managers for call-site instrumentation
# ---------------------------------------------------------------------------
#
# Usage (sync):
#     with phase_timer("provider_call", session_id, msg_id):
#         result = some_sync_call()
#
# Usage (async):
#     async with phase_timer("stt_transcribe", session_id, msg_id):
#         result = await loop.run_in_executor(_STT_EXECUTOR, _transcribe)
#
# Both variants:
#   - measure wall-time via time.perf_counter()
#   - call emit_phase() on __exit__ / __aexit__
#   - never raise — timing infrastructure must not crash real paths
#   - emit ok=False + error=<exc string> if the wrapped block raised
#   - re-raise the original exception after emitting


class _PhaseTimer:
    """Context manager object returned by phase_timer().

    Supports both synchronous (``with``) and asynchronous (``async with``)
    usage from a single instance.
    """

    def __init__(
        self,
        phase_name: str,
        session_id: str,
        message_id: str,
        trace_id: str | None = None,
    ) -> None:
        self._phase_name = phase_name
        self._session_id = session_id
        self._message_id = message_id
        self._trace_id = trace_id
        self._t0: float = 0.0

    # -- sync --

    def __enter__(self) -> "_PhaseTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        duration_ms = int((time.perf_counter() - self._t0) * 1000)
        ok = exc_type is None
        error_str = str(exc_val) if exc_val is not None else None
        try:
            get_chat_flow_log().emit_phase(
                phase_name=self._phase_name,
                session_id=self._session_id,
                message_id=self._message_id,
                duration_ms=duration_ms,
                ok=ok,
                error=error_str,
                trace_id=self._trace_id,
            )
        except Exception:  # noqa: BLE001 — instrumentation must never raise
            pass
        # Return None (falsy) so exceptions propagate normally.

    # -- async --

    async def __aenter__(self) -> "_PhaseTimer":
        self._t0 = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        duration_ms = int((time.perf_counter() - self._t0) * 1000)
        ok = exc_type is None
        error_str = str(exc_val) if exc_val is not None else None
        try:
            get_chat_flow_log().emit_phase(
                phase_name=self._phase_name,
                session_id=self._session_id,
                message_id=self._message_id,
                duration_ms=duration_ms,
                ok=ok,
                error=error_str,
                trace_id=self._trace_id,
            )
        except Exception:  # noqa: BLE001
            pass


def phase_timer(
    phase_name: str,
    session_id: str,
    message_id: str,
    trace_id: str | None = None,
) -> _PhaseTimer:
    """Return a context manager that times a phase and emits a chat.phase.* event.

    Works as both ``with phase_timer(...)`` (synchronous) and
    ``async with phase_timer(...)`` (asynchronous).

    Args:
        phase_name:  Short label for the phase (e.g. "stt_transcribe").
        session_id:  Channel / session identifier for the turn.
        message_id:  Per-message UUID (may be empty on voice path).
        trace_id:    W3C trace-id (32 hex chars) from CogOSProvider._make_traceparent().
                     When provided, the emitted chat.phase.* event includes a
                     ``trace_id`` field correlating it with the kernel's
                     bus_traces kernel.chat.subspan.v1 events.

    Never raises — exceptions from the wrapped block propagate normally;
    emit failures are silently dropped.
    """
    return _PhaseTimer(phase_name, session_id, message_id, trace_id=trace_id)
