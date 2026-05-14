"""Mod³ session-seat registry — in-memory seat management for channel clients.

Architecture
------------
A "seat" is a logical slot held by a channel_client.py subprocess within a
mod³ session.  Multiple channel clients (one per Claude Code session) can
attach to the same mod³ session and receive events fanned out by the daemon.

The seat registry is process-local (single FastAPI process, single daemon).
Seats are keyed by (session_id, seat_id).  Each seat owns an asyncio.Queue
that drives its SSE stream; when a dashboard message arrives it is pushed
into every seat queue attached to that session.

HTTP endpoints (wired in http_api.py):
  POST   /v1/sessions/{session_id}/seats
  DELETE /v1/sessions/{session_id}/seats/{seat_id}
  GET    /v1/sessions/{session_id}/seats/{seat_id}/events  (SSE)
  POST   /v1/sessions/{session_id}/messages               (dashboard fan-out)

Fan-out policy (v0): broadcast to all seats in the session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mod3.seats")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_CLIENT_TYPES = frozenset({"claude-code-channel", "generic"})

_SEAT_TTL_SECONDS = 3600  # seats auto-expire after 1 hour of inactivity


@dataclass
class Seat:
    seat_id: str
    session_id: str
    client_type: str
    device_uuid: str
    created_at: float = field(default_factory=time.time)
    # SSE event queue — one entry per pending event
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # asyncio loop that owns this seat's queue
    loop: asyncio.AbstractEventLoop | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "session_id": self.session_id,
            "client_type": self.client_type,
            "device_uuid": self.device_uuid,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SeatRegistry:
    """Thread-safe in-memory registry of channel-client seats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {session_id: {seat_id: Seat}}
        self._seats: dict[str, dict[str, Seat]] = {}

    # ------------------------------------------------------------------
    # Seat lifecycle
    # ------------------------------------------------------------------

    def register(
        self,
        session_id: str,
        client_type: str,
        device_uuid: str,
    ) -> Seat:
        """Create a new seat in *session_id*.  Auto-creates the session bucket."""
        if client_type not in VALID_CLIENT_TYPES:
            client_type = "generic"
        seat_id = str(uuid.uuid4())
        seat = Seat(
            seat_id=seat_id,
            session_id=session_id,
            client_type=client_type,
            device_uuid=device_uuid,
        )
        with self._lock:
            if session_id not in self._seats:
                self._seats[session_id] = {}
            self._seats[session_id][seat_id] = seat
        logger.info("Seat %s registered in session %s (client=%s)", seat_id, session_id, client_type)
        return seat

    def get(self, session_id: str, seat_id: str) -> Seat | None:
        with self._lock:
            return self._seats.get(session_id, {}).get(seat_id)

    def revoke(self, session_id: str, seat_id: str) -> bool:
        """Remove a seat.  Returns True if the seat existed."""
        with self._lock:
            session_seats = self._seats.get(session_id)
            if not session_seats:
                return False
            seat = session_seats.pop(seat_id, None)
            if seat is None:
                return False
            # Signal SSE stream to close
            _enqueue_nowait(seat, {"type": "_close"})
            logger.info("Seat %s revoked from session %s", seat_id, session_id)
            return True

    def list_session_seats(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._seats.get(session_id, {}).values()]

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._seats.keys())

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    def fan_out(self, session_id: str, event: dict[str, Any]) -> int:
        """Broadcast *event* to all seats attached to *session_id*.

        Returns the number of seats that received the event.
        """
        with self._lock:
            seats = list(self._seats.get(session_id, {}).values())
        count = 0
        for seat in seats:
            _enqueue_nowait(seat, event)
            count += 1
        if count:
            logger.debug("Fan-out to %d seats in session %s: type=%s", count, session_id, event.get("type"))
        return count

    def fan_out_all(self, event: dict[str, Any]) -> int:
        """Broadcast *event* to ALL seats across all sessions."""
        with self._lock:
            all_seats = [
                seat
                for session_seats in self._seats.values()
                for seat in session_seats.values()
            ]
        for seat in all_seats:
            _enqueue_nowait(seat, event)
        return len(all_seats)


def _enqueue_nowait(seat: Seat, event: dict[str, Any]) -> None:
    """Thread-safe enqueue — uses the seat's loop if available."""
    if seat.loop is not None and seat.loop.is_running():
        seat.loop.call_soon_threadsafe(seat.queue.put_nowait, event)
    else:
        try:
            seat.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug("Seat %s queue full — dropping event %s", seat.seat_id, event.get("type"))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry: SeatRegistry | None = None
_registry_lock = threading.Lock()


def get_seat_registry() -> SeatRegistry:
    """Return the process-singleton SeatRegistry."""
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = SeatRegistry()
    return _default_registry


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


async def sse_stream(seat: Seat):
    """Async generator yielding raw SSE text lines for a seat's event queue.

    Yields formatted SSE strings.  Yields a keep-alive comment every 15 s.
    Closes cleanly when a ``{"type": "_close"}`` sentinel is dequeued or
    when the caller cancels the task.
    """
    seat.loop = asyncio.get_running_loop()
    KEEPALIVE_INTERVAL = 15.0
    try:
        while True:
            try:
                event = await asyncio.wait_for(seat.queue.get(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # Keep-alive ping
                yield ": keepalive\n\n"
                continue

            if event.get("type") == "_close":
                break

            etype = event.get("type", "event")
            data = json.dumps(event, separators=(",", ":"))
            yield f"event: {etype}\ndata: {data}\n\n"

    except asyncio.CancelledError:
        pass
    finally:
        logger.debug("SSE stream closed for seat %s", seat.seat_id)
