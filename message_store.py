"""In-memory per-session chat history.

A small ring buffer keyed by ``session_id`` so the dashboard can:

- Hydrate the main chat pane on page refresh (without losing the conversation).
- Render the history of a session the operator clicks into from the sidebar
  (without having opened that conversation in the current tab).

The store is intentionally RAM-only and bounded — restart wipes it, parity with
the rest of mod3's in-memory state (seats, session registry, chat-flow log).
File-backed persistence is a future enhancement; the v1 surface here is shaped
so it can be swapped without touching callers.

Storage shape per session:

    {
      "id":         "<8-char uuid>",
      "session_id": "<sid>",
      "role":       "user" | "assistant",
      "content":    "<text>",
      "input_type": "text" | "voice",
      "ts":         <epoch seconds>,
    }

Thread-safe: append/get/clear are guarded by an asyncio-free RLock so callers
inside FastAPI request handlers and background tasks share the same view.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any

DEFAULT_MAX_PER_SESSION = 500


class MessageStore:
    """Per-session ring buffer of chat messages."""

    def __init__(self, max_per_session: int = DEFAULT_MAX_PER_SESSION) -> None:
        self._max = max_per_session
        self._lock = threading.RLock()
        self._buckets: dict[str, deque[dict[str, Any]]] = {}

    # ----- mutators -----

    def append(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        input_type: str = "text",
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required")
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")

        entry = {
            "id": uuid.uuid4().hex[:8],
            "session_id": session_id,
            "role": role,
            "content": content,
            "input_type": input_type,
            "ts": time.time(),
        }
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                bucket = deque(maxlen=self._max)
                self._buckets[session_id] = bucket
            bucket.append(entry)
        return entry

    def clear(self, session_id: str) -> int:
        with self._lock:
            bucket = self._buckets.pop(session_id, None)
        return len(bucket) if bucket else 0

    # ----- accessors -----

    def get(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            bucket = self._buckets.get(session_id)
            if not bucket:
                return []
            if limit is None or limit >= len(bucket):
                return list(bucket)
            # last `limit` entries
            return list(bucket)[-limit:]

    def count(self, session_id: str) -> int:
        with self._lock:
            bucket = self._buckets.get(session_id)
            return len(bucket) if bucket else 0

    def known_sessions(self) -> list[str]:
        with self._lock:
            return list(self._buckets.keys())


_default: MessageStore | None = None
_default_lock = threading.Lock()


def get_default_store() -> MessageStore:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = MessageStore()
    return _default


def reset_default_store_for_tests() -> None:
    """Discard the singleton — test-only helper."""
    global _default
    with _default_lock:
        _default = None
