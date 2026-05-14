"""Per-session audio subscriber registry (Wave 4.3 / RTVI 1.3.0).

A separate, tiny module so the routing surface stays independent of the
session registry (which is already load-bearing for ADR-082 Phase 1). The
dashboard WebSocket lands here; the kernel queries the
``/v1/sessions/{id}/subscribers`` HTTP endpoint before spawning afplay.

Thread-safety: registration happens in FastAPI's event loop thread (WS
handler), lookup and emit happen from the server playback thread. A
single RLock around the dict is sufficient — the set-per-session is small
(usually 1 dashboard) and contention is effectively zero.

Delivery semantics: the emit paths are best-effort. A WebSocket that has
already died just drops the frame — the kernel fallback ``afplay``
never happened because the check went through before we spawned it, but
the session will miss this turn's audio. That's acceptable: the dashboard
polls and reconnects; the user sees silence for one turn instead of
nothing at all.

Wire format (RTVI 1.3.0)
------------------------
Each utterance is delivered as three JSON text frames over the WebSocket:

  1. ``bot-tts-started`` -- before first chunk.
  2. ``bot-tts-audio`` -- base64-encoded int16 PCM extracted from the WAV
     (44-byte RIFF header stripped).
  3. ``bot-tts-stopped`` -- after last chunk.

No binary frames are sent. The legacy two-frame (header JSON + binary WAV)
shape has been replaced.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover — type-check-only import
    from fastapi import WebSocket

logger = logging.getLogger("mod3.audio_subscribers")

# RIFF WAV header is exactly 44 bytes for standard PCM files.
_WAV_HEADER_BYTES = 44


def _extract_pcm_from_wav(wav_bytes: bytes) -> bytes:
    """Strip the 44-byte RIFF header; return raw int16 PCM bytes.

    Standard PCM WAV: 44-byte header, then raw int16 little-endian samples.
    If the input is shorter than the header, return empty bytes rather than
    raising -- callers tolerate empty audio gracefully.
    """
    if len(wav_bytes) <= _WAV_HEADER_BYTES:
        return b""
    return wav_bytes[_WAV_HEADER_BYTES:]


@dataclass
class _Subscriber:
    """A single WebSocket subscription for a session."""

    ws: "WebSocket"
    # The event loop the WebSocket was accepted on. Emit calls from other
    # threads need to run_coroutine_threadsafe onto this loop.
    loop: asyncio.AbstractEventLoop
    # Monotonic sequence for logging / frame ordering. Opaque to callers.
    seq: int = 0


@dataclass
class _SessionBucket:
    """Subscribers currently attached to a session_id."""

    subscribers: list[_Subscriber] = field(default_factory=list)


class AudioSubscriberRegistry:
    """Thread-safe session_id → active WebSocket subscribers mapping.

    Callers never reach into the bucket lists directly — they go through
    register / unregister / has_subscribers / emit_wav.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._buckets: dict[str, _SessionBucket] = {}
        self._frame_seq = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, session_id: str, ws: "WebSocket", loop: asyncio.AbstractEventLoop) -> _Subscriber:
        """Attach ``ws`` to ``session_id``. Caller must have already
        ``accept()``-ed the WebSocket.
        """
        sub = _Subscriber(ws=ws, loop=loop)
        with self._lock:
            bucket = self._buckets.setdefault(session_id, _SessionBucket())
            bucket.subscribers.append(sub)
            count = len(bucket.subscribers)
        logger.info("audio subscriber attached: session=%s total=%d", session_id, count)
        return sub

    def unregister(self, session_id: str, sub: _Subscriber) -> None:
        """Detach. Idempotent — double-unregister is a no-op."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return
            try:
                bucket.subscribers.remove(sub)
            except ValueError:
                return
            remaining = len(bucket.subscribers)
            if not bucket.subscribers:
                # Drop empty buckets so the subscribed-check stays fast.
                self._buckets.pop(session_id, None)
        logger.info("audio subscriber detached: session=%s remaining=%d", session_id, remaining)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def has_subscribers(self, session_id: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(session_id)
            return bool(bucket and bucket.subscribers)

    def count(self, session_id: str) -> int:
        with self._lock:
            bucket = self._buckets.get(session_id)
            return len(bucket.subscribers) if bucket else 0

    def snapshot(self) -> dict[str, int]:
        """session_id → subscriber count. For diagnostics only."""
        with self._lock:
            return {sid: len(b.subscribers) for sid, b in self._buckets.items()}

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit_wav(
        self,
        session_id: str,
        wav_bytes: bytes,
        *,
        job_id: str | None = None,
        duration_sec: float | None = None,
        sample_rate: int | None = None,
    ) -> int:
        """Push a WAV blob to every subscriber of ``session_id`` as RTVI 1.3.0 frames.

        Returns the number of subscribers the frame was enqueued for. Each
        send is fire-and-forget via ``run_coroutine_threadsafe`` -- the caller
        doesn't block on socket I/O, which matches the existing
        ``BrowserChannel.broadcast_trace_event`` pattern.

        Wire format: RTVI 1.3.0 envelope -- three JSON text frames:
          1. ``bot-tts-started`` marker
          2. ``bot-tts-audio`` with base64-encoded int16 PCM (RIFF header stripped)
          3. ``bot-tts-stopped`` marker
        """
        with self._lock:
            bucket = self._buckets.get(session_id)
            if not bucket or not bucket.subscribers:
                return 0
            targets = list(bucket.subscribers)
            self._frame_seq += 1
            seq = self._frame_seq

        # Extract int16 PCM from WAV (strip 44-byte RIFF header).
        pcm_bytes = _extract_pcm_from_wav(wav_bytes)
        audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
        sr = sample_rate or 24000

        frames = _build_rtvi_frames(audio_b64=audio_b64, sample_rate=sr)

        delivered = 0
        for sub in targets:
            try:
                asyncio.run_coroutine_threadsafe(
                    _send_rtvi_frames(sub.ws, frames),
                    sub.loop,
                )
                delivered += 1
            except Exception as exc:  # noqa: BLE001 — disconnected subscribers are expected
                logger.debug("emit_wav: scheduling failed for %s: %s", session_id, exc)
        logger.info(
            "emit_wav: session=%s bytes=%d pcm_bytes=%d delivered_to=%d seq=%d",
            session_id,
            len(wav_bytes),
            len(pcm_bytes),
            delivered,
            seq,
        )
        return delivered


def _build_rtvi_frames(*, audio_b64: str, sample_rate: int) -> list[str]:
    """Build the three JSON strings that form one RTVI utterance delivery."""
    started_id = str(uuid4())
    audio_id = str(uuid4())
    stopped_id = str(uuid4())

    started = f'{{"label":"rtvi-ai","type":"bot-tts-started","id":"{started_id}","data":{{}}}}'
    audio = (
        f'{{"label":"rtvi-ai","type":"bot-tts-audio","id":"{audio_id}",'
        f'"data":{{"audio":"{audio_b64}","sample_rate":{sample_rate},"num_channels":1}}}}'
    )
    stopped = f'{{"label":"rtvi-ai","type":"bot-tts-stopped","id":"{stopped_id}","data":{{}}}}'
    return [started, audio, stopped]


async def _send_rtvi_frames(ws: "WebSocket", frames: list[str]) -> None:
    """Send the RTVI frame sequence (started, audio, stopped) over a WebSocket.

    Split into a module-level coroutine so ``run_coroutine_threadsafe``
    returns a Future the caller can ignore -- all frames are sent in order
    on the socket's own coroutine context.
    """
    try:
        for frame in frames:
            await ws.send_text(frame)
    except Exception as exc:  # noqa: BLE001 — disconnect mid-send is expected
        logger.debug("rtvi frame send failed: %s", exc)


# ---------------------------------------------------------------------------
# Process-global default registry — shared by http_api and server.
# ---------------------------------------------------------------------------

_default_registry: AudioSubscriberRegistry | None = None
_default_registry_lock = threading.Lock()


def get_default_audio_subscribers() -> AudioSubscriberRegistry:
    global _default_registry
    with _default_registry_lock:
        if _default_registry is None:
            _default_registry = AudioSubscriberRegistry()
        return _default_registry


def reset_default_audio_subscribers() -> None:
    """For tests — drop the module-level registry."""
    global _default_registry
    with _default_registry_lock:
        _default_registry = None


__all__ = [
    "AudioSubscriberRegistry",
    "get_default_audio_subscribers",
    "reset_default_audio_subscribers",
]
