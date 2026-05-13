"""Mod3TTSService — Pipecat TTSService wrapper for mod3's TTS engines.

Makes mod3 a first-class Pipecat citizen by subclassing ``TTSService`` from
``pipecat.services.tts_service``. The wrapper calls ``engine.generate_audio()``
and yields one ``TTSAudioRawFrame`` per engine-emitted ``AudioChunk``.

Usage::

    from integrations.pipecat.tts_service import Mod3TTSService

    service = Mod3TTSService(voice="bm_lewis", engine="kokoro")
    async for frame in service.run_tts("Hello, world!"):
        # frame is a pipecat TTSAudioRawFrame
        ...

Pipecat is an optional dependency. Users without ``pipecat-ai`` installed can
import mod3 without any error; the ImportError only surfaces when this module
is explicitly imported.
"""

from __future__ import annotations

from typing import AsyncGenerator

import numpy as np

try:
    from pipecat.frames.frames import (
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.services.tts_service import TTSService
except ImportError as _pipecat_err:
    raise ImportError(
        "pipecat-ai is required for Mod3TTSService. "
        "Install it with: pip install mod3[pipecat]"
    ) from _pipecat_err

# Module-level reference so tests can patch via:
#   patch("integrations.pipecat.tts_service.generate_audio", stub)
# The engine import is lazy (mod3 does NOT require mlx_audio at import time
# for non-TTS code paths), so we only import when this integration is used.
try:
    from engine import generate_audio
except ImportError:
    generate_audio = None  # type: ignore[assignment]


class Mod3TTSService(TTSService):
    """Pipecat TTSService backed by mod3's local TTS engines.

    Calls ``engine.generate_audio()`` (the mod3 inference core) and maps
    each yielded ``AudioChunk`` to a Pipecat ``TTSAudioRawFrame``.

    Frame sequence per utterance:
        TTSStartedFrame -> TTSAudioRawFrame (x N chunks) -> TTSStoppedFrame

    All mod3 engines produce float32 samples at 24 kHz, mono. This wrapper
    converts to int16 PCM at the wire boundary, matching Pipecat's wire
    convention for raw audio frames.

    Args:
        voice: mod3 voice identifier, e.g. ``"bm_lewis"``.
        engine: optional engine override (``"kokoro"``, ``"voxtral"``,
            ``"chatterbox"``, ``"spark"``). When None, mod3 resolves the
            engine from the voice name.
        speed: synthesis speed multiplier (default 1.25).
        emotion: emotion exaggeration for Chatterbox (default 0.5).
        streaming_interval: seconds of audio per engine chunk; 0 = default.
        sample_rate: output sample rate in Hz (default 24000; matches all
            current mod3 engines).
        num_channels: output channel count (default 1 = mono).
    """

    def __init__(
        self,
        *,
        voice: str = "bm_lewis",
        engine: str | None = None,
        speed: float = 1.25,
        emotion: float = 0.5,
        streaming_interval: float = 1.0,
        sample_rate: int = 24000,
        num_channels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._voice = voice
        self._engine_override = engine
        self._speed = speed
        self._emotion = emotion
        self._streaming_interval = streaming_interval
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    # ------------------------------------------------------------------
    # TTSService contract
    # ------------------------------------------------------------------

    async def run_tts(
        self, text: str
    ) -> AsyncGenerator[TTSAudioRawFrame | TTSStartedFrame | TTSStoppedFrame, None]:
        """Synthesize ``text`` and yield Pipecat audio frames.

        Yields a ``TTSStartedFrame``, one ``TTSAudioRawFrame`` per engine
        chunk, then a ``TTSStoppedFrame``.

        The engine call is synchronous (mlx_audio runs on the main thread);
        we run it in a thread executor to avoid blocking the event loop.
        """
        import asyncio

        yield TTSStartedFrame()

        loop = asyncio.get_event_loop()

        def _generate():
            """Collect all chunks synchronously in a thread.

            Calls the module-level ``generate_audio`` reference so tests can
            replace it via ``patch("integrations.pipecat.tts_service.generate_audio")``.
            """
            return list(
                generate_audio(
                    text,
                    voice=self._voice,
                    speed=self._speed,
                    emotion=self._emotion,
                    stream=True,
                    streaming_interval=self._streaming_interval,
                )
            )

        chunks = await loop.run_in_executor(None, _generate)

        for chunk in chunks:
            samples = chunk.samples
            if samples is None or len(samples) == 0:
                continue

            pcm_bytes = _float32_to_int16_bytes(samples)

            yield TTSAudioRawFrame(
                audio=pcm_bytes,
                sample_rate=chunk.sample_rate,
                num_channels=self._num_channels,
            )

        yield TTSStoppedFrame()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, value: str) -> None:
        self._voice = value


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _float32_to_int16_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to int16 PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()
