"""Tests for Mod3TTSService Pipecat wrapper.

Skips cleanly if pipecat-ai is not installed.
Mocks engine.generate_audio so no MLX models are loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the entire module if pipecat-ai is not installed.
pipecat = pytest.importorskip("pipecat", reason="pipecat-ai not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_chunk(samples_float32: np.ndarray, sample_rate: int = 24000, is_final: bool = False):
    """Build a minimal fake AudioChunk matching the engine.AudioChunk dataclass."""
    chunk = MagicMock()
    chunk.samples = samples_float32
    chunk.sample_rate = sample_rate
    chunk.metadata = {
        "gen_time_sec": 0.01,
        "rtf": 0.1,
        "samples": len(samples_float32),
        "tokens": 3,
        "is_final": is_final,
        "sentence": 0,
        "peak_memory_gb": 0.0,
        "engine": "kokoro",
    }
    return chunk


def _make_fake_engine(num_chunks: int = 2):
    """Return a generate_audio stub that yields ``num_chunks`` fake chunks."""

    def _generate(text, **kwargs):
        for i in range(num_chunks):
            samples = np.zeros(2400, dtype=np.float32)  # 0.1s of silence at 24kHz
            yield _make_fake_chunk(samples, is_final=(i == num_chunks - 1))

    return _generate


# ---------------------------------------------------------------------------
# Import the service (under pipecat guard)
# ---------------------------------------------------------------------------

from integrations.pipecat.tts_service import Mod3TTSService, _float32_to_int16_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_float32_to_int16_conversion():
    """Verify float32 samples are correctly converted to int16 PCM bytes."""
    samples = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    result = _float32_to_int16_bytes(samples)
    assert isinstance(result, bytes)
    assert len(result) == 8  # 4 samples * 2 bytes each
    int16_arr = np.frombuffer(result, dtype=np.int16)
    assert int16_arr[0] == 0
    assert int16_arr[1] == 32767
    assert int16_arr[2] == -32767
    assert int16_arr[3] == pytest.approx(16383, abs=2)


def test_mod3_tts_service_instantiates():
    """Mod3TTSService can be created without errors."""
    service = Mod3TTSService(voice="bm_lewis", speed=1.0)
    assert service.voice == "bm_lewis"
    assert service.sample_rate == 24000


@pytest.mark.asyncio
async def test_run_tts_yields_frames():
    """run_tts should yield TTSStartedFrame, at least one audio frame, then TTSStoppedFrame."""
    from pipecat.frames.frames import TTSAudioRawFrame

    service = Mod3TTSService(voice="bm_lewis")

    with patch("integrations.pipecat.tts_service.generate_audio", _make_fake_engine(num_chunks=2)):
        frames = []
        async for frame in service.run_tts("Hello world"):
            frames.append(frame)

    types = [type(f).__name__ for f in frames]
    assert types[0] == "TTSStartedFrame", f"expected TTSStartedFrame first, got {types}"
    assert types[-1] == "TTSStoppedFrame", f"expected TTSStoppedFrame last, got {types}"

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1, "expected at least one TTSAudioRawFrame"


@pytest.mark.asyncio
async def test_run_tts_audio_frame_shape():
    """Each TTSAudioRawFrame should have correct sample_rate, num_channels, and non-empty audio."""
    from pipecat.frames.frames import TTSAudioRawFrame

    service = Mod3TTSService(voice="bm_lewis", sample_rate=24000)

    with patch("integrations.pipecat.tts_service.generate_audio", _make_fake_engine(num_chunks=1)):
        frames = []
        async for frame in service.run_tts("Hi"):
            frames.append(frame)

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 1

    frame = audio_frames[0]
    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert len(frame.audio) > 0


@pytest.mark.asyncio
async def test_run_tts_empty_chunk_skipped():
    """Chunks with zero-length samples should be skipped (no frame emitted)."""
    from pipecat.frames.frames import TTSAudioRawFrame

    def _gen_with_empty(text, **kwargs):
        # First chunk: zero samples (gap chunk, should be skipped)
        empty = np.zeros(0, dtype=np.float32)
        yield _make_fake_chunk(empty)
        # Second chunk: real audio
        real = np.zeros(2400, dtype=np.float32)
        yield _make_fake_chunk(real, is_final=True)

    service = Mod3TTSService(voice="bm_lewis")

    with patch("integrations.pipecat.tts_service.generate_audio", _gen_with_empty):
        frames = []
        async for frame in service.run_tts("Hi"):
            frames.append(frame)

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    # Empty chunk should be filtered; only one real audio frame expected
    assert len(audio_frames) == 1


@pytest.mark.asyncio
async def test_run_tts_voice_setter():
    """Voice can be changed after construction."""
    service = Mod3TTSService(voice="bm_lewis")
    service.voice = "af_heart"
    assert service.voice == "af_heart"
