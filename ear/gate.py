"""EarGate: the inbound gate that asks "what kind of sound is this?" before STT.

Only ``kind == speech`` passes. Music, noise and silence are rejected with a
``GateResult`` whose metadata says what was heard, so the caller can report that
instead of inventing a transcript.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Any

import numpy as np

from modality import Gate, GateResult

from .core import read_clip
from .reading import SPEECH, EarReading

logger = logging.getLogger("mod3.ear")


def decode_raw(raw: bytes, sample_rate: int = 16000, sample_width: int = 4) -> tuple[np.ndarray, int]:
    """Turn the byte payloads mod3 passes around into (float32 mono samples, rate).

    - A RIFF/WAV container is parsed (its own rate wins).
    - ``sample_width=2``: int16 little-endian PCM.
    - ``sample_width=4`` (default): float32 PCM. Every ``bus.perceive`` caller in
      mod3 sends float32 (``audio.astype(np.float32).tobytes()``).
    """
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        with wave.open(io.BytesIO(raw), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width == 2:
            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return x, sr
    if sample_width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        usable = len(raw) - (len(raw) % 4)
        x = np.frombuffer(raw[:usable], dtype=np.float32)
    return x, sample_rate


def gate_result_from_reading(reading: EarReading, **extra: Any) -> GateResult:
    """Build the GateResult for a reading. ``passed`` is true only for speech."""
    passed = reading.kind == SPEECH
    meta: dict[str, Any] = {
        # keys the pre-ear VoiceGate reported; /v1/vad and the MCP vad_check read them
        "speech_ratio": reading.speech_fraction,
        "num_segments": reading.speech_segments,
        "total_speech_sec": reading.speech_sec,
        "total_audio_sec": reading.duration_sec,
        # what the ear heard
        "kind": reading.kind,
        "kind_confidence": reading.confidence,
        "confidences": reading.confidences,
        "reading": reading.to_dict(),
    }
    meta.update(extra)
    return GateResult(
        passed=passed,
        confidence=reading.confidence if passed else reading.confidences.get(SPEECH, 0.0),
        reason=f"kind={reading.kind} confidence={reading.confidence:.2f} speech_ratio={reading.speech_fraction}",
        metadata=meta,
    )


class EarGate(Gate):
    """Gate on the shared ear's ``kind``. Pass speech; stop everything else.

    ``threshold`` is the Silero speech-probability threshold handed to the ear
    (same meaning it had on the pre-ear VAD gate).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fallback_check(self, raw: bytes, **kwargs) -> GateResult | None:
        """Used when the ear itself errors. Base class has none: fail closed."""
        return None

    def check(self, raw: bytes, **kwargs) -> GateResult:
        sample_rate = int(kwargs.get("sample_rate", 16000))
        sample_width = int(kwargs.get("sample_width", 4))
        try:
            samples, sr = decode_raw(raw, sample_rate=sample_rate, sample_width=sample_width)
            reading = read_clip(samples, sr, vad_threshold=self.threshold)
            if not reading.vad_available:
                raise RuntimeError("speech detector (Silero VAD) unavailable")
        except Exception as exc:  # noqa: BLE001 - the ear must never take the input path down
            logger.warning("ear gate failed (%s); falling back", exc)
            try:
                fb = self.fallback_check(raw, **kwargs)
            except Exception as fb_exc:  # noqa: BLE001
                logger.warning("ear gate fallback failed too (%s); rejecting input", fb_exc)
                fb = None
            if fb is not None:
                return fb
            return GateResult(passed=False, confidence=0.0, reason=f"ear_error: {exc}", metadata={"kind": "unknown"})
        return gate_result_from_reading(reading)
