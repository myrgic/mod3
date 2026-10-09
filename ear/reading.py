"""EarReading: the structured result of listening to one chunk of audio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SPEECH = "speech"
MUSIC = "music"
NOISE = "noise"
SILENCE = "silence"

KINDS: tuple[str, ...] = (SPEECH, MUSIC, NOISE, SILENCE)

# Pitch-class names, index 0 = C.
PITCH_CLASSES: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class EarReading:
    """What the ear measured in one chunk.

    ``kind`` is the argmax of ``confidences``; ``confidences`` sums to 1.0 over
    ``KINDS``. Everything else is a measurement, not a judgement.
    """

    kind: str
    confidence: float
    confidences: dict[str, float]

    # level
    rms_db: float  # dBFS over the whole chunk (floor -120)
    peak_db: float  # dBFS sample peak (floor -120)

    # pitch / voicing
    voiced_fraction: float  # share of non-quiet frames with a periodic (pitched) signal
    f0_hz: float | None  # median fundamental over voiced frames, None if unvoiced

    # timbre
    flatness: float  # median spectral flatness over non-quiet frames (0 tonal .. 1 noise)
    band_energy: dict[str, float]  # share of spectral energy in sub / low / mid / high (sums to ~1)
    chroma: list[float]  # 12 pitch-class energies, normalised to sum 1 (C first)
    chroma_entropy: float  # 0 = one pitch class, 1 = uniform

    # speech evidence (Silero VAD, via vad.detect_speech)
    speech_fraction: float  # share of the chunk the VAD marks as speech
    speech_segments: int  # number of speech segments Silero found
    speech_sec: float  # seconds of speech Silero found
    vad_available: bool  # False when Silero could not be loaded: nothing is called speech then

    # bookkeeping
    duration_sec: float
    sample_rate: int  # rate the analysis ran at (after resampling)
    elapsed_ms: float  # wall time spent inside read()
    notes: list[str] = field(default_factory=list)

    @property
    def is_speech(self) -> bool:
        return self.kind == SPEECH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        """One line suitable for logs and for the text of a non-speech event."""
        return f"{self.kind} (confidence {self.confidence:.2f}, {self.rms_db:.0f} dBFS, {self.duration_sec:.1f}s)"
