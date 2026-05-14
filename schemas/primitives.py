"""In-process primitive types — what mod3's inference engines emit.

These are Pydantic models for what the underlying audio models yield.
They are the Python-side dual of the wire-level operation payloads in
:mod:`mod3.schemas.operations`: every TTS engine's streaming generator
emits ``AudioChunk`` instances; the Whisper decoders emit
``TranscriptResult`` and ``PartialTranscript``; the Silero VAD emits
``VADResult``.

Wire-level serialisation goes through :class:`~mod3.schemas.operations.
TTSChunkEvent` etc. — these primitives are the Python representation
the engines work with before encoding into the D2 wire.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Audio chunk — what every TTS engine streams
# ---------------------------------------------------------------------------


class AudioChunk(BaseModel):
    """A single chunk of synthesised audio.

    All mod3 TTS engines (Kokoro, Voxtral, Chatterbox, Spark) produce
    audio at 24000 Hz, mono, float32 in [-1.0, 1.0] internally. The
    wire encoding converts to int16 PCM. Sub-sentence streaming is the
    default; one ``AudioChunk`` per generator iteration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Audio payload — exactly one of these is populated.
    samples_float32: bytes | None = Field(
        default=None,
        description="raw float32 samples as bytes (internal engine format)",
    )
    samples_int16: bytes | None = Field(default=None, description="raw int16 PCM (wire format)")

    # Geometry
    sample_rate: int = 24000
    num_channels: int = 1
    sample_count: int = Field(default=0, ge=0)

    # Provenance
    engine: str = ""
    voice: str = ""

    # Sequencing
    chunk_index: int = Field(default=0, ge=0)
    sentence_index: int = Field(default=0, ge=0)
    is_final: bool = False

    # Per-chunk performance metrics — emitted by mlx-audio
    gen_time_sec: float = 0.0
    rtf: float = Field(default=0.0, description="generation_time / audio_duration")
    peak_memory_gb: float = 0.0
    tokens: int = 0

    @property
    def duration_sec(self) -> float:
        if self.sample_count == 0 or self.sample_rate == 0:
            return 0.0
        return self.sample_count / self.sample_rate


# ---------------------------------------------------------------------------
# STT results — what the Whisper decoders emit
# ---------------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    """One Whisper segment with timing + speech probability."""

    text: str
    start: float = 0.0
    end: float = 0.0
    no_speech_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class TranscriptResult(BaseModel):
    """Full-utterance transcription result (T3-tier / one-shot)."""

    text: str
    language: str = "en"
    duration_sec: float = 0.0
    stt_ms: float = 0.0
    segments: list[TranscriptSegment] = Field(default_factory=list)
    filtered: bool = False
    filter_reason: str = ""


class PartialTranscript(BaseModel):
    """Streaming transcription with confirmed/tentative split (T1/T2 tiers)."""

    confirmed: str = ""
    tentative: str = ""
    full_text: str = ""
    tier: str = "t1"
    changed: bool = False
    filtered: bool = False
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# VAD result — what Silero emits
# ---------------------------------------------------------------------------


class VADResult(BaseModel):
    """Voice-activity-detection result for a chunk of audio."""

    has_speech: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    speech_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    num_segments: int = 0
    total_speech_sec: float = 0.0
    total_audio_sec: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
