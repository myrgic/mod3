"""D2 wire operation request/response schemas.

Each worker module (``tts``, ``vad``, ``stt``) exposes a small set of
operations. The kernel sends a :class:`~mod3.schemas.wire.WireMessage`
with ``type=request``, ``module=<name>``, ``op=<verb>`` and a typed
``data`` dict; the worker replies with ``type=response`` and a typed
``result`` dict.

The shapes below mirror what ``cogos/modality_voice.go`` sends and
expects. Adding a new operation means: (a) adding the request/response
pair here, (b) handling it in the matching ``mod3/worker/*.py`` module,
(c) wiring the kernel-side caller in Go.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------


class _Op(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------------------------------------------------------------------------
# vad/detect — Voice Activity Detection
# ---------------------------------------------------------------------------


class VADDetectRequest(_Op):
    """Request: should this audio be passed to STT?"""

    audio_b64: str = Field(..., description="base64-encoded PCM audio")
    sample_rate: int = Field(default=16000, description="audio sample rate in Hz")


class VADDetectResponse(_Op):
    """Response: VAD verdict."""

    has_speech: bool
    confidence: float = Field(ge=0.0, le=1.0)
    speech_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    total_speech_sec: float = 0.0
    total_audio_sec: float = 0.0


# ---------------------------------------------------------------------------
# stt/transcribe — Speech to Text
# ---------------------------------------------------------------------------


class STTTranscribeRequest(_Op):
    """Request: full-utterance transcription."""

    audio_b64: str
    sample_rate: int = 16000
    language: str = "en"


class STTTranscribeResponse(_Op):
    """Response: transcription result."""

    transcript: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = "en"
    duration_sec: float = 0.0
    stt_ms: float = Field(default=0.0, description="wall-clock time spent in transcription")


# ---------------------------------------------------------------------------
# stt/transcribe_streaming — rolling LocalAgreement-2 transcription
# ---------------------------------------------------------------------------


class STTStreamingRequest(_Op):
    """Request: tiered streaming transcription on a growing buffer."""

    audio_b64: str
    sample_rate: int = 16000
    tier: str = Field(default="t1", description='"t1" (fast Base) | "t2" (Large)')


class STTStreamingResponse(_Op):
    """Response: confirmed + tentative segments."""

    confirmed: str = Field(default="", description="stable text up to the lock point")
    tentative: str = Field(default="", description="rolling text past the lock point")
    tier: str
    elapsed_ms: float = 0.0
    changed: bool = False
    filtered: bool = Field(default=False, description="suppressed (silence/hallucination)")


# ---------------------------------------------------------------------------
# tts/synthesize — full utterance, blob response
# ---------------------------------------------------------------------------


class TTSSynthesizeRequest(_Op):
    """Request: synthesize a complete utterance."""

    text: str
    voice: str = "bm_lewis"
    speed: float = 1.25
    emotion: float | None = None
    engine: str | None = Field(
        default=None,
        description="optional engine override (kokoro/voxtral/chatterbox/spark)",
    )


class TTSSynthesizeResponse(_Op):
    """Response: full WAV blob plus generation metrics."""

    audio_b64: str
    duration_sec: float
    sample_rate: int = 24000
    engine: str = ""
    voice: str = ""
    gen_time_sec: float = 0.0
    rtf: float = Field(default=0.0, description="real-time factor (gen_time / duration)")


# ---------------------------------------------------------------------------
# tts/stream — sub-sentence streaming chunks (event-based, not request/response)
# ---------------------------------------------------------------------------


class TTSStreamRequest(_Op):
    """Request: synthesize and stream sub-sentence audio chunks.

    The response is a sequence of ``WireMessage`` records with
    ``type=event``, ``event="tts.chunk"``, and a chunk payload in the
    ``data`` field shaped like :class:`TTSChunkEvent`. The final chunk
    sets ``done=True`` on the wire envelope.
    """

    text: str
    voice: str = "bm_lewis"
    speed: float = 1.25
    emotion: float | None = None
    engine: str | None = None
    streaming_interval: float = Field(
        default=0.0,
        description="seconds of audio per chunk; 0 = engine default",
    )


class TTSChunkEvent(_Op):
    """Payload of a streaming TTS chunk event.

    Wraps what the underlying engine generator yields per iteration.
    The audio is raw int16 PCM at ``sample_rate``, mono, base64-encoded.
    """

    audio_b64: str
    sample_rate: int = 24000
    num_channels: int = 1
    dtype: str = "int16"
    chunk_index: int = Field(ge=0)
    sentence_index: int = Field(default=0, ge=0)
    is_final: bool = False
    gen_time_sec: float = 0.0
    rtf: float = 0.0
    peak_memory_gb: float = 0.0
    tokens: int = 0
    samples: int = Field(default=0, ge=0)
    engine: str = ""
    voice: str = ""
