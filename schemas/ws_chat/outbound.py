"""schemas.ws_chat.outbound — Outbound WebSocket frame models for /ws/chat.

All outbound frames are JSON text. Binary audio is sent separately via
/ws/audio/{session_id} (see schemas.ws_audio).

Outbound frame types
--------------------
* ``audio``              — base64-encoded WAV audio for playback
* ``response_text``      — LLM response text for display
* ``response_complete``  — turn complete signal with metrics
* ``transcript``         — final STT transcript (T3 or text input)
* ``partial_transcript`` — rolling T1/T2 transcript (streaming preview)
* ``interrupted``        — TTS was cut short by barge-in
* ``draft_queue``        — draft queue state update
* ``trace_event``        — kernel cycle-trace event (ADR-083)
* ``error``              — structured error frame (JSON-RPC error shape)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class AudioFrame(_Base):
    """Base64-encoded WAV audio for in-page playback."""

    type: Literal["audio"]
    data: str = Field(..., description="base64-encoded WAV bytes")
    format: str = Field(default="wav")
    duration_sec: float = 0.0
    sample_rate: int = 24000


# ---------------------------------------------------------------------------
# Text / response
# ---------------------------------------------------------------------------


class ResponseTextFrame(_Base):
    """LLM response text — stream-display in the chat panel."""

    type: Literal["response_text"]
    text: str


class ResponseCompleteFrame(_Base):
    """Turn complete — clears the 'isResponding' spinner on the client."""

    type: Literal["response_complete"]
    metrics: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TranscriptFrame(_Base):
    """Final transcript from T3 Whisper or text input."""

    type: Literal["transcript"]
    text: str
    stt_ms: float = Field(default=0.0)
    source: str = Field(default="voice", description='"voice" | "text"')


class PartialTranscriptFrame(_Base):
    """Rolling partial transcript from T1/T2 Whisper (streaming preview)."""

    type: Literal["partial_transcript"]
    confirmed: str = ""
    tentative: str = ""
    tier: str = Field(default="t1", description='"t1" (fast Base) | "t2" (Large)')
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Barge-in / interruption
# ---------------------------------------------------------------------------


class InterruptedFrame(_Base):
    """TTS was interrupted by barge-in."""

    type: Literal["interrupted"]


# ---------------------------------------------------------------------------
# Draft queue
# ---------------------------------------------------------------------------


class DraftQueueFrame(_Base):
    """Draft queue state update."""

    type: Literal["draft_queue"]
    items: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace events (ADR-083)
# ---------------------------------------------------------------------------


class TraceEventFrame(_Base):
    """Kernel cycle-trace event fanned out to all connected dashboards."""

    type: Literal["trace_event"]
    event: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error (JSON-RPC error shape)
# ---------------------------------------------------------------------------


class WsErrorDetail(_Base):
    """Structured error payload inside a WsErrorFrame."""

    code: str
    message: str
    data: Any = None


class WsErrorFrame(_Base):
    """Server-side error on a /ws/chat handler exception.

    Shape follows JSON-RPC error object convention so client error handlers
    have a consistent, inspectable structure. The ``error.code`` field
    carries a machine-readable category (e.g. ``"stt_failed"``,
    ``"handler_error"``).
    """

    type: Literal["error"]
    error: WsErrorDetail


# ---------------------------------------------------------------------------
# Discriminated union — all outbound frame types
# ---------------------------------------------------------------------------

OutboundFrame = Annotated[
    Union[
        AudioFrame,
        ResponseTextFrame,
        ResponseCompleteFrame,
        TranscriptFrame,
        PartialTranscriptFrame,
        InterruptedFrame,
        DraftQueueFrame,
        TraceEventFrame,
        WsErrorFrame,
    ],
    Field(discriminator="type"),
]

__all__ = [
    "AudioFrame",
    "DraftQueueFrame",
    "InterruptedFrame",
    "OutboundFrame",
    "PartialTranscriptFrame",
    "ResponseCompleteFrame",
    "ResponseTextFrame",
    "TraceEventFrame",
    "TranscriptFrame",
    "WsErrorDetail",
    "WsErrorFrame",
]
