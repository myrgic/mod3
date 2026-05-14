"""HTTP schema — POST /v1/vad (file upload VAD check).

The VAD endpoint uses FastAPI's UploadFile for the audio payload, so
no request body model is needed. Only the response shape is defined here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class VadCheckResponse(_Base):
    """POST /v1/vad response."""

    job_id: str
    has_speech: bool
    confidence: float = Field(ge=0.0, le=1.0)
    speech_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    num_segments: int = 0
    total_speech_sec: float = 0.0
    total_audio_sec: float = 0.0
    processing_time_sec: float = 0.0


__all__ = ["VadCheckResponse"]
