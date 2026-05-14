"""HTTP schema — GET /health, GET /capabilities, GET /diagnostics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class HealthResponse(_Base):
    """GET /health — standardized CogOS service health format."""

    status: str  # "ok" | "degraded" | "error"
    service: str
    version: str
    uptime_sec: float = 0.0
    engines: dict[str, str] = Field(default_factory=dict)
    modalities: dict[str, bool] = Field(default_factory=dict)
    queue: dict[str, int] = Field(default_factory=dict)
    error: str | None = Field(default=None)


class ShutdownRequest(_Base):
    """POST /shutdown — graceful shutdown request from the kernel."""

    timeout_sec: float = Field(default=5.0, ge=0, le=60)
    reason: str = Field(default="shutdown-requested")


class ShutdownResponse(_Base):
    """POST /shutdown response."""

    status: str
    reason: str
    timeout_sec: float


class StopResponse(_Base):
    """POST /v1/stop response."""

    status: str
    message: str
    interrupted: dict[str, Any] | None = Field(default=None)


class VadFilterRequest(_Base):
    """POST /v1/filter — check if a transcription is a known Whisper hallucination."""

    text: str = Field(default="")


class VadFilterResponse(_Base):
    """POST /v1/filter response."""

    is_hallucination: bool
    text: str


__all__ = [
    "HealthResponse",
    "ShutdownRequest",
    "ShutdownResponse",
    "StopResponse",
    "VadFilterRequest",
    "VadFilterResponse",
]
