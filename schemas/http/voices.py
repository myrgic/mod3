"""HTTP schema — GET /v1/voices and voice-related response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class EngineInfo(_Base):
    """Single engine entry in the /v1/voices response."""

    model_id: str
    voices: list[str]
    default_voice: str
    supports: list[str] = Field(default_factory=list)
    custom_voices: list[str] = Field(default_factory=list)


class VoicesResponse(_Base):
    """GET /v1/voices response."""

    engines: dict[str, Any] = Field(default_factory=dict)


__all__ = ["EngineInfo", "VoicesResponse"]
