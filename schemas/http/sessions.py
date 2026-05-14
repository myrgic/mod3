"""HTTP schema — /v1/sessions/* endpoints (ADR-082)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SessionRegisterRequest(_Base):
    """POST /v1/sessions/register — register a session with the Mod3 communication bus."""

    session_id: str
    participant_id: str
    participant_type: str = Field(default="agent")
    preferred_voice: str | None = Field(default=None)
    preferred_output_device: str = Field(default="system-default")
    priority: int = Field(default=0)


class SessionSubscribersResponse(_Base):
    """GET /v1/sessions/{session_id}/subscribers response."""

    session_id: str
    subscribed: bool
    count: int


class SessionListResponse(_Base):
    """GET /v1/sessions response."""

    sessions: list[Any] = Field(default_factory=list)
    serializer: dict[str, Any] = Field(default_factory=dict)
    voice_pool: Any = None
    voice_holders: Any = None


__all__ = ["SessionListResponse", "SessionRegisterRequest", "SessionSubscribersResponse"]
