"""HTTP schema — /v1/sessions/* endpoints (ADR-082)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SessionRegisterRequest(_Base):
    """POST /v1/sessions/register — register a session with the Mod3 communication bus.

    Identity claims (Wave 6b / ADR-082):
        iss: OIDC issuer for the principal registering this session.
             Typically the CogOS kernel issuer (e.g. "cogos-dev").
        sub: OIDC subject — the stable identity slug (e.g. "chaz", "cog").

        Both fields are optional. When absent, the seat is treated as an
        unattributed seat (backward-compatible with pre-Wave-6b callers).
        When present, mod3 emits a ``presence.started`` event so the kernel
        reconciler can update the seat's identity binding.

        Agentic sessions (Claude Code, Cursor) bind two identities simultaneously:
            participant_id  → user identity (human operator)
            iss + sub       → agent identity (LLM-shaped substrate entity)
        This is the multi-identity harness shape per feedback_agentic_harness_multi_identity.
    """

    session_id: str
    participant_id: str
    participant_type: str = Field(default="agent")
    preferred_voice: str | None = Field(default=None)
    preferred_output_device: str = Field(default="system-default")
    priority: int = Field(default=0)
    # Identity claims (Wave 6b) — optional, backward compatible
    iss: str | None = Field(default=None, description="OIDC issuer for the registering identity")
    sub: str | None = Field(default=None, description="OIDC subject slug (e.g. 'cog', 'chaz')")


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
