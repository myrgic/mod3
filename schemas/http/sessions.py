"""HTTP schema — /v1/sessions/* endpoints (ADR-082)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SessionRegisterRequest(_Base):
    """POST /v1/sessions/register — register a session with the Mod3 communication bus.

    Identity claims (Wave 6b / ADR-082):
        iss: OIDC issuer for the user identity registering this session.
             Typically the CogOS kernel issuer (e.g. "cogos-dev").
        sub: OIDC subject — the user identity slug (e.g. "chaz").

        Both fields are optional. When absent, the seat is treated as an
        unattributed seat (backward-compatible with pre-Wave-6b callers).
        When present, mod3 emits a ``presence.started`` event so the kernel
        reconciler can update the seat's identity binding.

    Multi-identity harness binding (Wave 6c / Primitive 2):
        Agentic harnesses (Claude Code, Cursor) bind TWO identities simultaneously:
            iss / sub             → user identity (human operator, e.g. "chaz")
            assistant_iss / sub   → agent identity (LLM entity, e.g. "cog")

        This makes agentic harnesses first-class on the substrate. The
        ``participant_id`` field remains the human-readable user slug for
        backward compatibility.

        Trust model (v1): identity claims are accepted at face value from the
        registering process. The launch context (stdio child of a CogOS-managed
        kernel process) is the implicit attestation; no cryptographic verification
        is performed here. The ``HarnessBindingCRD`` record written on
        ``presence.started`` is the substrate-readable audit trail. Cryptographic
        enforcement (kernel-signed tokens, pre-check against ratified
        ``HarnessBinding`` records) is v2 work.
    """

    session_id: str
    participant_id: str
    participant_type: str = Field(default="agent")
    preferred_voice: str | None = Field(default=None)
    preferred_output_device: str = Field(default="system-default")
    priority: int = Field(default=0)
    # Wave 6b: user identity OIDC claims — optional, backward compatible
    iss: str | None = Field(default=None, description="OIDC issuer for the user identity")
    sub: str | None = Field(default=None, description="OIDC subject slug for user identity (e.g. 'chaz')")
    # Wave 6c / Primitive 2: agent identity claims (agentic harnesses only)
    # When set, this seat is bound to BOTH the user identity (iss/sub) and the
    # agent identity (assistant_iss/assistant_sub). Non-agentic clients leave
    # these None.
    assistant_iss: str | None = Field(default=None, description="OIDC issuer for agent identity")
    assistant_sub: str | None = Field(default=None, description="OIDC subject slug for agent identity (e.g. 'cog')")
    # Primitive 4: channel pipeline mode — optional, backward compatible.
    # "intentional" (default) = session-scoped, explicit participation.
    # "ambient" = always-on, VAD-gated, continuous diarization.
    channel_mode: str = Field(
        default="intentional",
        description="Pipeline composition mode. 'intentional' (default) or 'ambient'.",
    )


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
