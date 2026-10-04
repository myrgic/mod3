"""HTTP schema — /v1/bodies/* endpoints (agent-driven animated bodies)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class BodyActRequest(_Base):
    """POST /v1/bodies/{body_id}/act — move a connected body.

    ``state`` picks one of the body's declared named states. ``params`` sets
    raw parameters (clamped to the body's declared ranges) and holds them until
    ``release`` hands them back to the body's own animation. ``hold_ms``, when
    given, releases the params automatically after that long.
    """

    state: str | None = Field(default=None)
    params: dict[str, float] = Field(default_factory=dict)
    release: list[str] = Field(default_factory=list)
    hold_ms: int | None = Field(default=None)
    timeout_sec: float = Field(default=2.0, ge=0.1, le=30.0)
