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


class BodyPlayRequest(_Base):
    """POST /v1/bodies/{body_id}/play — run an agent-authored clip.

    ``clip`` is the clip document itself (see clip.py); agents keep their own
    libraries and send the clip they want. ``args`` fills the clip's declared
    args (clamped to their ranges). The response carries the compiled clip,
    anything dropped for this body, and the body's receipt.
    """

    clip: dict = Field(...)
    args: dict[str, float | str] = Field(default_factory=dict)
    timeout_sec: float = Field(default=2.0, ge=0.1, le=30.0)


class BodyStopRequest(_Base):
    """POST /v1/bodies/{body_id}/stop — stop one clip by name, or all clips."""

    clip: str | None = Field(default=None)
    timeout_sec: float = Field(default=2.0, ge=0.1, le=30.0)
