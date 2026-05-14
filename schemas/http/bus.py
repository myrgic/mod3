"""HTTP schema — /v1/bus/* endpoints (ModalityBus REST surface)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class BusActRequest(_Base):
    """POST /v1/bus/act — route a cognitive intent through the bus.

    Resolved modality and channel are optional; when omitted the bus uses
    its registered defaults. Extra keys are forwarded as encoder metadata.
    """

    content: str = Field(default="")
    modality: str | None = Field(default=None)
    channel: str = Field(default="")
    voice: str | None = Field(default=None)
    speed: float | None = Field(default=None)
    emotion: float | None = Field(default=None)


class BusActResponse(_Base):
    """POST /v1/bus/act response."""

    status: str
    modality: str
    format: str
    duration_sec: float
    bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class BusPerceiveResponse(_Base):
    """POST /v1/bus/perceive response — filtered or decoded event."""

    status: str
    modality: str = ""
    channel: str = ""
    event: dict[str, Any] | None = None


__all__ = ["BusActRequest", "BusActResponse", "BusPerceiveResponse"]
