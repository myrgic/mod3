"""HTTP schema — /v1/voices/compositions surface."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SegmentSpec(_Base):
    path: str
    label: str = ""
    duration_sec: float | None = None


class CompositionCreateRequest(_Base):
    """POST /v1/voices/compositions — create a new draft."""

    name: str
    segments: list[SegmentSpec] = Field(default_factory=list)
    engine: str = "chatterbox-turbo"
    exaggeration: float = Field(default=0.5)
    gap_sec: float = Field(default=0.15, ge=0.0, le=2.0)
    notes: str = ""


class CompositionUpdateRequest(_Base):
    """PATCH /v1/voices/compositions/{name} — partial update."""

    segments: list[SegmentSpec] | None = None
    engine: str | None = None
    exaggeration: float | None = None
    gap_sec: float | None = Field(default=None, ge=0.0, le=2.0)
    notes: str | None = None


class CompositionResponse(_Base):
    name: str
    segments: list[SegmentSpec]
    engine: str
    exaggeration: float
    gap_sec: float
    notes: str
    created_at: str
    updated_at: str


class CompositionListResponse(_Base):
    compositions: list[CompositionResponse] = Field(default_factory=list)


__all__ = [
    "CompositionCreateRequest",
    "CompositionListResponse",
    "CompositionResponse",
    "CompositionUpdateRequest",
    "SegmentSpec",
]
