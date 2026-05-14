"""HTTP schema — GET /v1/jobs and GET /v1/jobs/{job_id}."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class JobListResponse(_Base):
    """GET /v1/jobs response."""

    jobs: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0


__all__ = ["JobListResponse"]
