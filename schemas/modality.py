"""Canonical modality schemas — Python mirror of ``cogos/pkg/modality``.

These Pydantic models are the wire-level contract between mod3's Python
workers and the CogOS kernel. JSON field names must stay byte-identical
to the Go struct tags in ``pkg/modality/types.go`` and ``pkg/modality/
channel.go``; the wire format is the source of truth and is shared.

For mod3's in-process dataclass equivalents (``CognitiveEvent`` etc. in
``mod3.modality``), use :mod:`mod3.schemas.converters` to round-trip.
The dataclasses are retained for existing call sites; new code should
prefer these schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums — must stay aligned with pkg/modality/types.go
# ---------------------------------------------------------------------------


class ModalityType(str, Enum):
    """Sensory modality identifier."""

    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    SPATIAL = "spatial"


class ModuleStatus(str, Enum):
    """Operational status of a modality module.

    This is the *lifecycle* status the kernel observes for HUD and health
    routing. It is distinct from mod3's internal ``ModuleStatus`` in
    ``mod3.modality`` which tracks runtime activity (idle/encoding/...).
    Both are valid; the wire schema uses this lifecycle vocabulary.
    """

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    CRASHED = "crashed"


# ---------------------------------------------------------------------------
# Core data types — wire-shape Pydantic mirrors
# ---------------------------------------------------------------------------


class CognitiveEvent(BaseModel):
    """A decoded perception — raw signal transformed into meaning."""

    model_config = ConfigDict(populate_by_name=True)

    modality: ModalityType
    channel: str = Field(default="", description="source channel ID")
    content: str = Field(default="", description="decoded content (text, caption)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="when the perception was decoded",
    )


class CognitiveIntent(BaseModel):
    """A desire to act — meaning to be encoded into raw signal."""

    model_config = ConfigDict(populate_by_name=True)

    modality: ModalityType
    channel: str = Field(default="", description="target channel ID")
    content: str = Field(default="", description="content to encode")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="encoder-specific parameters (voice, speed, emotion, ...)",
    )


class EncodedOutput(BaseModel):
    """An encoded intent — raw signal ready for channel delivery."""

    model_config = ConfigDict(populate_by_name=True)

    modality: ModalityType
    data: bytes = Field(default=b"", description="raw encoded bytes (WAV, PNG, ...)")
    mime_type: str = Field(default="", description='MIME type, e.g. "audio/wav"')
    duration: float = Field(
        default=0.0,
        ge=0.0,
        description="duration in seconds (audio/video only); 0 otherwise",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class GateResult(BaseModel):
    """The decision of an input gate — whether raw input merits decoding."""

    model_config = ConfigDict(populate_by_name=True)

    allowed: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ModuleState(BaseModel):
    """Operational state of a modality module — kernel-side HUD view."""

    model_config = ConfigDict(populate_by_name=True)

    status: ModuleStatus
    modality: ModalityType
    pid: int = 0
    uptime: float = Field(default=0.0, ge=0.0, description="uptime in seconds")
    last_error: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
