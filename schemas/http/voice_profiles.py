"""HTTP schema — POST/GET/DELETE /v1/voices/profiles."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class RegisterProfileRequest(_Base):
    """POST /v1/voices/profiles — register a new voice profile from a reference audio file path."""

    name: str
    engine: str
    ref_audio_path: str
    ref_text: str | None = Field(default=None)
    exaggeration: float = Field(default=0.5)


class VoiceProfilesResponse(_Base):
    """GET /v1/voices/profiles response."""

    profiles: list[dict] = Field(default_factory=list)


class DeleteProfileResponse(_Base):
    """DELETE /v1/voices/profiles/{name} — success response."""

    deleted: bool


__all__ = ["DeleteProfileResponse", "RegisterProfileRequest", "VoiceProfilesResponse"]
