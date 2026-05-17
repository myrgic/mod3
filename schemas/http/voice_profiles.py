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


class ComposeProfileRequest(_Base):
    """POST /v1/voices/profiles/compose — concatenate clips and register.

    Each segment must be a 16-bit WAV at 24 kHz (mono or stereo). Stereo
    sources are downmixed to mono. Segments are joined in the given order
    with `gap_sec` seconds of silence between consecutive entries.
    """

    name: str
    engine: str
    segment_paths: list[str]
    gap_sec: float = Field(default=0.15, ge=0.0, le=2.0)
    exaggeration: float = Field(default=0.5)
    ref_text: str | None = Field(default=None)


class VoiceProfilesResponse(_Base):
    """GET /v1/voices/profiles response."""

    profiles: list[dict] = Field(default_factory=list)


class DeleteProfileResponse(_Base):
    """DELETE /v1/voices/profiles/{name} — success response."""

    deleted: bool


__all__ = [
    "ComposeProfileRequest",
    "DeleteProfileResponse",
    "RegisterProfileRequest",
    "VoiceProfilesResponse",
]
