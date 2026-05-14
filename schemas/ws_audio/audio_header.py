"""schemas.ws_audio.audio_header — Header frame for /ws/audio/{session_id}.

The per-session audio fan-out channel sends two-frame bursts:

  1. A JSON text frame shaped like ``AudioHeaderFrame``.
  2. A binary frame containing the raw WAV bytes (length = ``bytes`` field).

The browser decodes the WAV blob via ``AudioContext.decodeAudioData``.
Whole-WAV-in-one-blob is the v1 contract; a future revision may stream
PCM chunks using ``seq`` as the forward-compatibility seam.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioHeaderFrame(BaseModel):
    """JSON text frame that precedes each binary WAV payload on /ws/audio/{session_id}.

    Wire shape::

        {
          "type": "audio_header",
          "session_id": "cog-abc123",
          "job_id": "a1b2c3d4",
          "duration_sec": 2.317,
          "sample_rate": 24000,
          "bytes": 111274,
          "format": "wav",
          "seq": 0
        }

    The binary frame that immediately follows carries exactly ``bytes`` raw
    WAV bytes. The client should buffer until the binary frame arrives
    before attempting decode.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: Literal["audio_header"] = "audio_header"
    session_id: str
    job_id: str = ""
    duration_sec: float = Field(default=0.0, ge=0.0)
    sample_rate: int = Field(default=24000, gt=0)
    bytes: int = Field(default=0, ge=0, description="byte length of the following binary frame")
    format: str = Field(default="wav")
    seq: int = Field(default=0, ge=0, description="sequence number for ordering")


__all__ = ["AudioHeaderFrame"]
