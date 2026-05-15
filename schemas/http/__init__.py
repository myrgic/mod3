"""schemas.http — Pydantic models for every REST endpoint in http_api.py.

One sub-module per logical capability:

* :mod:`.synthesize` — POST /v1/synthesize, POST /v1/audio/speech
* :mod:`.voices`     — GET /v1/voices
* :mod:`.voice_profiles` — POST/GET/DELETE /v1/voices/profiles
* :mod:`.sessions`   — /v1/sessions/* (ADR-082)
* :mod:`.jobs`       — GET /v1/jobs, GET /v1/jobs/{id}
* :mod:`.bus`        — /v1/bus/* (ModalityBus REST surface)
* :mod:`.health`     — GET /health, POST /shutdown, POST /v1/stop
* :mod:`.vad_filter` — POST /v1/vad, POST /v1/filter

All models use ``model_config = ConfigDict(populate_by_name=True, extra="allow")``
for forward-compatibility with fields added by future kernel versions.
"""

from .bus import BusActRequest, BusActResponse, BusPerceiveResponse
from .health import (
    HealthResponse,
    ShutdownRequest,
    ShutdownResponse,
    StopResponse,
    VadFilterRequest,
    VadFilterResponse,
)
from .jobs import JobListResponse
from .sessions import SessionListResponse, SessionRegisterRequest, SessionSubscribersResponse
from .synthesize import SpeakRequest, SpeechRequest, SynthesizeRequest
from .vad_filter import VadCheckResponse
from .voice_profiles import DeleteProfileResponse, RegisterProfileRequest, VoiceProfilesResponse
from .voices import EngineInfo, VoicesResponse

__all__ = [
    # synthesize
    "SpeakRequest",
    "SpeechRequest",
    "SynthesizeRequest",
    # voices
    "EngineInfo",
    "VoicesResponse",
    # voice_profiles
    "DeleteProfileResponse",
    "RegisterProfileRequest",
    "VoiceProfilesResponse",
    # sessions
    "SessionListResponse",
    "SessionRegisterRequest",
    "SessionSubscribersResponse",
    # jobs
    "JobListResponse",
    # bus
    "BusActRequest",
    "BusActResponse",
    "BusPerceiveResponse",
    # health / shutdown / stop
    "HealthResponse",
    "ShutdownRequest",
    "ShutdownResponse",
    "StopResponse",
    "VadFilterRequest",
    "VadFilterResponse",
    # vad file upload
    "VadCheckResponse",
]
