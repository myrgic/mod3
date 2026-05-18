"""Tests for schemas.http — REST request/response model shapes.

Pins field names, defaults, and JSON serialisation against the wire
contract so any accidental drift in http_api.py endpoints is caught
before it reaches a client (dashboard, MCP shim, curl users).

Run with: ``.venv/bin/python -m pytest tests/test_http_schemas.py -v``
"""

from __future__ import annotations

import json

import pytest

from schemas.http import (
    BusActRequest,
    BusActResponse,
    BusPerceiveResponse,
    DeleteProfileResponse,
    EngineInfo,
    HealthResponse,
    JobListResponse,
    RegisterProfileRequest,
    SessionRegisterRequest,
    SessionSubscribersResponse,
    ShutdownRequest,
    ShutdownResponse,
    SpeechRequest,
    StopResponse,
    SynthesizeRequest,
    VadCheckResponse,
    VadFilterRequest,
    VadFilterResponse,
    VoiceProfilesResponse,
    VoicesResponse,
)

# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesizeRequest:
    def test_defaults(self):
        req = SynthesizeRequest(text="hello")
        assert req.voice == "bm_lewis"
        assert req.speed == 1.25
        assert req.emotion == 0.5
        assert req.format == "wav"
        assert req.session_id is None
        assert req.ref_audio is None

    def test_pcm_format_accepted(self):
        req = SynthesizeRequest(text="hello", format="pcm")
        assert req.format == "pcm"

    def test_invalid_format_rejected(self):
        with pytest.raises(Exception):
            SynthesizeRequest(text="hello", format="mp3")

    def test_session_id_passthrough(self):
        req = SynthesizeRequest(text="hi", session_id="sess-abc")
        assert req.session_id == "sess-abc"

    def test_ref_audio_passthrough(self):
        req = SynthesizeRequest(text="hi", ref_audio="/tmp/ref.wav")
        assert req.ref_audio == "/tmp/ref.wav"

    def test_field_names_in_json(self):
        req = SynthesizeRequest(text="hi")
        d = json.loads(req.model_dump_json())
        for field in ("text", "voice", "speed", "emotion", "format"):
            assert field in d

    def test_extra_fields_allowed(self):
        req = SynthesizeRequest(text="hi", future_field="x")
        assert req.future_field == "x"  # type: ignore[attr-defined]


class TestSpeechRequest:
    def test_defaults(self):
        req = SpeechRequest(input="hello")
        assert req.model == "kokoro"
        assert req.voice == "af_heart"
        assert req.response_format == "mp3"
        assert req.speed == 1.0
        assert req.session_id is None

    def test_field_names(self):
        req = SpeechRequest(input="hello", voice="bm_lewis")
        d = req.model_dump()
        assert "input" in d
        assert "voice" in d
        assert d["voice"] == "bm_lewis"


# ---------------------------------------------------------------------------
# voice_profiles
# ---------------------------------------------------------------------------


class TestRegisterProfileRequest:
    def test_required_fields(self):
        req = RegisterProfileRequest(name="chaz", engine="chatterbox", ref_audio_path="/tmp/ref.wav")
        assert req.name == "chaz"
        assert req.engine == "chatterbox"
        assert req.ref_audio_path == "/tmp/ref.wav"
        assert req.exaggeration == 0.5
        assert req.ref_text is None

    def test_ref_text_optional(self):
        req = RegisterProfileRequest(name="chaz", engine="chatterbox", ref_audio_path="/tmp/ref.wav", ref_text="Hello")
        assert req.ref_text == "Hello"


class TestVoiceProfilesResponse:
    def test_empty(self):
        r = VoiceProfilesResponse()
        assert r.profiles == []

    def test_with_profiles(self):
        r = VoiceProfilesResponse(profiles=[{"name": "chaz", "engine": "chatterbox"}])
        assert len(r.profiles) == 1


class TestDeleteProfileResponse:
    def test_deleted_true(self):
        r = DeleteProfileResponse(deleted=True)
        assert r.deleted is True

    def test_deleted_false(self):
        r = DeleteProfileResponse(deleted=False)
        assert r.deleted is False


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


class TestSessionRegisterRequest:
    def test_defaults(self):
        req = SessionRegisterRequest(session_id="s1", participant_id="cog")
        assert req.participant_type == "agent"
        assert req.preferred_voice is None
        assert req.preferred_output_device == "system-default"
        assert req.priority == 0

    def test_agent_participant_type(self):
        req = SessionRegisterRequest(session_id="s1", participant_id="cog", participant_type="user")
        assert req.participant_type == "user"

    # Wave 6b: identity claims (iss/sub) — backward compatible
    def test_identity_claims_default_none(self):
        """Pre-Wave-6b callers that don't pass iss/sub get None — backward compat."""
        req = SessionRegisterRequest(session_id="s1", participant_id="cog")
        assert req.iss is None
        assert req.sub is None

    def test_identity_claims_present(self):
        req = SessionRegisterRequest(
            session_id="s1",
            participant_id="cog",
            iss="cogos-dev",
            sub="cog",
        )
        assert req.iss == "cogos-dev"
        assert req.sub == "cog"

    def test_identity_claims_in_json(self):
        req = SessionRegisterRequest(
            session_id="s1",
            participant_id="cog",
            iss="cogos-dev",
            sub="cog",
        )
        import json
        d = json.loads(req.model_dump_json())
        assert d["iss"] == "cogos-dev"
        assert d["sub"] == "cog"

    def test_partial_identity_claims(self):
        """sub without iss is valid (sub-only unattributed pattern)."""
        req = SessionRegisterRequest(session_id="s1", participant_id="cog", sub="cog")
        assert req.iss is None
        assert req.sub == "cog"


class TestSessionSubscribersResponse:
    def test_subscribed(self):
        r = SessionSubscribersResponse(session_id="s1", subscribed=True, count=2)
        assert r.subscribed is True
        assert r.count == 2

    def test_not_subscribed(self):
        r = SessionSubscribersResponse(session_id="s1", subscribed=False, count=0)
        d = r.model_dump()
        assert d["subscribed"] is False
        assert d["count"] == 0


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


class TestJobListResponse:
    def test_empty(self):
        r = JobListResponse()
        assert r.jobs == []
        assert r.total == 0

    def test_with_jobs(self):
        r = JobListResponse(jobs=[{"job_id": "abc", "status": "complete"}], total=1)
        assert r.total == 1
        assert r.jobs[0]["job_id"] == "abc"


# ---------------------------------------------------------------------------
# bus
# ---------------------------------------------------------------------------


class TestBusActRequest:
    def test_defaults(self):
        req = BusActRequest()
        assert req.content == ""
        assert req.modality is None
        assert req.channel == ""

    def test_with_content(self):
        req = BusActRequest(content="hello", modality="voice", voice="bm_lewis", speed=1.25)
        assert req.content == "hello"
        assert req.voice == "bm_lewis"


class TestBusActResponse:
    def test_shape(self):
        r = BusActResponse(status="ok", modality="voice", format="wav", duration_sec=1.0, bytes=12345)
        d = r.model_dump()
        assert d["status"] == "ok"
        assert d["bytes"] == 12345
        assert d["metadata"] == {}


class TestBusPerceiveResponse:
    def test_filtered(self):
        r = BusPerceiveResponse(status="filtered", modality="voice", channel="http")
        assert r.event is None

    def test_with_event(self):
        r = BusPerceiveResponse(
            status="ok",
            modality="voice",
            channel="ws",
            event={"content": "hello", "confidence": 0.9},
        )
        assert r.event["content"] == "hello"


# ---------------------------------------------------------------------------
# health / shutdown
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_ok_shape(self):
        r = HealthResponse(
            status="ok",
            service="mod3",
            version="0.4.0",
            uptime_sec=42.0,
            engines={"kokoro": "loaded"},
            modalities={"tts": True, "stt": False, "vad": True},
            queue={"depth": 0, "active_jobs": 0},
        )
        d = r.model_dump()
        assert d["status"] == "ok"
        assert d["engines"]["kokoro"] == "loaded"
        assert d["modalities"]["tts"] is True

    def test_error_shape(self):
        r = HealthResponse(status="error", service="mod3", version="0.4.0", error="boom")
        assert r.error == "boom"

    def test_cogos_agent_enabled_default_false(self):
        r = HealthResponse(status="ok", service="mod3", version="0.4.0")
        assert r.cogos_agent_enabled is False

    def test_cogos_agent_enabled_true(self):
        r = HealthResponse(
            status="ok",
            service="mod3",
            version="0.4.0",
            cogos_agent_enabled=True,
        )
        d = r.model_dump()
        assert d["cogos_agent_enabled"] is True


class TestShutdownRequest:
    def test_defaults(self):
        req = ShutdownRequest()
        assert req.timeout_sec == 5.0
        assert req.reason == "shutdown-requested"

    def test_custom(self):
        req = ShutdownRequest(timeout_sec=10.0, reason="kernel-restart")
        assert req.timeout_sec == 10.0


class TestShutdownResponse:
    def test_shape(self):
        r = ShutdownResponse(status="shutting_down", reason="kernel-restart", timeout_sec=5.0)
        assert r.status == "shutting_down"


class TestStopResponse:
    def test_no_interrupt(self):
        r = StopResponse(status="ok", message="cancelled 3 queued items")
        assert r.interrupted is None

    def test_with_interrupt(self):
        r = StopResponse(
            status="ok",
            message="interrupted",
            interrupted={"spoken_pct": 0.4, "full_text": "hello world"},
        )
        assert r.interrupted["spoken_pct"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# vad / filter
# ---------------------------------------------------------------------------


class TestVadCheckResponse:
    def test_shape(self):
        r = VadCheckResponse(
            job_id="abc123",
            has_speech=True,
            confidence=0.95,
            speech_ratio=0.8,
            num_segments=3,
            total_speech_sec=1.2,
            total_audio_sec=2.0,
            processing_time_sec=0.05,
        )
        assert r.has_speech is True
        assert r.confidence == pytest.approx(0.95)

    def test_defaults(self):
        r = VadCheckResponse(job_id="x", has_speech=False, confidence=0.1)
        assert r.num_segments == 0
        assert r.total_speech_sec == 0.0


class TestVadFilterRequest:
    def test_default_text(self):
        req = VadFilterRequest()
        assert req.text == ""

    def test_with_text(self):
        req = VadFilterRequest(text="thank you")
        assert req.text == "thank you"


class TestVadFilterResponse:
    def test_hallucination(self):
        r = VadFilterResponse(is_hallucination=True, text="thank you")
        assert r.is_hallucination is True

    def test_not_hallucination(self):
        r = VadFilterResponse(is_hallucination=False, text="hello world")
        assert r.is_hallucination is False


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------


class TestEngineInfo:
    def test_shape(self):
        e = EngineInfo(
            model_id="kokoro-v1",
            voices=["af_heart", "bm_lewis"],
            default_voice="bm_lewis",
            supports=["speed", "emotion"],
        )
        assert e.default_voice == "bm_lewis"
        assert "speed" in e.supports

    def test_custom_voices_default_empty(self):
        e = EngineInfo(model_id="x", voices=[], default_voice="y")
        assert e.custom_voices == []


class TestVoicesResponse:
    def test_empty(self):
        r = VoicesResponse()
        assert r.engines == {}
