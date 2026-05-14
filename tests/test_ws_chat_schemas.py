"""Tests for schemas.ws_chat and schemas.ws_audio — WebSocket frame shapes.

Pins field names, discriminators, defaults, and JSON serialisation so
any accidental drift in channels.py is caught before the dashboard
breaks. Also verifies wire format byte-identity: old inline-dict style
must JSON-encode to the same bytes as new model.model_dump(exclude_none=True).

Run with: ``.venv/bin/python -m pytest tests/test_ws_chat_schemas.py -v``
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from schemas.ws_audio import AudioHeaderFrame
from schemas.ws_chat import (
    AudioFrame,
    ConfigFrame,
    DraftQueueFrame,
    EndOfSpeechFrame,
    Frame,
    InboundFrame,
    InterruptedFrame,
    InterruptFrame,
    OutboundFrame,
    PartialTranscriptFrame,
    ResponseCompleteFrame,
    ResponseTextFrame,
    TextMessageFrame,
    TraceEventFrame,
    TranscriptFrame,
    WsErrorDetail,
    WsErrorFrame,
)

_INBOUND = TypeAdapter(InboundFrame)
_OUTBOUND = TypeAdapter(OutboundFrame)


# ---------------------------------------------------------------------------
# Wire format byte-identity — old dict == new model for 5 frame types
# ---------------------------------------------------------------------------


class TestWireByteIdentity:
    def _same(self, old: dict, new: dict) -> bool:
        return json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)

    def test_audio_frame(self):
        old = {"type": "audio", "data": "AAAA", "format": "wav", "duration_sec": 2.32, "sample_rate": 24000}
        new = AudioFrame(type="audio", data="AAAA", format="wav", duration_sec=2.32, sample_rate=24000).model_dump(
            exclude_none=True
        )
        assert self._same(old, new)

    def test_response_text_frame(self):
        old = {"type": "response_text", "text": "Hello world"}
        new = ResponseTextFrame(type="response_text", text="Hello world").model_dump(exclude_none=True)
        assert self._same(old, new)

    def test_transcript_frame(self):
        old = {"type": "transcript", "text": "hi there", "stt_ms": 120.0, "source": "voice"}
        new = TranscriptFrame(type="transcript", text="hi there", stt_ms=120.0, source="voice").model_dump(
            exclude_none=True
        )
        assert self._same(old, new)

    def test_partial_transcript_frame(self):
        old = {
            "type": "partial_transcript",
            "confirmed": "hello",
            "tentative": "wor",
            "tier": "t1",
            "elapsed_ms": 42.0,
        }
        new = PartialTranscriptFrame(
            type="partial_transcript",
            confirmed="hello",
            tentative="wor",
            tier="t1",
            elapsed_ms=42.0,
        ).model_dump(exclude_none=True)
        assert self._same(old, new)

    def test_response_complete_frame(self):
        old = {"type": "response_complete", "metrics": {"llm_ms": 120}}
        new = ResponseCompleteFrame(type="response_complete", metrics={"llm_ms": 120}).model_dump(exclude_none=True)
        assert self._same(old, new)


# ---------------------------------------------------------------------------
# Inbound frames
# ---------------------------------------------------------------------------


class TestInboundFrames:
    def test_end_of_speech_discriminator(self):
        f = _INBOUND.validate_python({"type": "end_of_speech"})
        assert isinstance(f, EndOfSpeechFrame)

    def test_text_message_discriminator(self):
        f = _INBOUND.validate_python({"type": "text_message", "text": "hello"})
        assert isinstance(f, TextMessageFrame)
        assert f.text == "hello"

    def test_interrupt_discriminator(self):
        f = _INBOUND.validate_python({"type": "interrupt"})
        assert isinstance(f, InterruptFrame)

    def test_config_discriminator(self):
        f = _INBOUND.validate_python({"type": "config", "voice": "bm_lewis", "speed": 1.5})
        assert isinstance(f, ConfigFrame)
        assert f.voice == "bm_lewis"
        assert f.speed == pytest.approx(1.5)
        assert f.model is None  # not provided

    def test_config_partial_update(self):
        f = _INBOUND.validate_python({"type": "config", "model": "voxtral"})
        assert isinstance(f, ConfigFrame)
        assert f.model == "voxtral"
        assert f.voice is None

    def test_unknown_type_raises(self):
        with pytest.raises(ValidationError):
            _INBOUND.validate_python({"type": "unknown_frame_type"})

    def test_text_message_empty_text_ok(self):
        # Empty text is valid (the handler strips and skips it)
        f = _INBOUND.validate_python({"type": "text_message"})
        assert isinstance(f, TextMessageFrame)
        assert f.text == ""

    def test_extra_fields_pass_through(self):
        f = _INBOUND.validate_python({"type": "config", "voice": "bm_lewis", "future_param": True})
        assert isinstance(f, ConfigFrame)


# ---------------------------------------------------------------------------
# Outbound frames
# ---------------------------------------------------------------------------


class TestOutboundFrames:
    def test_audio_frame_fields(self):
        f = AudioFrame(type="audio", data="AAAA", format="wav", duration_sec=1.5, sample_rate=24000)
        assert f.type == "audio"
        assert f.data == "AAAA"
        assert f.sample_rate == 24000

    def test_audio_frame_defaults(self):
        f = AudioFrame(type="audio", data="AAAA")
        assert f.format == "wav"
        assert f.duration_sec == 0.0
        assert f.sample_rate == 24000

    def test_response_text_frame(self):
        f = ResponseTextFrame(type="response_text", text="hello")
        d = f.model_dump(exclude_none=True)
        assert d == {"type": "response_text", "text": "hello"}

    def test_response_complete_empty_metrics(self):
        f = ResponseCompleteFrame(type="response_complete")
        assert f.metrics == {}

    def test_transcript_source_default(self):
        f = TranscriptFrame(type="transcript", text="hi")
        assert f.source == "voice"
        assert f.stt_ms == 0.0

    def test_partial_transcript_tiers(self):
        for tier in ("t1", "t2"):
            f = PartialTranscriptFrame(type="partial_transcript", tier=tier)
            assert f.tier == tier
            assert f.confirmed == ""
            assert f.tentative == ""

    def test_interrupted_frame(self):
        f = InterruptedFrame(type="interrupted")
        d = f.model_dump(exclude_none=True)
        assert d == {"type": "interrupted"}

    def test_draft_queue_frame(self):
        f = DraftQueueFrame(type="draft_queue", items=[{"id": "1", "text": "hi"}])
        assert len(f.items) == 1

    def test_trace_event_frame(self):
        ev = {"id": "ev1", "ts": "2026-05-13T00:00:00Z", "kind": "agent_response"}
        f = TraceEventFrame(type="trace_event", event=ev)
        assert f.event["kind"] == "agent_response"
        d = f.model_dump(exclude_none=True)
        assert d["type"] == "trace_event"

    def test_error_frame_shape(self):
        f = WsErrorFrame(
            type="error",
            error=WsErrorDetail(code="stt_failed", message="transcription error"),
        )
        d = f.model_dump(exclude_none=True)
        assert d["type"] == "error"
        assert d["error"]["code"] == "stt_failed"
        assert d["error"]["message"] == "transcription error"
        # data is None — should be excluded
        assert "data" not in d["error"]

    def test_error_frame_with_data(self):
        f = WsErrorFrame(
            type="error",
            error=WsErrorDetail(code="handler_error", message="boom", data={"detail": "stack trace"}),
        )
        d = f.model_dump(exclude_none=True)
        assert d["error"]["data"]["detail"] == "stack trace"

    def test_outbound_discriminator_audio(self):
        f = _OUTBOUND.validate_python(
            {"type": "audio", "data": "AAAA", "format": "wav", "duration_sec": 1.0, "sample_rate": 24000}
        )
        assert isinstance(f, AudioFrame)

    def test_outbound_discriminator_error(self):
        f = _OUTBOUND.validate_python({"type": "error", "error": {"code": "x", "message": "y"}})
        assert isinstance(f, WsErrorFrame)

    def test_frame_alias_is_outbound(self):
        # Frame is the convenience alias for OutboundFrame
        f = TypeAdapter(Frame).validate_python({"type": "interrupted"})
        assert isinstance(f, InterruptedFrame)


# ---------------------------------------------------------------------------
# WsErrorDetail
# ---------------------------------------------------------------------------


class TestWsErrorDetail:
    def test_required_fields(self):
        d = WsErrorDetail(code="stt_failed", message="whisper crashed")
        assert d.code == "stt_failed"
        assert d.message == "whisper crashed"
        assert d.data is None

    def test_optional_data(self):
        d = WsErrorDetail(code="x", message="y", data=[1, 2, 3])
        assert d.data == [1, 2, 3]


# ---------------------------------------------------------------------------
# schemas.ws_audio — AudioHeaderFrame
# ---------------------------------------------------------------------------


class TestAudioHeaderFrame:
    def test_required_fields(self):
        h = AudioHeaderFrame(session_id="sess-1", bytes=12345)
        assert h.type == "audio_header"
        assert h.session_id == "sess-1"
        assert h.bytes == 12345

    def test_defaults(self):
        h = AudioHeaderFrame(session_id="sess-1")
        assert h.job_id == ""
        assert h.duration_sec == 0.0
        assert h.sample_rate == 24000
        assert h.format == "wav"
        assert h.seq == 0

    def test_full_header(self):
        h = AudioHeaderFrame(
            session_id="cog-abc",
            job_id="a1b2c3d4",
            duration_sec=2.317,
            sample_rate=24000,
            bytes=111274,
            format="wav",
            seq=3,
        )
        d = h.model_dump(exclude_none=True)
        assert d["type"] == "audio_header"
        assert d["bytes"] == 111274
        assert d["seq"] == 3

    def test_json_serialisation(self):
        h = AudioHeaderFrame(session_id="s1", job_id="j1", bytes=500, sample_rate=24000, seq=0)
        raw = h.model_dump_json(exclude_none=True)
        parsed = json.loads(raw)
        assert parsed["type"] == "audio_header"
        assert parsed["session_id"] == "s1"

    def test_bytes_field_ge_zero(self):
        with pytest.raises(Exception):
            AudioHeaderFrame(session_id="s", bytes=-1)

    def test_sample_rate_gt_zero(self):
        with pytest.raises(Exception):
            AudioHeaderFrame(session_id="s", sample_rate=0)
