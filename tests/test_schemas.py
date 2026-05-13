"""Tests for the canonical schemas in ``mod3.schemas``.

These tests pin the wire-level field names, since divergence from the
Go schema in ``cogos/pkg/modality`` would silently break IPC. JSON
field names here must match the Go struct JSON tags byte-for-byte.

Run with: ``.venv/bin/python -m pytest tests/test_schemas.py -v``
"""

from __future__ import annotations

import json

import pytest

from schemas import (
    AudioChunk,
    ChannelDescriptor,
    CognitiveEvent,
    CognitiveIntent,
    EncodedOutput,
    GateResult,
    MAX_WIRE_LINE_SIZE,
    ModalityType,
    ModuleState,
    ModuleStatus,
    PartialTranscript,
    STTStreamingRequest,
    STTStreamingResponse,
    STTTranscribeRequest,
    STTTranscribeResponse,
    TTSChunkEvent,
    TTSStreamRequest,
    TTSSynthesizeRequest,
    TTSSynthesizeResponse,
    TranscriptResult,
    VADDetectRequest,
    VADDetectResponse,
    VADResult,
    WireMessage,
)


# ---------------------------------------------------------------------------
# Wire envelope
# ---------------------------------------------------------------------------


class TestWireMessage:
    def test_request_round_trip(self):
        msg = WireMessage(
            id="tts-1",
            type="request",
            module="tts",
            op="synthesize",
            data={"text": "hello", "voice": "bm_lewis"},
        )
        line = msg.to_jsonl()
        parsed = WireMessage.model_validate_json(line)
        assert parsed.id == "tts-1"
        assert parsed.type == "request"
        assert parsed.module == "tts"
        assert parsed.op == "synthesize"
        assert parsed.data == {"text": "hello", "voice": "bm_lewis"}

    def test_response_round_trip(self):
        msg = WireMessage(
            id="tts-1",
            type="response",
            result={"audio_b64": "AAAA", "duration_sec": 0.5},
        )
        parsed = WireMessage.model_validate_json(msg.to_jsonl())
        assert parsed.result["audio_b64"] == "AAAA"
        assert parsed.result["duration_sec"] == 0.5

    def test_event_round_trip(self):
        msg = WireMessage(id="bootstrap-1", type="event", event="ready", status="ok")
        parsed = WireMessage.model_validate_json(msg.to_jsonl())
        assert parsed.event == "ready"
        assert parsed.status == "ok"

    def test_command_round_trip(self):
        msg = WireMessage(id="cmd-1", type="command", command="shutdown")
        parsed = WireMessage.model_validate_json(msg.to_jsonl())
        assert parsed.command == "shutdown"

    def test_error_round_trip(self):
        msg = WireMessage(
            id="tts-1",
            type="error",
            error="model not loaded",
            error_type="ModelNotReady",
            recoverable=True,
        )
        parsed = WireMessage.model_validate_json(msg.to_jsonl())
        assert parsed.error == "model not loaded"
        assert parsed.error_type == "ModelNotReady"
        assert parsed.recoverable is True

    def test_jsonl_excludes_none_fields(self):
        """Empty fields must be omitted — the Go side uses omitempty."""
        msg = WireMessage(id="x", type="request", module="tts", op="synthesize")
        line = msg.to_jsonl()
        assert "result" not in line
        assert "error" not in line
        assert "command" not in line

    def test_unknown_fields_pass_through(self):
        """Forward-compat: a future field on the Go side mustn't break parse."""
        line = json.dumps(
            {
                "id": "x",
                "type": "request",
                "ts": "2026-05-13T20:00:00Z",
                "module": "tts",
                "op": "synthesize",
                "experimental_field": 42,
            }
        )
        parsed = WireMessage.model_validate_json(line)
        assert parsed.id == "x"

    def test_max_wire_line_size_constant(self):
        assert MAX_WIRE_LINE_SIZE == 1024 * 1024


# ---------------------------------------------------------------------------
# Modality core types — field names must match Go JSON tags exactly
# ---------------------------------------------------------------------------


class TestModalityTypes:
    def test_cognitive_event_field_names(self):
        ev = CognitiveEvent(
            modality=ModalityType.VOICE,
            channel="browser-1",
            content="hello",
            confidence=0.9,
        )
        d = ev.model_dump()
        # Must match Go JSON tags exactly
        assert "modality" in d
        assert "channel" in d
        assert "content" in d
        assert "confidence" in d
        assert "metadata" in d
        assert "timestamp" in d
        # Not "source_channel" (that's the mod3.modality dataclass field name)
        assert "source_channel" not in d

    def test_cognitive_intent_field_names(self):
        intent = CognitiveIntent(
            modality=ModalityType.VOICE,
            channel="browser-1",
            content="say this",
            params={"voice": "bm_lewis", "speed": 1.25},
        )
        d = intent.model_dump()
        assert d["modality"] == "voice"
        assert d["params"]["voice"] == "bm_lewis"

    def test_encoded_output_uses_mime_type(self):
        out = EncodedOutput(modality=ModalityType.VOICE, mime_type="audio/wav", duration=0.5)
        d = out.model_dump()
        # Must be mime_type (Go), not format (mod3.modality dataclass)
        assert "mime_type" in d
        assert "format" not in d

    def test_gate_result_uses_allowed(self):
        gr = GateResult(allowed=True, confidence=0.8, reason="vad")
        d = gr.model_dump()
        # Must be "allowed" (Go), not "passed" (mod3.modality dataclass)
        assert "allowed" in d
        assert "passed" not in d

    def test_module_status_lifecycle_values(self):
        # Lifecycle-flavored statuses (kernel), not activity-flavored (mod3 legacy).
        for s in ["starting", "healthy", "degraded", "stopped", "crashed"]:
            ModuleStatus(s)
        with pytest.raises(ValueError):
            ModuleStatus("idle")  # legacy mod3 activity status — not on wire

    def test_module_state_round_trip(self):
        state = ModuleState(
            status=ModuleStatus.HEALTHY,
            modality=ModalityType.VOICE,
            pid=12345,
            uptime=42.0,
            metrics={"queue_depth": 2},
        )
        parsed = ModuleState.model_validate_json(state.model_dump_json())
        assert parsed.status is ModuleStatus.HEALTHY
        assert parsed.modality is ModalityType.VOICE
        assert parsed.pid == 12345


# ---------------------------------------------------------------------------
# Channel descriptor
# ---------------------------------------------------------------------------


class TestChannelDescriptor:
    def test_dual_modality_channel(self):
        ch = ChannelDescriptor(
            id="mod3-dashboard-abc",
            transport="websocket",
            input=[ModalityType.VOICE, ModalityType.TEXT],
            output=[ModalityType.VOICE, ModalityType.TEXT],
        )
        assert ch.supports_input(ModalityType.VOICE)
        assert ch.supports_output(ModalityType.TEXT)
        assert not ch.supports_input(ModalityType.VISION)

    def test_session_key_pattern(self):
        ch = ChannelDescriptor(
            id="mod3-dashboard-abc",
            transport="websocket",
            session_key="mod3:{session_id}",
        )
        assert ch.session_key == "mod3:{session_id}"


# ---------------------------------------------------------------------------
# Operation request/response — vad/stt/tts contracts
# ---------------------------------------------------------------------------


class TestOperationSchemas:
    def test_vad_detect_round_trip(self):
        req = VADDetectRequest(audio_b64="AAAA", sample_rate=16000)
        resp = VADDetectResponse(has_speech=True, confidence=0.95, speech_ratio=0.8)
        assert req.audio_b64 == "AAAA"
        assert resp.has_speech is True

    def test_stt_transcribe_round_trip(self):
        req = STTTranscribeRequest(audio_b64="AAAA", sample_rate=16000, language="en")
        resp = STTTranscribeResponse(transcript="hello world", confidence=0.92, stt_ms=120.0)
        assert resp.transcript == "hello world"

    def test_stt_streaming_tiers(self):
        for tier in ("t1", "t2"):
            req = STTStreamingRequest(audio_b64="AAAA", tier=tier)
            resp = STTStreamingResponse(
                confirmed="hello",
                tentative="wor",
                tier=tier,
                elapsed_ms=40.0,
                changed=True,
            )
            assert resp.tier == tier

    def test_tts_synthesize_request_defaults(self):
        req = TTSSynthesizeRequest(text="hi")
        # Defaults match the kernel-side voice/speed in modality_voice.go
        assert req.voice == "bm_lewis"
        assert req.speed == 1.25

    def test_tts_synthesize_response(self):
        resp = TTSSynthesizeResponse(
            audio_b64="AAAA",
            duration_sec=0.5,
            sample_rate=24000,
            engine="kokoro",
            voice="bm_lewis",
            gen_time_sec=0.07,
            rtf=0.14,
        )
        assert resp.sample_rate == 24000

    def test_tts_stream_event_shape(self):
        chunk = TTSChunkEvent(
            audio_b64="AAAA",
            sample_rate=24000,
            num_channels=1,
            dtype="int16",
            chunk_index=3,
            sentence_index=1,
            is_final=False,
            gen_time_sec=0.04,
            rtf=0.12,
            engine="kokoro",
            voice="bm_lewis",
        )
        d = chunk.model_dump()
        assert d["sample_rate"] == 24000
        assert d["num_channels"] == 1
        assert d["dtype"] == "int16"
        assert d["chunk_index"] == 3

    def test_operation_round_trip_through_wire(self):
        """Operations must serialise cleanly into a WireMessage envelope."""
        op = TTSSynthesizeRequest(text="hello", voice="bm_lewis", speed=1.25)
        msg = WireMessage(
            id="x", type="request", module="tts", op="synthesize", data=op.model_dump()
        )
        line = msg.to_jsonl()
        parsed = WireMessage.model_validate_json(line)
        reconstructed = TTSSynthesizeRequest(**parsed.data)
        assert reconstructed.text == "hello"
        assert reconstructed.voice == "bm_lewis"


# ---------------------------------------------------------------------------
# Primitives (engine-side Python types)
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_audio_chunk_duration(self):
        chunk = AudioChunk(sample_rate=24000, sample_count=24000)  # 1.0 s
        assert chunk.duration_sec == pytest.approx(1.0)

    def test_audio_chunk_zero_duration_when_empty(self):
        assert AudioChunk().duration_sec == 0.0

    def test_transcript_result_segments(self):
        t = TranscriptResult(
            text="hello world",
            language="en",
            duration_sec=2.0,
            stt_ms=120.0,
        )
        assert t.text == "hello world"
        assert t.segments == []

    def test_partial_transcript_tiers(self):
        p = PartialTranscript(
            confirmed="hello",
            tentative="wor",
            tier="t2",
            changed=True,
            elapsed_ms=300.0,
        )
        assert p.tier == "t2"

    def test_vad_result_defaults(self):
        v = VADResult(has_speech=False, confidence=0.1)
        assert v.has_speech is False
        assert v.metadata == {}
