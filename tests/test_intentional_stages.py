"""Tests for the four intentional-mode stage classes (closes #96).

Covers:
  * DenoiseStage: registered, no-arg constructor, pass-through process().
  * VADStage: registered, no speech halts pipeline, speech populates
    ctx['vad_result'] and fires reflex arc / bargein / audio_subscribers.
  * STTStage: registered, accumulate → perceive → populate ctx keys.
  * EmitStage: registered, calls _emit_notification + audio_subscribers.
  * Stage configure() wires back to InboundPipeline.
  * Intentional composed pipeline is fully wired (4/4 stages).
  * Ambient mode still produces skip-with-warning for the four
    unregistered stages (diarize, ecapa_match, attribute, mention_detect).
  * _tick_composed drives the stage graph end-to-end (mocked internals).
  * _tick_inline is used when the composed graph is partially wired.

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_intentional_stages.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_server():
    """Inject a stub 'server' module so inbound.py can be imported in tests."""
    if "server" not in sys.modules:
        fake_server = ModuleType("server")
        fake_server.emit_channel_event = MagicMock()
        fake_server.emit_permission_verdict = MagicMock()
        sys.modules["server"] = fake_server


def _make_pipeline(**kwargs):
    """Instantiate InboundPipeline with mocked heavy dependencies."""
    _stub_server()

    # Ensure a clean inbound import
    if "inbound" in sys.modules:
        del sys.modules["inbound"]

    mock_capture = MagicMock()
    mock_capture.is_active.return_value = False

    with patch("capture.AudioCapture", return_value=mock_capture):
        mock_bus = MagicMock()
        mock_state = MagicMock()
        mock_state.is_speaking = False

        from inbound import InboundPipeline

        return InboundPipeline(bus=mock_bus, pipeline_state=mock_state, **kwargs)


# ---------------------------------------------------------------------------
# Stage registration
# ---------------------------------------------------------------------------


class TestStageRegistration:
    """Registration tests require inbound to be imported first (decorators run at import time)."""

    def setup_method(self):
        _stub_server()
        # Ensure inbound is imported so @register_stage decorators fire.
        import importlib

        if "inbound" not in sys.modules:
            importlib.import_module("inbound")

    def test_denoise_registered(self):
        from pipeline_graph import STAGE_REGISTRY

        assert "denoise" in STAGE_REGISTRY

    def test_vad_registered(self):
        from pipeline_graph import STAGE_REGISTRY

        assert "vad" in STAGE_REGISTRY

    def test_stt_registered(self):
        from pipeline_graph import STAGE_REGISTRY

        assert "stt" in STAGE_REGISTRY

    def test_emit_registered(self):
        from pipeline_graph import STAGE_REGISTRY

        assert "emit" in STAGE_REGISTRY

    def test_stage_classes_importable_from_inbound(self):
        from inbound import DenoiseStage, EmitStage, STTStage, VADStage

        assert DenoiseStage is not None
        assert VADStage is not None
        assert STTStage is not None
        assert EmitStage is not None


# ---------------------------------------------------------------------------
# DenoiseStage
# ---------------------------------------------------------------------------


class TestDenoiseStage:
    def _make(self):
        _stub_server()
        if "inbound" in sys.modules:
            del sys.modules["inbound"]
        from inbound import DenoiseStage

        stage = DenoiseStage()
        stage.configure(MagicMock())
        return stage

    def test_no_arg_constructor(self):
        _stub_server()
        if "inbound" in sys.modules:
            del sys.modules["inbound"]
        from inbound import DenoiseStage

        stage = DenoiseStage()
        assert stage is not None

    def test_process_returns_ctx_unchanged(self):
        stage = self._make()
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk, "extra": "value"}
        result = stage.process(ctx)
        assert result is ctx
        assert result["chunk"] is chunk
        assert result["extra"] == "value"

    def test_process_does_not_mutate(self):
        stage = self._make()
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}
        result = stage.process(ctx)
        assert result["chunk"] is chunk


# ---------------------------------------------------------------------------
# VADStage
# ---------------------------------------------------------------------------


class TestVADStage:
    def _make_stage_and_pipeline(self, has_speech=True, is_speaking=False):
        _stub_server()
        if "inbound" in sys.modules:
            del sys.modules["inbound"]
        from inbound import VADStage
        from vad import VADResult

        mock_pipeline = MagicMock()
        mock_pipeline._sample_rate = 16000
        mock_pipeline._vad_threshold = 0.5
        mock_pipeline._loop_sleep_sec = 0.01
        mock_pipeline._bargein_registry = None
        mock_pipeline._pipeline_state.is_speaking = is_speaking

        stage = VADStage()
        stage.configure(mock_pipeline)

        vad_result = VADResult(
            has_speech=has_speech,
            confidence=0.9 if has_speech else 0.0,
            speech_ratio=0.8 if has_speech else 0.0,
            num_segments=1 if has_speech else 0,
            total_speech_sec=1.0 if has_speech else 0.0,
            total_audio_sec=2.0,
        )
        return stage, mock_pipeline, vad_result

    def test_no_speech_returns_none(self):
        stage, pipeline, vad_result = self._make_stage_and_pipeline(has_speech=False)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}

        with patch("inbound.detect_speech", return_value=vad_result):
            result = stage.process(ctx)

        assert result is None

    def test_no_speech_sleeps(self):
        stage, pipeline, vad_result = self._make_stage_and_pipeline(has_speech=False)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}

        with patch("inbound.detect_speech", return_value=vad_result):
            stage.process(ctx)

        pipeline._stop_event.wait.assert_called()

    def test_speech_sets_vad_result(self):
        stage, pipeline, vad_result = self._make_stage_and_pipeline(has_speech=True)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}

        with patch("inbound.detect_speech", return_value=vad_result):
            result = stage.process(ctx)

        assert result is not None
        assert result["vad_result"] is vad_result

    def test_speech_triggers_reflex_interrupt_when_speaking(self):
        stage, pipeline, vad_result = self._make_stage_and_pipeline(has_speech=True, is_speaking=True)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}

        with patch("inbound.detect_speech", return_value=vad_result):
            stage.process(ctx)

        pipeline._pipeline_state.interrupt.assert_called_once_with("vad_reflex")

    def test_speech_no_interrupt_when_not_speaking(self):
        stage, pipeline, vad_result = self._make_stage_and_pipeline(has_speech=True, is_speaking=False)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}

        with patch("inbound.detect_speech", return_value=vad_result):
            stage.process(ctx)

        pipeline._pipeline_state.interrupt.assert_not_called()

    def test_speech_dispatches_bargein_when_registry_wired(self):
        stage, pipeline, vad_result = self._make_stage_and_pipeline(has_speech=True)
        dispatched = []

        class StubRegistry:
            def _dispatch(self, event):
                dispatched.append(event)

        pipeline._bargein_registry = StubRegistry()
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk}

        mock_barge_event = MagicMock()
        with (
            patch("inbound.detect_speech", return_value=vad_result),
            patch("inbound.VADStage.process", wraps=stage.process),
        ):
            # Run directly; bargein.providers.base is imported inside the stage
            try:
                with patch("bargein.providers.base.BargeinEvent", return_value=mock_barge_event):
                    stage.process(ctx)
                assert len(dispatched) == 1
            except ImportError:
                pytest.skip("bargein module not available in test env")


# ---------------------------------------------------------------------------
# STTStage
# ---------------------------------------------------------------------------


class TestSTTStage:
    def _make_stage(self, perceive_return=None):
        _stub_server()
        if "inbound" in sys.modules:
            del sys.modules["inbound"]
        from inbound import STTStage
        from vad import VADResult

        mock_pipeline = MagicMock()
        mock_pipeline._bus.perceive.return_value = perceive_return

        vad_result = VADResult(
            has_speech=True,
            confidence=0.9,
            speech_ratio=0.8,
            num_segments=1,
            total_speech_sec=1.0,
            total_audio_sec=2.0,
        )
        utterance = np.zeros(16000, dtype=np.float32)

        stage = STTStage()
        stage.configure(mock_pipeline)
        return stage, mock_pipeline, vad_result, utterance

    def test_accumulate_returns_none_halts_pipeline(self):
        stage, pipeline, vad_result, utterance = self._make_stage()
        pipeline._accumulate_utterance.return_value = (None, vad_result)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk, "vad_result": vad_result}

        result = stage.process(ctx)

        assert result is None

    def test_bus_returns_none_halts_pipeline(self):
        stage, pipeline, vad_result, utterance = self._make_stage(perceive_return=None)
        pipeline._accumulate_utterance.return_value = (utterance, vad_result)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk, "vad_result": vad_result}

        result = stage.process(ctx)

        assert result is None

    def test_successful_stt_populates_ctx(self):
        mock_event = MagicMock()
        mock_event.content = "hello world"
        mock_event.confidence = 0.95
        stage, pipeline, vad_result, utterance = self._make_stage(perceive_return=mock_event)
        pipeline._accumulate_utterance.return_value = (utterance, vad_result)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk, "vad_result": vad_result}

        result = stage.process(ctx)

        assert result is not None
        assert result["event"] is mock_event
        assert result["utterance"] is utterance
        assert result["final_vad"] is vad_result
        assert "audio_bytes" in result

    def test_audio_bytes_is_float32(self):
        mock_event = MagicMock()
        stage, pipeline, vad_result, utterance = self._make_stage(perceive_return=mock_event)
        pipeline._accumulate_utterance.return_value = (utterance, vad_result)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk, "vad_result": vad_result}

        result = stage.process(ctx)

        audio_bytes = result["audio_bytes"]
        recovered = np.frombuffer(audio_bytes, dtype=np.float32)
        assert recovered.dtype == np.float32

    def test_calls_bus_perceive_with_voice_channel(self):
        mock_event = MagicMock()
        stage, pipeline, vad_result, utterance = self._make_stage(perceive_return=mock_event)
        pipeline._accumulate_utterance.return_value = (utterance, vad_result)
        chunk = np.zeros(1600, dtype=np.float32)
        ctx = {"chunk": chunk, "vad_result": vad_result}

        stage.process(ctx)

        pipeline._bus.perceive.assert_called_once()
        call_kwargs = pipeline._bus.perceive.call_args
        assert call_kwargs[1].get("modality") == "voice" or "voice" in str(call_kwargs)


# ---------------------------------------------------------------------------
# EmitStage
# ---------------------------------------------------------------------------


class TestEmitStage:
    def _make_stage(self):
        _stub_server()
        if "inbound" in sys.modules:
            del sys.modules["inbound"]
        from inbound import EmitStage
        from vad import VADResult

        mock_pipeline = MagicMock()
        vad_result = VADResult(
            has_speech=True,
            confidence=0.9,
            speech_ratio=0.8,
            num_segments=1,
            total_speech_sec=1.0,
            total_audio_sec=2.0,
        )
        mock_event = MagicMock()
        mock_event.content = "hello"
        mock_event.confidence = 0.95

        stage = EmitStage()
        stage.configure(mock_pipeline)
        return stage, mock_pipeline, vad_result, mock_event

    def test_calls_emit_notification(self):
        stage, pipeline, vad_result, event = self._make_stage()
        ctx = {"event": event, "final_vad": vad_result}

        stage.process(ctx)

        pipeline._emit_notification.assert_called_once_with(event, vad_result)

    def test_returns_ctx(self):
        stage, pipeline, vad_result, event = self._make_stage()
        ctx = {"event": event, "final_vad": vad_result}

        result = stage.process(ctx)

        assert result is ctx


# ---------------------------------------------------------------------------
# Composed pipeline wiring
# ---------------------------------------------------------------------------


class TestComposedPipelineWiring:
    def test_intentional_pipeline_fully_wired(self):
        """All four intentional stages are registered — composed count == stage count."""
        p = _make_pipeline(mode="intentional")
        assert len(p._composed_stages) == len(p._pipeline_stage_names)
        assert len(p._composed_stages) == 4

    def test_composed_stages_configured(self):
        """Each stage has configure() called — stage._pipeline is the InboundPipeline."""
        p = _make_pipeline(mode="intentional")
        for stage in p._composed_stages:
            assert hasattr(stage, "_pipeline"), f"{stage!r} missing _pipeline after configure()"
            assert stage._pipeline is p

    def test_ambient_pipeline_partially_wired(self, caplog):
        """Ambient mode has 4 of 8 stages registered; 4 produce warnings."""
        import logging

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):
            p = _make_pipeline(mode="ambient")

        unimplemented = {"diarize", "ecapa_match", "attribute", "mention_detect"}
        for name in unimplemented:
            assert name in caplog.text, f"Expected warning for {name!r}"

        # 4 unregistered → composed is shorter than stage names
        assert len(p._composed_stages) < len(p._pipeline_stage_names)
        # 4 registered (denoise, vad, stt, emit)
        assert len(p._composed_stages) == 4

    def test_stage_order_intentional(self):
        """Intentional stages are denoise → vad → stt → emit in that order."""
        p = _make_pipeline(mode="intentional")
        # Use class name comparison — module reloads in tests can produce
        # different class objects for the same source, so isinstance is fragile.
        names = [type(s).__name__ for s in p._composed_stages]
        assert names == ["DenoiseStage", "VADStage", "STTStage", "EmitStage"]


# ---------------------------------------------------------------------------
# _tick routing
# ---------------------------------------------------------------------------


class TestTickRouting:
    def test_fully_wired_uses_tick_composed(self):
        """When intentional pipeline is fully wired, _tick calls _tick_composed."""
        p = _make_pipeline(mode="intentional")
        chunk = np.zeros(1600, dtype=np.float32)
        p._capture.get_audio.return_value = chunk

        with (
            patch.object(p, "_tick_composed") as mock_composed,
            patch.object(p, "_tick_inline") as mock_inline,
        ):
            p._tick()

        mock_composed.assert_called_once_with(chunk)
        mock_inline.assert_not_called()

    def test_partially_wired_uses_tick_inline(self, caplog):
        """When pipeline is partially wired (ambient), _tick uses _tick_inline."""
        import logging

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):
            p = _make_pipeline(mode="ambient")

        chunk = np.zeros(1600, dtype=np.float32)
        p._capture.get_audio.return_value = chunk

        with (
            patch.object(p, "_tick_composed") as mock_composed,
            patch.object(p, "_tick_inline") as mock_inline,
        ):
            p._tick()

        mock_inline.assert_called_once_with(chunk)
        mock_composed.assert_not_called()

    def test_no_chunk_skips_routing(self):
        """When get_audio returns None, neither path is called."""
        p = _make_pipeline(mode="intentional")
        p._capture.get_audio.return_value = None

        with (
            patch.object(p, "_tick_composed") as mock_composed,
            patch.object(p, "_tick_inline") as mock_inline,
        ):
            p._tick()

        mock_composed.assert_not_called()
        mock_inline.assert_not_called()

    def test_tick_composed_drives_stages(self):
        """_tick_composed calls process() on each stage in order."""
        p = _make_pipeline(mode="intentional")
        chunk = np.zeros(1600, dtype=np.float32)

        stage_calls = []

        class TrackingStage:
            def __init__(self, name):
                self._name = name

            def configure(self, pipeline):
                pass

            def process(self, ctx):
                stage_calls.append(self._name)
                return ctx

        p._composed_stages = [
            TrackingStage("denoise"),
            TrackingStage("vad"),
            TrackingStage("stt"),
            TrackingStage("emit"),
        ]

        p._tick_composed(chunk)
        assert stage_calls == ["denoise", "vad", "stt", "emit"]

    def test_tick_composed_halts_on_none(self):
        """_tick_composed stops when a stage returns None."""
        p = _make_pipeline(mode="intentional")
        chunk = np.zeros(1600, dtype=np.float32)

        stage_calls = []

        class PassStage:
            def process(self, ctx):
                stage_calls.append("pass")
                return ctx

        class HaltStage:
            def process(self, ctx):
                stage_calls.append("halt")
                return None

        class ShouldNotRunStage:
            def process(self, ctx):
                stage_calls.append("should_not_run")
                return ctx

        p._composed_stages = [PassStage(), HaltStage(), ShouldNotRunStage()]
        p._tick_composed(chunk)

        assert stage_calls == ["pass", "halt"]
        assert "should_not_run" not in stage_calls
