"""Tests for pipeline_graph.py — ChannelMode, STAGE_REGISTRY, DEFAULT_PIPELINES.

Covers:
  * ChannelMode enum values and str-enum contract.
  * DEFAULT_PIPELINES: intentional vs ambient stage lists.
  * register_stage decorator: round-trips through STAGE_REGISTRY.
  * compose_stages: instantiates registered stages; skips + warns on missing.
  * resolve_pipeline: mode default vs caller override.
  * InboundPipeline: mode/pipeline_stages constructor args + backward compat.
  * Seat: channel_mode field default + explicit ambient.
  * SeatRegistry.register: channel_mode kwarg flows through to Seat.
  * SessionRegisterRequest: channel_mode field default + explicit ambient.

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_channel_pipeline_graph.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from pipeline_graph import (  # noqa: E402
    DEFAULT_PIPELINES,
    STAGE_REGISTRY,
    ChannelMode,
    compose_stages,
    register_stage,
    resolve_pipeline,
)

# ---------------------------------------------------------------------------
# ChannelMode
# ---------------------------------------------------------------------------


class TestChannelMode:
    def test_intentional_value(self):
        assert ChannelMode.INTENTIONAL == "intentional"

    def test_ambient_value(self):
        assert ChannelMode.AMBIENT == "ambient"

    def test_is_str_enum(self):
        """ChannelMode is a str subclass so it compares equal to plain strings."""
        assert ChannelMode.INTENTIONAL == "intentional"
        assert ChannelMode.AMBIENT == "ambient"

    def test_from_string(self):
        assert ChannelMode("intentional") is ChannelMode.INTENTIONAL
        assert ChannelMode("ambient") is ChannelMode.AMBIENT

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError):
            ChannelMode("holographic")


# ---------------------------------------------------------------------------
# DEFAULT_PIPELINES
# ---------------------------------------------------------------------------


class TestDefaultPipelines:
    def test_intentional_stages(self):
        stages = DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]
        assert stages == ["denoise", "vad", "stt", "emit"]

    def test_ambient_stages(self):
        stages = DEFAULT_PIPELINES[ChannelMode.AMBIENT]
        # Must contain the full pipeline in the correct order.
        assert stages[0] == "denoise"
        assert "vad" in stages
        assert "diarize" in stages
        assert "ecapa_match" in stages
        assert "stt" in stages
        assert "attribute" in stages
        assert "mention_detect" in stages
        assert stages[-1] == "emit"

    def test_ambient_stages_longer_than_intentional(self):
        assert len(DEFAULT_PIPELINES[ChannelMode.AMBIENT]) > len(
            DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]
        )


# ---------------------------------------------------------------------------
# register_stage
# ---------------------------------------------------------------------------


class TestRegisterStage:
    def setup_method(self):
        """Save and clear the global registry for test isolation."""
        self._saved = dict(STAGE_REGISTRY)
        STAGE_REGISTRY.clear()

    def teardown_method(self):
        """Restore the registry."""
        STAGE_REGISTRY.clear()
        STAGE_REGISTRY.update(self._saved)

    def test_registers_class_by_name(self):
        @register_stage("fake_vad")
        class FakeVAD:
            pass

        assert "fake_vad" in STAGE_REGISTRY
        assert STAGE_REGISTRY["fake_vad"] is FakeVAD

    def test_decorator_returns_class_unchanged(self):
        @register_stage("passthrough")
        class PT:
            value = 42

        assert PT.value == 42

    def test_overwrite_logs_warning(self, caplog):
        import logging

        @register_stage("dup")
        class First:
            pass

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):

            @register_stage("dup")
            class Second:
                pass

        assert "overwriting" in caplog.text.lower() or "dup" in caplog.text


# ---------------------------------------------------------------------------
# compose_stages
# ---------------------------------------------------------------------------


class TestComposeStages:
    def setup_method(self):
        self._saved = dict(STAGE_REGISTRY)
        STAGE_REGISTRY.clear()

    def teardown_method(self):
        STAGE_REGISTRY.clear()
        STAGE_REGISTRY.update(self._saved)

    def test_instantiates_registered_stage(self):
        @register_stage("my_stage")
        class MyStage:
            pass

        result = compose_stages(["my_stage"])
        assert len(result) == 1
        assert isinstance(result[0], MyStage)

    def test_skips_unregistered_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):
            result = compose_stages(["not_registered"])

        assert len(result) == 0
        assert "not_registered" in caplog.text

    def test_mixed_registered_and_unregistered(self, caplog):
        import logging

        @register_stage("real_stage")
        class RealStage:
            pass

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):
            result = compose_stages(["real_stage", "ghost_stage"])

        assert len(result) == 1
        assert isinstance(result[0], RealStage)
        assert "ghost_stage" in caplog.text

    def test_empty_list_returns_empty(self):
        assert compose_stages([]) == []


# ---------------------------------------------------------------------------
# resolve_pipeline
# ---------------------------------------------------------------------------


class TestResolvePipeline:
    def test_mode_intentional_default(self):
        stages = resolve_pipeline(ChannelMode.INTENTIONAL)
        assert stages == DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]

    def test_mode_ambient_default(self):
        stages = resolve_pipeline(ChannelMode.AMBIENT)
        assert stages == DEFAULT_PIPELINES[ChannelMode.AMBIENT]

    def test_pipeline_stages_override(self):
        override = ["vad", "stt", "emit"]
        stages = resolve_pipeline(ChannelMode.AMBIENT, pipeline_stages=override)
        assert stages == override

    def test_string_mode_intentional(self):
        stages = resolve_pipeline("intentional")
        assert stages == DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]

    def test_string_mode_ambient(self):
        stages = resolve_pipeline("ambient")
        assert stages == DEFAULT_PIPELINES[ChannelMode.AMBIENT]

    def test_unknown_string_mode_defaults_to_intentional(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):
            stages = resolve_pipeline("holographic")

        assert stages == DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]
        assert "intentional" in caplog.text.lower() or "holographic" in caplog.text

    def test_returns_copy_not_reference(self):
        stages = resolve_pipeline(ChannelMode.INTENTIONAL)
        stages.append("injected")
        assert DEFAULT_PIPELINES[ChannelMode.INTENTIONAL][-1] != "injected"


# ---------------------------------------------------------------------------
# InboundPipeline constructor — mode + pipeline_stages args
# ---------------------------------------------------------------------------


class TestInboundPipelineMode:
    """Tests for Primitive 4 additions to InboundPipeline.__init__."""

    def _make_pipeline(self, **kwargs):
        """Create an InboundPipeline with mocked heavy dependencies.

        server.py's emit_channel_event is a coroutine that requires an active
        MCP session — stub it at import time via sys.modules so InboundPipeline
        can be instantiated in tests without a live server.
        """
        import sys
        from types import ModuleType

        # Stub out the 'server' module if not already present
        if "server" not in sys.modules:
            fake_server = ModuleType("server")
            fake_server.emit_channel_event = MagicMock()
            fake_server.emit_permission_verdict = MagicMock()
            sys.modules["server"] = fake_server

        # Also ensure capture + vad are importable
        mock_capture_inst = MagicMock()
        mock_capture_inst.is_active.return_value = False

        with (
            patch("capture.AudioCapture", return_value=mock_capture_inst),
        ):
            mock_bus = MagicMock()
            mock_state = MagicMock()

            # Re-import inbound fresh so our sys.modules stub takes effect
            if "inbound" in sys.modules:
                del sys.modules["inbound"]

            from inbound import InboundPipeline

            pipeline = InboundPipeline(bus=mock_bus, pipeline_state=mock_state, **kwargs)
            return pipeline

    def test_default_mode_is_intentional(self):
        p = self._make_pipeline()
        assert p._channel_mode == ChannelMode.INTENTIONAL

    def test_ambient_mode_selects_ambient_stages(self):
        p = self._make_pipeline(mode=ChannelMode.AMBIENT)
        assert p._channel_mode == ChannelMode.AMBIENT
        assert p._pipeline_stage_names == DEFAULT_PIPELINES[ChannelMode.AMBIENT]

    def test_explicit_pipeline_stages_override(self):
        custom = ["vad", "stt", "emit"]
        p = self._make_pipeline(pipeline_stages=custom)
        assert p._pipeline_stage_names == custom

    def test_string_mode_intentional(self):
        p = self._make_pipeline(mode="intentional")
        assert p._channel_mode == ChannelMode.INTENTIONAL

    def test_string_mode_ambient(self):
        p = self._make_pipeline(mode="ambient")
        assert p._channel_mode == ChannelMode.AMBIENT

    def test_backward_compat_no_mode_arg(self):
        """Callers that pass no mode arg get intentional pipeline (no crash)."""
        p = self._make_pipeline()
        assert p._pipeline_stage_names == DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]

    def test_ambient_missing_stages_logs_warning(self, caplog):
        """Ambient stages not yet implemented produce warnings, not errors."""
        import logging

        with caplog.at_level(logging.WARNING, logger="mod3.pipeline_graph"):
            p = self._make_pipeline(mode=ChannelMode.AMBIENT)

        # diarize, ecapa_match, attribute, mention_detect are not registered
        # — each should produce a warning log.
        unimplemented = {"diarize", "ecapa_match", "attribute", "mention_detect"}
        for stage in unimplemented:
            assert stage in caplog.text, f"Expected warning for unregistered stage {stage!r}"

        # Pipeline must not crash — composed_stages may be shorter than stage_names.
        assert len(p._composed_stages) <= len(p._pipeline_stage_names)


# ---------------------------------------------------------------------------
# Seat — channel_mode field
# ---------------------------------------------------------------------------


class TestSeatChannelMode:
    def test_default_channel_mode_is_intentional(self):
        from seats import Seat

        seat = Seat(
            seat_id="s1",
            session_id="sess",
            client_type="generic",
            device_uuid="dev-1",
        )
        assert seat.channel_mode == "intentional"

    def test_ambient_channel_mode(self):
        from seats import Seat

        seat = Seat(
            seat_id="s2",
            session_id="sess",
            client_type="generic",
            device_uuid="dev-2",
            channel_mode="ambient",
        )
        assert seat.channel_mode == "ambient"

    def test_to_dict_includes_channel_mode(self):
        from seats import Seat

        seat = Seat(
            seat_id="s3",
            session_id="sess",
            client_type="generic",
            device_uuid="dev-3",
            channel_mode="ambient",
        )
        d = seat.to_dict()
        assert d["channel_mode"] == "ambient"


# ---------------------------------------------------------------------------
# SeatRegistry.register — channel_mode kwarg
# ---------------------------------------------------------------------------


class TestSeatRegistryChannelMode:
    def test_register_defaults_to_intentional(self):
        from seats import SeatRegistry

        reg = SeatRegistry()
        seat = reg.register("sess", "generic", "dev-uuid")
        assert seat.channel_mode == "intentional"

    def test_register_ambient(self):
        from seats import SeatRegistry

        reg = SeatRegistry()
        seat = reg.register("sess", "generic", "dev-uuid", channel_mode="ambient")
        assert seat.channel_mode == "ambient"

    def test_register_ambient_persists_in_registry(self):
        from seats import SeatRegistry

        reg = SeatRegistry()
        seat = reg.register("sess-ambient", "generic", "dev", channel_mode="ambient")
        retrieved = reg.get("sess-ambient", seat.seat_id)
        assert retrieved is not None
        assert retrieved.channel_mode == "ambient"


# ---------------------------------------------------------------------------
# SessionRegisterRequest — channel_mode field
# ---------------------------------------------------------------------------


class TestSessionRegisterRequestChannelMode:
    def test_default_channel_mode(self):
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(session_id="s1", participant_id="chaz")
        assert req.channel_mode == "intentional"

    def test_explicit_ambient(self):
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(
            session_id="s1",
            participant_id="chaz",
            channel_mode="ambient",
        )
        assert req.channel_mode == "ambient"

    def test_backward_compat_no_channel_mode(self):
        """Pre-Primitive-4 callers that omit channel_mode get intentional."""
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(session_id="s1", participant_id="cog")
        assert req.channel_mode == "intentional"
