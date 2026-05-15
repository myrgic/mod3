"""Unit tests for the phase_timer context manager and emit_phase helper.

Tests both the synchronous (``with``) and asynchronous (``async with``)
variants of phase_timer, verifying:
  - exactly one event is emitted on exit
  - event schema matches the chat.phase.* specification
  - duration_ms is a non-negative integer
  - ok=True on clean exit, ok=False + error on exception
  - original exception is re-raised after emit

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_phase_timer.py -v
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat_flow_log import (
    CHAT_PHASE_PREFIX,
    ChatFlowLog,
    _PhaseTimer,
    get_chat_flow_log,
    phase_timer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_log() -> ChatFlowLog:
    """Return a fresh (empty) ChatFlowLog, isolated from the singleton."""
    return ChatFlowLog()


def _drain_phase_events(log: ChatFlowLog) -> list[dict]:
    """Return all chat.phase.* events currently in the ring buffer."""
    return log.query(event_type="chat.phase.*", limit=100)


# ---------------------------------------------------------------------------
# emit_phase — direct tests
# ---------------------------------------------------------------------------


class TestEmitPhase:
    def test_event_in_ring(self):
        log = _fresh_log()
        log.emit_phase("provider_call", "session-1", "msg-1", 123)
        events = _drain_phase_events(log)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "chat.phase.provider_call"
        assert ev["phase_name"] == "provider_call"
        assert ev["session_id"] == "session-1"
        assert ev["message_id"] == "msg-1"
        assert ev["duration_ms"] == 123
        assert ev["ok"] is True
        assert "error" not in ev

    def test_ok_false_includes_error(self):
        log = _fresh_log()
        log.emit_phase("stt_transcribe", "s", "m", 50, ok=False, error="timeout")
        events = _drain_phase_events(log)
        assert len(events) == 1
        ev = events[0]
        assert ev["ok"] is False
        assert ev["error"] == "timeout"

    def test_never_raises(self):
        """emit_phase must not raise even if called with junk args."""
        log = _fresh_log()
        # Force an internal error by patching _emit_phase_inner
        original = log._emit_phase_inner
        log._emit_phase_inner = None  # type: ignore[assignment]
        # Should swallow the AttributeError / TypeError silently
        result = log.emit_phase("x", "s", "m", 0)
        assert result == {}
        log._emit_phase_inner = original  # restore


# ---------------------------------------------------------------------------
# Synchronous phase_timer
# ---------------------------------------------------------------------------


class TestPhaseTimerSync:
    def test_emits_one_event_on_exit(self):
        log = _fresh_log()
        _original_emit = log.emit_phase
        emitted: list[dict] = []

        def _capture(*args, **kwargs):
            ev = _original_emit(*args, **kwargs)
            emitted.append(ev)
            return ev

        log.emit_phase = _capture  # type: ignore[method-assign]

        timer = phase_timer("provider_call", "ch-1", "msg-a")
        # Monkeypatch: the timer calls get_chat_flow_log() internally, so we
        # test via the singleton to verify end-to-end behaviour.
        singleton = get_chat_flow_log()
        initial_count = len(singleton.query(event_type="chat.phase.*", limit=1000))

        with phase_timer("provider_call", "ch-singleton", "msg-sync"):
            time.sleep(0.001)  # ensure measurable wall time

        events = singleton.query(event_type="chat.phase.provider_call", limit=10)
        new_events = [e for e in events if e["session_id"] == "ch-singleton"]
        assert len(new_events) == 1, f"expected 1 new phase event, got {new_events}"
        ev = new_events[0]
        assert ev["phase_name"] == "provider_call"
        assert ev["duration_ms"] >= 0
        assert ev["ok"] is True

    def test_duration_ms_is_positive_integer(self):
        singleton = get_chat_flow_log()
        with phase_timer("stt_transcribe", "ch-dur", "msg-dur"):
            time.sleep(0.005)

        events = singleton.query(event_type="chat.phase.stt_transcribe", limit=10)
        recent = [e for e in events if e["session_id"] == "ch-dur"]
        assert len(recent) >= 1
        ev = recent[-1]
        assert isinstance(ev["duration_ms"], int)
        assert ev["duration_ms"] >= 1  # at least 1ms for 5ms sleep

    def test_exception_sets_ok_false_and_reraises(self):
        singleton = get_chat_flow_log()
        with pytest.raises(ValueError, match="deliberate"):
            with phase_timer("agent_dispatch", "ch-exc", "msg-exc"):
                raise ValueError("deliberate")

        events = singleton.query(event_type="chat.phase.agent_dispatch", limit=10)
        recent = [e for e in events if e["session_id"] == "ch-exc"]
        assert len(recent) >= 1
        ev = recent[-1]
        assert ev["ok"] is False
        assert "deliberate" in ev.get("error", "")

    def test_event_schema_fields_present(self):
        singleton = get_chat_flow_log()
        with phase_timer("turn_total", "ch-schema", "msg-schema"):
            pass

        events = singleton.query(event_type="chat.phase.turn_total", limit=10)
        recent = [e for e in events if e["session_id"] == "ch-schema"]
        assert len(recent) >= 1
        ev = recent[-1]
        required_fields = {"ts", "event_type", "session_id", "message_id", "phase_name", "duration_ms", "ok"}
        assert required_fields.issubset(ev.keys()), f"missing fields: {required_fields - ev.keys()}"
        assert ev["event_type"].startswith(CHAT_PHASE_PREFIX)


# ---------------------------------------------------------------------------
# Asynchronous phase_timer
# ---------------------------------------------------------------------------


class TestPhaseTimerAsync:
    def test_async_emits_one_event(self):
        async def _run():
            singleton = get_chat_flow_log()
            async with phase_timer("tts_synthesize", "ch-async", "msg-async"):
                await asyncio.sleep(0.001)

            events = singleton.query(event_type="chat.phase.tts_synthesize", limit=10)
            recent = [e for e in events if e["session_id"] == "ch-async"]
            assert len(recent) >= 1
            ev = recent[-1]
            assert ev["phase_name"] == "tts_synthesize"
            assert ev["duration_ms"] >= 0
            assert ev["ok"] is True

        asyncio.run(_run())

    def test_async_duration_ms_reflects_sleep(self):
        async def _run():
            singleton = get_chat_flow_log()
            async with phase_timer("stt_capture", "ch-async-dur", "m"):
                await asyncio.sleep(0.01)  # 10ms

            events = singleton.query(event_type="chat.phase.stt_capture", limit=10)
            recent = [e for e in events if e["session_id"] == "ch-async-dur"]
            assert len(recent) >= 1
            ev = recent[-1]
            # Allow wide tolerance for CI — at least 1ms from a 10ms sleep
            assert ev["duration_ms"] >= 1

        asyncio.run(_run())

    def test_async_exception_sets_ok_false_and_reraises(self):
        async def _run():
            singleton = get_chat_flow_log()
            with pytest.raises(RuntimeError, match="async-boom"):
                async with phase_timer("provider_call", "ch-async-exc", "m"):
                    raise RuntimeError("async-boom")

            events = singleton.query(event_type="chat.phase.provider_call", limit=10)
            recent = [e for e in events if e["session_id"] == "ch-async-exc"]
            assert len(recent) >= 1
            ev = recent[-1]
            assert ev["ok"] is False
            assert "async-boom" in ev.get("error", "")

        asyncio.run(_run())

    def test_async_event_schema_fields_present(self):
        async def _run():
            singleton = get_chat_flow_log()
            async with phase_timer("tool_execute", "ch-async-schema", "m"):
                pass

            events = singleton.query(event_type="chat.phase.tool_execute", limit=10)
            recent = [e for e in events if e["session_id"] == "ch-async-schema"]
            assert len(recent) >= 1
            ev = recent[-1]
            required = {"ts", "event_type", "session_id", "message_id", "phase_name", "duration_ms", "ok"}
            assert required.issubset(ev.keys())

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# query wildcard support
# ---------------------------------------------------------------------------


class TestQueryWildcard:
    def test_wildcard_matches_all_phase_events(self):
        log = _fresh_log()
        log.emit_phase("stt_capture", "s", "m", 10)
        log.emit_phase("provider_call", "s", "m", 500)
        log.emit_phase("tts_synthesize", "s", "m", 300)
        log.emit(
            "chat.message_received",
            "s", "m", "ws", [], "hello", "inbound"
        )

        phase_events = log.query(event_type="chat.phase.*", limit=20)
        assert len(phase_events) == 3
        for ev in phase_events:
            assert ev["event_type"].startswith(CHAT_PHASE_PREFIX)

    def test_wildcard_and_exact_combined(self):
        log = _fresh_log()
        log.emit_phase("stt_capture", "s", "m", 10)
        log.emit_phase("provider_call", "s", "m", 500)
        log.emit("chat.message_received", "s", "m", "ws", [], "hi", "inbound")

        # Ask for both phase events AND the chat.message_received event
        events = log.query(event_type="chat.phase.*,chat.message_received", limit=20)
        types = {ev["event_type"] for ev in events}
        assert "chat.phase.stt_capture" in types
        assert "chat.phase.provider_call" in types
        assert "chat.message_received" in types
