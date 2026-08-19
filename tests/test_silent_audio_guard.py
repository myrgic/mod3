"""The test suite must never open the operator's real speakers.

Regression guard for a defect where running pytest made the machine audibly
talk: tests/test_server_topology.py POSTs /v1/speak to obtain an in-flight
job it can poll, and that daemon opened the real audio device. The fixture
had carefully isolated every other shared resource (mic, barge-in signal,
speaking lock, LM Studio probe) but not the speaker, because no seam existed
to silence it.

These tests bind to the public seam (MOD3_SILENT_AUDIO -> _NullOutputStream)
rather than to line numbers, so they keep holding if the playback internals
are rewritten.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import adaptive_player
from adaptive_player import AdaptivePlayer, _NullOutputStream, silent_audio_enabled


class TestSilentAudioFlag:
    def test_conftest_silences_audio_for_the_whole_suite(self):
        """Silence is the default, so a new test cannot forget to opt in."""
        assert os.environ.get("MOD3_SILENT_AUDIO") == "1", (
            "conftest.pytest_configure must set MOD3_SILENT_AUDIO=1 — without it "
            "any test that reaches playback plays out of the operator's speakers"
        )
        assert silent_audio_enabled() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_spellings_enable_silence(self, monkeypatch, value):
        monkeypatch.setenv("MOD3_SILENT_AUDIO", value)
        assert silent_audio_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsy_spellings_leave_real_audio_intact(self, monkeypatch, value):
        """The guard must be opt-in for production — the daemon still plays."""
        monkeypatch.setenv("MOD3_SILENT_AUDIO", value)
        assert silent_audio_enabled() is False


class TestPlaybackNeverTouchesTheDevice:
    def test_player_uses_null_stream_when_silenced(self, monkeypatch):
        """The real sd.OutputStream must not be constructed under the guard."""
        monkeypatch.setenv("MOD3_SILENT_AUDIO", "1")

        def _boom(*args, **kwargs):
            raise AssertionError(
                "sd.OutputStream was constructed while MOD3_SILENT_AUDIO=1 — "
                "this is the exact call that makes the operator's laptop talk"
            )

        monkeypatch.setattr(adaptive_player.sd, "OutputStream", _boom)

        player = AdaptivePlayer(sample_rate=24_000, buffer_size=256)
        player.queue_audio(np.zeros(2048, dtype=np.float32))
        player.mark_done()
        metrics = player.wait()

        assert isinstance(player._stream, (_NullOutputStream, type(None)))
        assert metrics is not None, "playback metrics must survive silencing"

    def test_metrics_survive_the_null_stream(self, monkeypatch):
        """Silencing must not hollow out the instrumentation tests assert on."""
        monkeypatch.setenv("MOD3_SILENT_AUDIO", "1")

        player = AdaptivePlayer(sample_rate=24_000, buffer_size=256)
        samples = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)
        player.queue_audio(samples)
        player.mark_done()
        metrics = player.wait()

        as_dict = metrics if isinstance(metrics, dict) else getattr(metrics, "__dict__", {})
        assert as_dict, f"expected structured playback metrics, got {metrics!r}"

    def test_null_stream_fires_finished_callback_exactly_once(self):
        """PortAudio semantics: finished_callback runs once, on stop."""
        calls = []
        stream = _NullOutputStream(
            samplerate=24_000,
            channels=1,
            dtype="float32",
            callback=lambda outdata, frames, t, status: None,
            finished_callback=lambda: calls.append(1),
            blocksize=256,
        )
        stream.start()
        stream.stop()
        stream.close()  # close() calls stop() again — must not double-fire

        assert sum(calls) == 1, f"finished_callback fired {sum(calls)} times, expected exactly 1"


class TestSubprocessFixtureIsolation:
    def test_topology_fixture_declares_silent_audio(self):
        """The subprocess fixture must pass the guard to its child daemon.

        conftest covers in-process playback, but test_server_topology spawns a
        real server via subprocess.Popen. It inherits os.environ, so it is
        already covered -- this asserts the fixture ALSO states it explicitly,
        so the intent is visible to anyone reading that file in isolation.
        """
        from pathlib import Path

        source = Path(__file__).with_name("test_server_topology.py").read_text()
        assert "MOD3_SILENT_AUDIO" in source, (
            "test_server_topology._running_daemon must set MOD3_SILENT_AUDIO — "
            "it spawns a real daemon and POSTs /v1/speak"
        )
