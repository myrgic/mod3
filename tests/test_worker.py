"""Tests for the mod3.worker subprocess CLI.

Tests use subprocess + pipes to exercise the wire protocol without invoking
any real MLX models. Engine calls are patched via a MOCK_ENGINE env var that
the worker modules detect and route to stub implementations.

Each test:
1. Spawns the worker subprocess.
2. Writes JSON-line requests to stdin.
3. Reads JSON-line responses from stdout.
4. Asserts on the wire shape.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTHON = sys.executable
WORKER_MODULE = "mod3.worker"

# Path to the worktree root (parent of this tests/ dir)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env() -> dict:
    """Subprocess env: PYTHONPATH set to repo root + MOD3_WORKER_MOCK=1."""
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO_ROOT + ((":" + existing_pp) if existing_pp else "")
    env["MOD3_WORKER_MOCK"] = "1"
    return env


def _spawn(subcommand: str) -> subprocess.Popen:
    """Start a worker subprocess, return the Popen handle."""
    proc = subprocess.Popen(
        [PYTHON, "-m", WORKER_MODULE, subcommand],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
        cwd=REPO_ROOT,
    )
    return proc


def _read_line(proc: subprocess.Popen, timeout: float = 10.0) -> dict:
    """Read one JSON line from the subprocess stdout."""
    proc.stdout._CHUNK_SIZE = 1  # type: ignore[attr-defined]
    line = proc.stdout.readline()  # type: ignore[union-attr]
    if not line:
        raise EOFError("subprocess stdout closed before sending a line")
    return json.loads(line.strip())


def _write_line(proc: subprocess.Popen, msg: dict) -> None:
    """Write one JSON line to the subprocess stdin."""
    proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
    proc.stdin.flush()  # type: ignore[union-attr]


def _wait_ready(proc: subprocess.Popen, timeout: float = 10.0) -> dict:
    """Read until we see the ready event."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = _read_line(proc, timeout=timeout)
        if msg.get("type") == "event" and msg.get("event") == "ready":
            return msg
    raise TimeoutError("worker never emitted ready event")


def _silence_b64(duration_sec: float = 0.1, sample_rate: int = 16000) -> str:
    """Return base64-encoded silent PCM16 audio for test requests."""
    samples = np.zeros(int(sample_rate * duration_sec), dtype=np.int16)
    return base64.b64encode(samples.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Common lifecycle tests (parametrize over all three subcommands)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subcommand", ["tts", "vad", "stt"])
def test_startup_emits_ready(subcommand: str) -> None:
    """Every worker must emit a ready event before reading any input."""
    proc = _spawn(subcommand)
    try:
        msg = _wait_ready(proc)
        assert msg["type"] == "event"
        assert msg["event"] == "ready"
        assert msg["status"] == "ok"
    finally:
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait(timeout=5)


@pytest.mark.parametrize("subcommand", ["tts", "vad", "stt"])
def test_shutdown_command_exits_cleanly(subcommand: str) -> None:
    """Sending shutdown must cause the worker to exit with code 0."""
    proc = _spawn(subcommand)
    _wait_ready(proc)
    _write_line(proc, {"id": "req-shutdown", "type": "command", "command": "shutdown", "ts": "2026-01-01T00:00:00Z"})
    proc.stdin.close()  # type: ignore[union-attr]
    ret = proc.wait(timeout=10)
    assert ret == 0, f"worker exited with code {ret}"


@pytest.mark.parametrize("subcommand", ["tts", "vad", "stt"])
def test_health_command_returns_event(subcommand: str) -> None:
    """Health command must return an event with status=ok."""
    proc = _spawn(subcommand)
    _wait_ready(proc)
    _write_line(proc, {"id": "req-health", "type": "command", "command": "health", "ts": "2026-01-01T00:00:00Z"})
    msg = _read_line(proc)
    assert msg["type"] == "event"
    assert msg["event"] == "health"
    assert msg["status"] == "ok"
    assert msg["id"] == "req-health"
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


@pytest.mark.parametrize("subcommand", ["tts", "vad", "stt"])
def test_malformed_input_emits_structured_error(subcommand: str) -> None:
    """Malformed JSON input must produce a structured error, not crash the worker."""
    proc = _spawn(subcommand)
    _wait_ready(proc)

    # Write garbage that isn't valid JSON
    proc.stdin.write("not valid json at all\n")  # type: ignore[union-attr]
    proc.stdin.flush()  # type: ignore[union-attr]

    msg = _read_line(proc)
    assert msg["type"] == "error"
    assert msg["error_type"] == "ParseError"
    assert "malformed" in msg["error"].lower()

    # Worker should still be alive
    assert proc.poll() is None, "worker should survive malformed input"
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# TTS tests
# ---------------------------------------------------------------------------


def test_tts_synthesize_round_trips() -> None:
    """tts/synthesize must return a response with audio_b64 and duration_sec."""
    proc = _spawn("tts")
    _wait_ready(proc)

    _write_line(
        proc,
        {
            "id": "req-tts-1",
            "type": "request",
            "module": "tts",
            "op": "synthesize",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"text": "Hello world", "voice": "bm_lewis", "speed": 1.25},
        },
    )
    msg = _read_line(proc, timeout=15)
    assert msg["id"] == "req-tts-1"
    assert msg["type"] == "response"
    result = msg["result"]
    assert "audio_b64" in result
    assert "duration_sec" in result
    assert isinstance(result["duration_sec"], (int, float))
    assert result["sample_rate"] == 24000

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


def test_tts_stream_emits_chunk_events() -> None:
    """tts/stream must emit tts.chunk events ending with done=true."""
    proc = _spawn("tts")
    _wait_ready(proc)

    _write_line(
        proc,
        {
            "id": "req-stream-1",
            "type": "request",
            "module": "tts",
            "op": "stream",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"text": "Hello world", "voice": "bm_lewis", "speed": 1.25},
        },
    )

    chunks = []
    for _ in range(50):  # read up to 50 messages looking for done=true
        msg = _read_line(proc, timeout=15)
        assert msg["type"] == "event", f"expected event, got {msg['type']}"
        assert msg["event"] == "tts.chunk"
        chunks.append(msg)
        if msg.get("done"):
            break
    else:
        pytest.fail("stream never emitted done=true chunk")

    assert len(chunks) >= 1
    # All chunks must have chunk_index in data
    for i, chunk in enumerate(chunks[:-1]):
        assert "data" in chunk
        assert chunk["data"]["audio_b64"] != "" or chunk.get("done"), "non-final chunk should have audio"
    # Final chunk: done=true
    assert chunks[-1]["done"] is True

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


def test_tts_unknown_op_returns_error() -> None:
    """Unknown tts op must return a structured error, not crash."""
    proc = _spawn("tts")
    _wait_ready(proc)
    _write_line(
        proc,
        {
            "id": "req-bad",
            "type": "request",
            "module": "tts",
            "op": "nonexistent",
            "ts": "2026-01-01T00:00:00Z",
            "data": {},
        },
    )
    msg = _read_line(proc)
    assert msg["type"] == "error"
    assert msg["id"] == "req-bad"
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# VAD tests
# ---------------------------------------------------------------------------


def test_vad_detect_round_trips() -> None:
    """vad/detect must return has_speech and confidence."""
    proc = _spawn("vad")
    _wait_ready(proc)

    _write_line(
        proc,
        {
            "id": "req-vad-1",
            "type": "request",
            "module": "vad",
            "op": "detect",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"audio_b64": _silence_b64(), "sample_rate": 16000},
        },
    )
    msg = _read_line(proc, timeout=15)
    assert msg["id"] == "req-vad-1"
    assert msg["type"] == "response"
    result = msg["result"]
    assert "has_speech" in result
    assert "confidence" in result
    assert isinstance(result["has_speech"], bool)

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# STT tests
# ---------------------------------------------------------------------------


def test_stt_transcribe_round_trips() -> None:
    """stt/transcribe must return a transcript string."""
    proc = _spawn("stt")
    _wait_ready(proc)

    _write_line(
        proc,
        {
            "id": "req-stt-1",
            "type": "request",
            "module": "stt",
            "op": "transcribe",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"audio_b64": _silence_b64(0.5), "sample_rate": 16000, "language": "en"},
        },
    )
    msg = _read_line(proc, timeout=30)
    assert msg["id"] == "req-stt-1"
    assert msg["type"] == "response"
    result = msg["result"]
    assert "transcript" in result
    assert isinstance(result["transcript"], str)
    assert "stt_ms" in result

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)


def test_stt_transcribe_streaming_round_trips() -> None:
    """stt/transcribe_streaming must return confirmed/tentative fields."""
    proc = _spawn("stt")
    _wait_ready(proc)

    _write_line(
        proc,
        {
            "id": "req-stt-stream-1",
            "type": "request",
            "module": "stt",
            "op": "transcribe_streaming",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"audio_b64": _silence_b64(0.5), "sample_rate": 16000, "tier": "t1"},
        },
    )
    msg = _read_line(proc, timeout=30)
    assert msg["id"] == "req-stt-stream-1"
    assert msg["type"] == "response"
    result = msg["result"]
    assert "confirmed" in result
    assert "tentative" in result
    assert "tier" in result

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=5)
