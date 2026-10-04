"""Tests for animation clips: clip.py compiler + /v1/bodies/{id}/play|stop + /v1/clips/check.

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/test_clips.py -v
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clip import ClipError, compile_clip, evaluate  # noqa: E402

CLIPS_DIR = Path(__file__).resolve().parents[1] / "clips"

NOD = {
    "name": "nod",
    "args": {"times": {"default": 2, "min": 1, "max": 4}, "depth": {"default": 0.5, "min": 0.1, "max": 1}},
    "duration_ms": "500*times+200",
    "tracks": {"head.y": {"wave": {"amp": "-depth", "period_ms": 500}}},
}


# ------------------------------------------------------------ expressions
def test_expressions_are_arithmetic_only():
    assert evaluate("2*x+1", {"x": 3}, "t") == 7
    assert evaluate("max(0.2, 1-x)", {"x": 0.5}, "t") == 0.5
    assert evaluate("clamp(x, 0, 1)", {"x": 4}, "t") == 1
    for bad in ("__import__('os')", "x.real", "[1][0]", "x if x else 1", "open('f')", "2**99999", "lambda: 1"):
        with pytest.raises(ClipError):
            evaluate(bad, {"x": 1}, "t")
    with pytest.raises(ClipError, match="unknown name"):
        evaluate("y", {"x": 1}, "t")
    with pytest.raises(ClipError, match="division by zero"):
        evaluate("1/0", {}, "t")
    with pytest.raises(ClipError):
        evaluate(True, {}, "t")


# ------------------------------------------------------------ compile
def test_args_default_override_clamp_and_reject():
    c, rejected = compile_clip(NOD, args={"times": 9, "bogus": 1})
    assert c["args"] == {"times": 4, "depth": 0.5}
    assert c["duration_ms"] == 2200
    assert c["tracks"]["head.y"]["wave"][0]["amp"] == -0.5
    assert rejected == ["arg:bogus"]


def test_body_mapping_drops_what_the_body_lacks():
    clip = {
        "name": "x",
        "loop": True,
        "tracks": {"head.x": {"const": 0.5}, "cheek": {"const": 1}, "param:ParamHair": {"const": 3}},
    }
    c, rejected = compile_clip(
        clip,
        body_channels={"head.x": {}},
        body_params={"ParamAngleX": (-30, 30)},
    )
    assert list(c["tracks"]) == ["head.x"]
    assert sorted(rejected) == ["cheek", "param:ParamHair"]
    with pytest.raises(ClipError, match="no track survives"):
        compile_clip({"name": "y", "loop": True, "tracks": {"cheek": {"const": 1}}}, body_channels={"head.x": {}})


@pytest.mark.parametrize(
    "clip,match",
    [
        ({"name": "Bad Name", "loop": True, "tracks": {"head.x": {"const": 0}}}, "name"),
        ({"name": "a", "loop": True, "tracks": {"tail.wag": {"const": 0}}}, "unknown channel"),
        ({"name": "a", "loop": True, "tracks": {"head.x": {"spin": 1}}}, "unknown field"),
        ({"name": "a", "loop": True, "tracks": {"head.x": {}}}, "needs at least one"),
        ({"name": "a", "tracks": {"head.x": {"const": 0}}}, "duration_ms"),
        ({"name": "a", "loop": True, "tracks": {"head.x": {"wave": {"period_ms": 5}}}}, "period_ms"),
        ({"name": "a", "loop": True, "tracks": {"head.x": {"keys": [[100, 0], [50, 1]]}}}, "must not decrease"),
        ({"name": "a", "loop": True, "tracks": {"head.x": {"follow": {"of": "nose"}}}}, "of"),
        ({"name": "a", "loop": True, "tracks": {"head.x": {"const": 0, "smooth": 0}}}, "smooth"),
        ({"name": "a", "duration_ms": 10**9, "tracks": {"head.x": {"const": 0}}}, "duration_ms"),
    ],
)
def test_structural_errors_name_the_field(clip, match):
    with pytest.raises(ClipError, match=match):
        compile_clip(clip)


def test_keys_infer_duration_and_base_state_checked():
    c, _ = compile_clip({"name": "k", "tracks": {"head.x": {"keys": [[0, 0], [800, 1]]}}})
    assert c["duration_ms"] == 800
    with pytest.raises(ClipError, match="base"):
        compile_clip(
            {"name": "k", "base": "MOONWALK", "loop": True, "tracks": {"head.x": {"const": 0}}}, body_states=["IDLE"]
        )


def test_shipped_clips_all_compile():
    files = sorted(CLIPS_DIR.glob("*.json"))
    assert len(files) >= 10
    for f in files:
        clip = json.loads(f.read_text())
        assert clip["name"] == f.stem, f
        compile_clip(clip)


# ------------------------------------------------------------ endpoints
MANIFEST = {
    "kind": "live2d",
    "states": ["IDLE"],
    "params": [{"id": "ParamAngleY", "min": -30, "max": 30, "default": 0}],
    "channels": {"angleY": "ParamAngleY"},
    "semantic_channels": {"head.y": {"polarity": "bi", "params": ["ParamAngleY"]}},
}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import body
    import http_api

    body._default = None
    return TestClient(http_api.app, base_url="http://localhost:7860")


def _fake_body(client, body_id, stop, seen):
    with client.websocket_connect(f"/ws/body/{body_id}") as ws:
        ws.send_json({"type": "hello", "manifest": MANIFEST})
        assert ws.receive_json()["type"] == "welcome"
        while not stop.is_set():
            msg = ws.receive_json()
            seen.append(msg)
            if msg["type"] == "play":
                ws.send_json(
                    {"type": "receipt", "id": msg["id"], "playing": msg["clip"]["name"], "semantic": {"head.y": -0.4}}
                )
            elif msg["type"] == "stop":
                ws.send_json({"type": "receipt", "id": msg["id"], "stopped": [msg.get("clip")]})
                return


def test_play_compiles_for_the_body_and_returns_receipt(client):
    stop, seen = threading.Event(), []
    t = threading.Thread(target=_fake_body, args=(client, "b1", stop, seen), daemon=True)
    t.start()
    for _ in range(50):
        if client.get("/v1/bodies/b1").status_code == 200:
            break
        time.sleep(0.02)
    clip = {**NOD, "tracks": {**NOD["tracks"], "cheek": {"const": 1}}}
    r = client.post("/v1/bodies/b1/play", json={"clip": clip, "args": {"times": 3}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["rejected"] == ["cheek"]
    assert out["receipt"]["playing"] == "nod"
    sent = seen[-1]
    assert sent["type"] == "play" and sent["clip"]["duration_ms"] == 1700
    assert list(sent["clip"]["tracks"]) == ["head.y"]

    bad = client.post(
        "/v1/bodies/b1/play", json={"clip": {"name": "x", "loop": True, "tracks": {"head.y": {"const": "os.system"}}}}
    )
    assert bad.status_code == 400

    r = client.post("/v1/bodies/b1/stop", json={"clip": "nod"})
    assert r.status_code == 200 and r.json()["receipt"]["stopped"] == ["nod"]
    stop.set()
    t.join(timeout=2)


def test_play_unknown_body_404_and_check_endpoint(client):
    assert client.post("/v1/bodies/nobody/play", json={"clip": NOD}).status_code == 404
    r = client.post("/v1/clips/check", json={"clip": NOD, "args": {"depth": 0.9}})
    assert r.status_code == 200 and r.json()["compiled"]["args"]["depth"] == 0.9
    assert client.post("/v1/clips/check", json={"clip": {"name": "x"}}).status_code == 400


# ------------------------------------------------------------ CLI: library paths stay inside the library
CLI = Path(__file__).resolve().parents[1] / "scripts" / "mod3-body"


def _cli(tmp_path, *args):
    import os
    import subprocess

    env = {
        **os.environ,
        "MOD3_BODY_LIBRARY": str(tmp_path / "lib"),
        "MOD3_BODY_SHARED": str(tmp_path / "shared"),
        "MOD3_URL": "http://127.0.0.1:9",  # nothing listens; commands under test never call it
    }
    return subprocess.run([sys.executable, str(CLI), *args], env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize("bad", ["/tmp/pwned", "../escape", "../../x", "a/b", "UPPER", ".hidden", "a..b"])
def test_cli_refuses_names_that_leave_the_library(tmp_path, bad):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "evil.json").write_text(json.dumps({"name": bad, "loop": True, "tracks": {"head.x": {"const": 0}}}))
    for args in (["fork", "evil"], ["fork", "nod", "--as", bad], ["rm", bad], ["show", bad]):
        r = _cli(tmp_path, *args)
        assert r.returncode != 0, (args, r.stdout)
        assert "invalid clip name" in r.stderr or "no clip" in r.stderr, (args, r.stderr)
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.parent != shared]
    assert written == [], written


def test_cli_fork_and_rm_stay_in_library(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "nod.json").write_text(json.dumps(NOD))
    r = _cli(tmp_path, "fork", "nod", "--as", "my-nod")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "lib" / "my-nod.json").exists()
    assert _cli(tmp_path, "rm", "my-nod").returncode == 0
    assert not (tmp_path / "lib" / "my-nod.json").exists()
