"""Tests for the body channel: /ws/body/{id} + /v1/bodies/*.

A fake body connects over the WebSocket, declares a manifest, and answers
`act` frames with receipts. Run: PYTHONPATH=. .venv/bin/python -m pytest tests/test_body.py -v
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MANIFEST = {
    "kind": "live2d",
    "states": ["IDLE", "DANCING"],
    "params": [
        {"id": "ParamAngleX", "min": -30, "max": 30, "default": 0},
        {"id": "ParamMouthOpenY", "min": 0, "max": 1, "default": 0},
    ],
}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import body
    import http_api

    body._default = None  # fresh registry per test
    return TestClient(http_api.app, base_url="http://localhost:7860")


def _run_fake_body(client, body_id, stop, seen, *, answer=True):
    with client.websocket_connect(f"/ws/body/{body_id}") as ws:
        ws.send_json({"type": "hello", "manifest": MANIFEST})
        assert ws.receive_json() == {"type": "welcome", "body_id": body_id}
        seen["ready"].set()
        while not stop.is_set():
            msg = ws.receive_json()
            if msg.get("type") == "stop":
                break
            seen["acts"].append(msg)
            if answer:
                ws.send_json(
                    {
                        "type": "receipt",
                        "id": msg["id"],
                        "state": msg.get("state", "IDLE"),
                        "params": {k: v * 0.9 for k, v in (msg.get("params") or {}).items()},
                    }
                )


def _start(client, body_id, **kw):
    from body import get_default_body_registry

    stop = threading.Event()
    seen = {"ready": threading.Event(), "acts": []}
    t = threading.Thread(target=_run_fake_body, args=(client, body_id, stop, seen), kwargs=kw, daemon=True)
    t.start()
    assert seen["ready"].wait(3), "fake body did not connect"
    for _ in range(50):
        if get_default_body_registry().get(body_id):
            break
        time.sleep(0.02)
    return stop, seen, t


class TestValidation:
    def test_clamps_and_rejects(self):
        from body import _Body, validate_command

        b = _Body(body_id="x", manifest=MANIFEST, send=None)
        clean, rejected = validate_command(b, {"params": {"ParamAngleX": 99, "ParamNope": 1}, "release": ["ParamNope"]})
        assert clean == {"params": {"ParamAngleX": 30.0}}
        assert rejected == ["ParamNope"]

    def test_unknown_state(self):
        from body import BodyError, _Body, validate_command

        b = _Body(body_id="x", manifest=MANIFEST, send=None)
        with pytest.raises(BodyError) as e:
            validate_command(b, {"state": "MOONWALK"})
        assert e.value.status == 400


class TestBodyChannel:
    def test_no_body_is_404(self, client):
        r = client.post("/v1/bodies/ghost/act", json={"state": "IDLE"})
        assert r.status_code == 404

    def test_act_round_trip_returns_receipt(self, client):
        stop, seen, t = _start(client, "storm")
        listed = client.get("/v1/bodies").json()["bodies"]
        assert [b["body_id"] for b in listed] == ["storm"]
        assert listed[0]["states"] == ["IDLE", "DANCING"]

        r = client.post(
            "/v1/bodies/storm/act", json={"state": "DANCING", "params": {"ParamAngleX": 50, "ParamBogus": 1}}
        )
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["command"] == {"state": "DANCING", "params": {"ParamAngleX": 30.0}}
        assert out["rejected"] == ["ParamBogus"]
        # receipt is what the body reported, not an echo of the request
        assert out["receipt"]["params"]["ParamAngleX"] == pytest.approx(27.0)
        assert out["receipt"]["state"] == "DANCING"
        assert seen["acts"][0]["type"] == "act"

        assert client.get("/v1/bodies/storm").json()["last_receipt"]["state"] == "DANCING"
        stop.set()

    def test_bad_state_never_reaches_body(self, client):
        stop, seen, t = _start(client, "luna")
        r = client.post("/v1/bodies/luna/act", json={"state": "MOONWALK"})
        assert r.status_code == 400
        assert seen["acts"] == []
        stop.set()

    def test_silent_body_times_out(self, client):
        stop, seen, t = _start(client, "mute", answer=False)
        r = client.post("/v1/bodies/mute/act", json={"state": "IDLE", "timeout_sec": 0.2})
        assert r.status_code == 504
        stop.set()

    def test_hello_required(self, client):
        from starlette.websockets import WebSocketDisconnect

        with client.websocket_connect("/ws/body/rude") as ws:
            ws.send_json({"type": "act"})
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_json()
            assert e.value.code == 4400


class TestReplacedConnection:
    """A second connection for the same body_id closes the first (code 4409)."""

    def test_old_socket_is_closed_with_4409(self, client):
        from starlette.websockets import WebSocketDisconnect

        from body import get_default_body_registry

        with client.websocket_connect("/ws/body/dup") as first:
            first.send_json({"type": "hello", "manifest": MANIFEST})
            assert first.receive_json()["type"] == "welcome"
            old = get_default_body_registry().get("dup")
            with client.websocket_connect("/ws/body/dup") as second:
                second.send_json({"type": "hello", "manifest": MANIFEST})
                assert second.receive_json()["type"] == "welcome"
                with pytest.raises(WebSocketDisconnect) as exc:
                    first.receive_json()
                assert exc.value.code == 4409
                new = get_default_body_registry().get("dup")
                assert new is not None and new is not old
