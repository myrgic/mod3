"""Integration tests for the /ws/acp endpoint.

Uses FastAPI TestClient (synchronous WebSocket test) to exercise the full
JSON-RPC handshake without loading actual MLX models.

Run with: PYTHONPATH=. .venv/bin/python -m pytest tests/test_acp_server.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app)


# ---------------------------------------------------------------------------
# Helper to build ACP messages
# ---------------------------------------------------------------------------


def _req(method: str, params: dict, id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params})


def _notif(method: str, params: dict) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params})


def _parse(raw: str) -> dict:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestAcpInitialize:
    def test_initialize_returns_capabilities(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(
                _req(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {"fs": {}, "terminal": False},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                    id=0,
                )
            )
            resp = _parse(ws.receive_text())

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 0
        assert "result" in resp
        caps = resp["result"]["agentCapabilities"]
        assert caps["promptCapabilities"]["audio"] is False
        assert caps["promptCapabilities"]["image"] is False
        assert caps["promptCapabilities"]["embeddedContext"] is False
        assert caps["sessionCapabilities"] == {}

    def test_initialize_handles_string_id(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {"protocolVersion": 1}, id="init-1"))
            resp = _parse(ws.receive_text())
        assert resp["id"] == "init-1"


# ---------------------------------------------------------------------------
# session/new
# ---------------------------------------------------------------------------


class TestAcpSessionNew:
    def test_session_new_returns_session_id(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()  # consume initialize response

            ws.send_text(_req("session/new", {"cwd": "/", "mcpServers": []}, id=1))
            resp = _parse(ws.receive_text())

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "sessionId" in resp["result"]
        assert resp["result"]["sessionId"].startswith("mod3-")

    def test_session_id_is_unique_per_call(self, client):
        ids = []
        for i in range(2):
            with client.websocket_connect("/ws/acp") as ws:
                ws.send_text(_req("initialize", {}, id=0))
                ws.receive_text()
                ws.send_text(_req("session/new", {}, id=1))
                resp = _parse(ws.receive_text())
                ids.append(resp["result"]["sessionId"])
        assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# session/prompt
# ---------------------------------------------------------------------------


class TestAcpSessionPrompt:
    def test_session_prompt_empty_text_returns_error(self, client):
        """Prompt with no text blocks should return an InvalidParams error."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()
            ws.send_text(_req("session/new", {}, id=1))
            new_resp = _parse(ws.receive_text())
            session_id = new_resp["result"]["sessionId"]

            ws.send_text(
                _req(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": []},
                    id=2,
                )
            )
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    @patch("http_api.os.environ.get")
    def test_session_prompt_returns_error_when_bridge_disabled(self, mock_env_get, client):
        """When MOD3_USE_COGOS_AGENT is unset, session/prompt returns a structured
        error rather than hanging. The error message tells the operator what
        env var to set so they can recover.
        """
        # Force is_enabled() to return False by clearing the env var.
        mock_env_get.return_value = ""

        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()
            ws.send_text(_req("session/new", {}, id=1))
            new_resp = _parse(ws.receive_text())
            session_id = new_resp["result"]["sessionId"]

            ws.send_text(
                _req(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": "Hello?"}],
                    },
                    id=2,
                )
            )

            resp = _parse(ws.receive_text())

        assert resp.get("id") == 2
        assert "error" in resp
        assert resp["error"]["code"] == -32000
        assert "MOD3_USE_COGOS_AGENT" in resp["error"]["message"]

    def test_session_prompt_passes_through_to_kernel_cycle(self, client):
        """When the kernel bridge is enabled and reachable, session/prompt:
            1. registers an ACP listener
            2. posts the user text to the kernel chat bus
            3. waits for the response on the listener queue
            4. emits a session/update agent_message_chunk with the response text
            5. resolves the original session/prompt request with stopReason=end_turn

        This mocks the bridge so no real kernel is required. The kernel-side
        response is injected directly into the listener queue, simulating what
        run_response_bridge does when a real kernel event arrives.
        """
        import asyncio

        # Track the registered listener queue so we can inject the response.
        captured_queue: dict[str, asyncio.Queue] = {}
        real_register = None

        # Capture-and-pass-through wrapper around register_acp_listener.
        import cogos_agent_bridge as _bridge_mod

        real_register = _bridge_mod.register_acp_listener

        def _capture_register(session_id: str) -> asyncio.Queue:
            q = real_register(session_id)
            captured_queue[session_id] = q
            return q

        # post_user_message mock: succeeds, then schedules the kernel "response"
        # arrival on the listener queue. We use asyncio.create_task so it runs
        # concurrently with the awaiting _stream_prompt.
        async def _fake_post(text: str, session_id: str) -> bool:
            # Schedule the response delivery on the next event-loop tick.
            async def _deliver():
                # Tiny sleep so the handler is actually waiting on the queue.
                await asyncio.sleep(0.01)
                q = captured_queue.get(session_id)
                if q is not None:
                    from bus_bridge import BusEnvelope

                    env = BusEnvelope(
                        raw={},
                        kind="agent_response",
                        payload={"text": "kernel says hello"},
                        event_id="evt-1",
                        ts=None,
                    )
                    await q.put(("kernel says hello", env))

            asyncio.create_task(_deliver())
            return True

        with (
            patch("cogos_agent_bridge.is_enabled", return_value=True),
            patch("cogos_agent_bridge.register_acp_listener", side_effect=_capture_register),
            patch("cogos_agent_bridge.post_user_message", side_effect=_fake_post),
        ):
            with client.websocket_connect("/ws/acp") as ws:
                ws.send_text(_req("initialize", {}, id=0))
                ws.receive_text()
                ws.send_text(_req("session/new", {}, id=1))
                new_resp = _parse(ws.receive_text())
                session_id = new_resp["result"]["sessionId"]

                ws.send_text(
                    _req(
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "Hello?"}],
                        },
                        id=2,
                    )
                )

                messages = []
                for _ in range(10):
                    raw = ws.receive_text()
                    parsed = _parse(raw)
                    messages.append(parsed)
                    if parsed.get("id") == 2:
                        break

        # Check: at least one session/update notification with the kernel text.
        updates = [
            m
            for m in messages
            if m.get("method") == "session/update" and m.get("params", {}).get("sessionUpdate") == "agent_message_chunk"
        ]
        assert updates, f"No agent_message_chunk update; got: {messages}"
        chunk = updates[0]["params"]["content"]
        assert chunk.get("type") == "text"
        assert chunk.get("text") == "kernel says hello"

        # And: the final session/prompt response resolves with end_turn.
        prompt_responses = [m for m in messages if m.get("id") == 2]
        assert prompt_responses, f"No response for session/prompt; got: {messages}"
        final = prompt_responses[0]
        assert "result" in final
        assert final["result"]["stopReason"] == "end_turn"

    def test_session_prompt_returns_error_when_kernel_unreachable(self, client):
        """When post_user_message returns False (kernel POST failed), session/prompt
        returns a structured error rather than hanging on the listener queue.
        """

        async def _fake_post_fail(text: str, session_id: str) -> bool:
            return False

        with (
            patch("cogos_agent_bridge.is_enabled", return_value=True),
            patch("cogos_agent_bridge.post_user_message", side_effect=_fake_post_fail),
        ):
            with client.websocket_connect("/ws/acp") as ws:
                ws.send_text(_req("initialize", {}, id=0))
                ws.receive_text()
                ws.send_text(_req("session/new", {}, id=1))
                new_resp = _parse(ws.receive_text())
                session_id = new_resp["result"]["sessionId"]

                ws.send_text(
                    _req(
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "Hello?"}],
                        },
                        id=2,
                    )
                )

                resp = _parse(ws.receive_text())

        assert resp.get("id") == 2
        assert "error" in resp
        assert resp["error"]["code"] == -32000
        assert "Kernel unreachable" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# session/cancel
# ---------------------------------------------------------------------------


class TestAcpSessionCancel:
    def test_session_cancel_notification_no_response(self, client):
        """session/cancel is a notification -- no response should be sent."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()
            ws.send_text(_req("session/new", {}, id=1))
            new_resp = _parse(ws.receive_text())
            session_id = new_resp["result"]["sessionId"]

            # Send cancel notification (no id).
            ws.send_text(_notif("session/cancel", {"sessionId": session_id}))

            # Immediately send another request to confirm the connection is still alive.
            ws.send_text(_req("session/new", {}, id=2))
            resp = _parse(ws.receive_text())
            assert resp["id"] == 2
            assert "sessionId" in resp["result"]


# ---------------------------------------------------------------------------
# Unknown method
# ---------------------------------------------------------------------------


class TestAcpUnknownMethod:
    def test_unknown_method_returns_error(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("bogus/method", {}, id=99))
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32601
        assert resp["id"] == 99

    def test_parse_error_on_invalid_json(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text("not-json{{")
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32700

    def test_invalid_jsonrpc_version(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(json.dumps({"jsonrpc": "1.0", "id": 1, "method": "initialize"}))
            resp = _parse(ws.receive_text())
        assert "error" in resp
