"""Regression tests for mod3_speak's chat-panel mirror.

Before this fix, calling ``mod3_speak`` only hit ``POST /v1/speak``: audio
played, but no message reached the dashboard chat panel and no entry landed
in the per-session ring buffer. Operators rereading their conversation later
saw a transcript with the user's prompts but no agent responses for any turn
that was delivered as speech.

The fix mirrors the spoken text to ``POST /v1/dashboard-chat`` (as an
``assistant`` message under the seat's ``session_id``) before invoking the
speak endpoint. ``post_to_chat=False`` opts out for non-conversational audio
(system sounds, UI cues).

The MCP tool wires into a ``ChannelClient`` instance; we exercise the bound
closure directly via ``build_mcp_server``'s tool registry to avoid spinning
up an actual MCP stdio transport.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def speak_tool(monkeypatch, tmp_path):
    """Build the mod3_speak tool bound to a ChannelClient with stubbed HTTP."""
    import clients.channel_client as cc

    monkeypatch.setattr(cc, "_TOKEN_PATH", tmp_path / "channel-client-token")

    client = cc.ChannelClient("http://localhost:7860", "session-under-test")
    client.seat_id = "seat-under-test"

    captured_requests = []

    class FakeResponse:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {"job_id": "abc", "status": "speaking"}

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "http://x")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError(
                    f"http {self.status_code}",
                    request=request,
                    response=response,
                )

        def json(self):
            return self._payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None, timeout=None):
            captured_requests.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(cc.httpx, "AsyncClient", FakeClient)

    mcp = cc.build_mcp_server(client)
    # FastMCP exposes registered tools via mcp._tool_manager._tools (private API).
    # We invoke the underlying coroutine directly to keep the test transport-free.
    tools = mcp._tool_manager._tools
    speak_tool_obj = tools["mod3_speak"]

    async def call(text: str = "hello", **kwargs):
        return await speak_tool_obj.fn(text=text, **kwargs)

    return call, captured_requests


class TestSpeakChatMirror:
    @pytest.mark.asyncio
    async def test_default_mirrors_to_chat_then_speaks(self, speak_tool):
        call, captured = speak_tool
        result = await call(text="hi there")
        assert result == {"job_id": "abc", "status": "speaking"}

        urls = [r["url"] for r in captured]
        assert "http://localhost:7860/v1/dashboard-chat" in urls
        assert "http://localhost:7860/v1/speak" in urls
        # Chat post must come BEFORE speak so the transcript lands first.
        assert urls.index("http://localhost:7860/v1/dashboard-chat") < urls.index("http://localhost:7860/v1/speak")

    @pytest.mark.asyncio
    async def test_chat_post_carries_session_id_and_role(self, speak_tool):
        call, captured = speak_tool
        await call(text="another line")
        chat_req = next(r for r in captured if r["url"].endswith("/v1/dashboard-chat"))
        body = chat_req["json"]
        assert body["text"] == "another line"
        assert body["role"] == "assistant"
        assert body["session_id"] == "session-under-test"
        assert body["seat_id"] == "seat-under-test"

    @pytest.mark.asyncio
    async def test_post_to_chat_false_skips_mirror(self, speak_tool):
        call, captured = speak_tool
        await call(text="system ping", post_to_chat=False)
        urls = [r["url"] for r in captured]
        assert "http://localhost:7860/v1/dashboard-chat" not in urls
        assert "http://localhost:7860/v1/speak" in urls

    @pytest.mark.asyncio
    async def test_chat_post_failure_does_not_block_speak(self, monkeypatch, tmp_path):
        """A failed chat-mirror must not prevent speech — audio is the
        primary path; the chat panel is best-effort."""
        import clients.channel_client as cc

        monkeypatch.setattr(cc, "_TOKEN_PATH", tmp_path / "channel-client-token")
        client = cc.ChannelClient("http://localhost:7860", "session-2")
        client.seat_id = "seat-2"

        captured = []

        class FakeResponse:
            def __init__(self, status=200):
                self.status_code = status

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "fail",
                        request=httpx.Request("POST", "http://x"),
                        response=httpx.Response(self.status_code, request=httpx.Request("POST", "http://x")),
                    )

            def json(self):
                return {"job_id": "z", "status": "speaking"}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None, timeout=None):
                captured.append(url)
                if url.endswith("/v1/dashboard-chat"):
                    raise httpx.ConnectError("chat panel offline")
                return FakeResponse()

        monkeypatch.setattr(cc.httpx, "AsyncClient", FakeClient)

        mcp = cc.build_mcp_server(client)
        speak = mcp._tool_manager._tools["mod3_speak"].fn
        result = await speak(text="audio-only")
        # Speech must still succeed despite chat-mirror failure
        assert result == {"job_id": "z", "status": "speaking"}
        assert "http://localhost:7860/v1/speak" in captured
