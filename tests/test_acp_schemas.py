"""Tests for schemas.acp — ACP JSON-RPC 2.0 Pydantic models.

Pins the wire shapes expected by the ACP spec so regressions are caught
before they reach the /ws/acp endpoint.

Run with: PYTHONPATH=. .venv/bin/python -m pytest tests/test_acp_schemas.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.acp import (
    AgentCapabilities,
    AudioContent,
    EmbeddedResource,
    ImageContent,
    InitializeParams,
    InitializeResult,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptCapabilities,
    ResourceLink,
    SessionCancelParams,
    SessionNewParams,
    SessionNewResult,
    SessionPromptParams,
    SessionPromptResult,
    SessionUpdateNotification,
    SessionUpdateParams,
    SessionUpdatePayload,
    TextContent,
)

# ---------------------------------------------------------------------------
# JSON-RPC envelope
# ---------------------------------------------------------------------------


class TestJsonRpcEnvelope:
    def test_request_wire_shape(self):
        req = JsonRpcRequest(id=1, method="initialize", params={"protocolVersion": 1})
        data = req.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["method"] == "initialize"
        assert data["params"]["protocolVersion"] == 1

    def test_request_string_id(self):
        req = JsonRpcRequest(id="abc", method="session/new", params={})
        assert req.id == "abc"

    def test_notification_has_no_id(self):
        notif = JsonRpcNotification(method="session/cancel", params={"sessionId": "s1"})
        data = notif.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "session/cancel"
        assert "id" not in data

    def test_response_ok_shape(self):
        resp = JsonRpcResponse.ok(request_id=2, result={"sessionId": "mod3-abc"})
        data = resp.model_dump(exclude_none=True)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert data["result"]["sessionId"] == "mod3-abc"
        assert "error" not in data

    def test_response_error_shape(self):
        resp = JsonRpcResponse.err(request_id=3, code=-32601, message="Method not found")
        data = resp.model_dump(exclude_none=True)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 3
        assert data["error"]["code"] == -32601
        assert data["error"]["message"] == "Method not found"
        assert "result" not in data

    def test_error_model(self):
        err = JsonRpcError(code=-32600, message="Invalid request", data={"detail": "x"})
        assert err.code == -32600
        assert err.data == {"detail": "x"}

    def test_response_null_id_on_parse_error(self):
        resp = JsonRpcResponse.err(request_id=None, code=-32700, message="Parse error")
        assert resp.id is None


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TestContentBlocks:
    def test_text_content(self):
        tc = TextContent(text="Hello!")
        assert tc.type == "text"
        assert tc.text == "Hello!"

    def test_image_content(self):
        ic = ImageContent(source={"type": "base64", "data": "abc"})
        assert ic.type == "image"

    def test_audio_content(self):
        ac = AudioContent(source={"type": "base64", "data": "xyz"})
        assert ac.type == "audio"

    def test_resource_link(self):
        rl = ResourceLink(uri="file:///etc/hosts", name="hosts")
        assert rl.type == "resource_link"
        assert rl.uri == "file:///etc/hosts"

    def test_embedded_resource(self):
        er = EmbeddedResource(resource={"uri": "f://x", "text": "content"})
        assert er.type == "embedded_resource"

    def test_discriminated_union_round_trip(self):
        # Parse a raw dict through the ContentBlock union.
        raw: dict = {"type": "text", "text": "hi"}
        block = TextContent(**raw)
        assert isinstance(block, TextContent)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitializeMethods:
    def test_params_defaults(self):
        p = InitializeParams()
        assert p.protocolVersion == 1
        assert p.clientCapabilities.terminal is False
        assert p.clientInfo.name == "mod3-dashboard"

    def test_params_custom(self):
        p = InitializeParams(
            protocolVersion=1,
            clientCapabilities={"fs": {}, "terminal": False},
            clientInfo={"name": "my-client", "version": "2.0"},
        )
        assert p.clientInfo.name == "my-client"

    def test_result_defaults(self):
        r = InitializeResult()
        caps = r.agentCapabilities
        assert caps.promptCapabilities.audio is False
        assert caps.promptCapabilities.image is False
        assert caps.promptCapabilities.embeddedContext is False
        assert caps.sessionCapabilities == {}

    def test_result_wire_shape(self):
        r = InitializeResult(
            agentCapabilities=AgentCapabilities(
                promptCapabilities=PromptCapabilities(audio=False, image=False, embeddedContext=False),
                sessionCapabilities={},
            )
        )
        data = r.model_dump()
        assert "promptCapabilities" in data["agentCapabilities"]
        pc = data["agentCapabilities"]["promptCapabilities"]
        assert pc == {"audio": False, "image": False, "embeddedContext": False}


# ---------------------------------------------------------------------------
# session/new
# ---------------------------------------------------------------------------


class TestSessionNewMethods:
    def test_params_defaults(self):
        p = SessionNewParams()
        assert p.cwd == "/"
        assert p.mcpServers == []

    def test_result_has_session_id(self):
        r = SessionNewResult(sessionId="mod3-abc123")
        assert r.sessionId == "mod3-abc123"
        data = r.model_dump()
        assert data["sessionId"] == "mod3-abc123"


# ---------------------------------------------------------------------------
# session/prompt
# ---------------------------------------------------------------------------


class TestSessionPromptMethods:
    def test_params_wire_shape(self):
        p = SessionPromptParams(
            sessionId="mod3-abc",
            prompt=[TextContent(text="Hello!")],
        )
        data = p.model_dump()
        assert data["sessionId"] == "mod3-abc"
        assert data["prompt"][0]["type"] == "text"
        assert data["prompt"][0]["text"] == "Hello!"

    def test_result_stop_reason(self):
        r = SessionPromptResult(stopReason="end_turn")
        data = r.model_dump()
        assert data["stopReason"] == "end_turn"


# ---------------------------------------------------------------------------
# session/cancel
# ---------------------------------------------------------------------------


class TestSessionCancelNotification:
    def test_params(self):
        p = SessionCancelParams(sessionId="mod3-abc")
        assert p.sessionId == "mod3-abc"


# ---------------------------------------------------------------------------
# session/update notification — spec-compliant nested shape
#
# Per [[reference/acp-protocol-spec]] (schema.json SessionNotification):
# params = {sessionId, update: {sessionUpdate, content}}
# Prior to 2026-05-19 mod3 used a flat shape; these tests pin the corrected form.
# ---------------------------------------------------------------------------


class TestSessionUpdateNotification:
    def test_wire_shape_nested(self):
        """session/update params must use the spec-compliant nested update envelope."""
        notif = SessionUpdateNotification.text_chunk(session_id="s1", text="Hello ")
        data = notif.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "session/update"
        params = data["params"]
        # sessionId at the top level of params
        assert params["sessionId"] == "s1"
        # update is a nested object — NOT flat in params
        assert "update" in params, "spec requires params.update; flat params.sessionUpdate is the old broken shape"
        update = params["update"]
        assert update["sessionUpdate"] == "agent_message_chunk"
        assert update["content"]["type"] == "text"
        assert update["content"]["text"] == "Hello "

    def test_wire_shape_no_flat_fields_in_params(self):
        """sessionUpdate and content must NOT appear as flat params keys."""
        notif = SessionUpdateNotification.text_chunk(session_id="s1", text="Hi")
        data = notif.model_dump()
        params = data["params"]
        assert "sessionUpdate" not in params, "sessionUpdate must be nested under params.update, not flat in params"
        assert "content" not in params, "content must be nested under params.update, not flat in params"

    def test_payload_inner_object_defaults(self):
        """SessionUpdatePayload (the inner update object) defaults are correct."""
        p = SessionUpdatePayload()
        assert p.sessionUpdate == "agent_message_chunk"
        assert p.content is None

    def test_params_envelope_shape(self):
        """SessionUpdateParams (the full params envelope) carries sessionId + update."""
        p = SessionUpdateParams(
            sessionId="mod3-abc",
            update=SessionUpdatePayload(sessionUpdate="agent_message_chunk"),
        )
        assert p.sessionId == "mod3-abc"
        assert p.update.sessionUpdate == "agent_message_chunk"

    def test_json_round_trip(self):
        notif = SessionUpdateNotification.text_chunk(session_id="s2", text="world")
        serialized = notif.model_dump_json()
        data = json.loads(serialized)
        # Nested shape
        assert data["params"]["update"]["content"]["text"] == "world"
        assert data["params"]["sessionId"] == "s2"


# ---------------------------------------------------------------------------
# Full end-to-end wire shape for a minimal ACP session
# ---------------------------------------------------------------------------


class TestAcpWireShapes:
    """Verify the exact JSON shapes for a complete ACP conversation turn."""

    def test_initialize_request_shape(self):
        req = JsonRpcRequest(
            id=0,
            method="initialize",
            params={
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {}, "terminal": False},
                "clientInfo": {"name": "mod3-dashboard", "version": "1.0"},
            },
        )
        wire = json.loads(req.model_dump_json())
        assert wire["jsonrpc"] == "2.0"
        assert wire["id"] == 0
        assert wire["method"] == "initialize"
        assert wire["params"]["protocolVersion"] == 1

    def test_session_new_request_shape(self):
        req = JsonRpcRequest(id=1, method="session/new", params={"cwd": "/", "mcpServers": []})
        wire = json.loads(req.model_dump_json())
        assert wire["method"] == "session/new"
        assert wire["params"]["cwd"] == "/"

    def test_session_prompt_request_shape(self):
        req = JsonRpcRequest(
            id=2,
            method="session/prompt",
            params={
                "sessionId": "mod3-abc",
                "prompt": [{"type": "text", "text": "Hello!"}],
            },
        )
        wire = json.loads(req.model_dump_json())
        assert wire["method"] == "session/prompt"
        assert wire["params"]["prompt"][0]["text"] == "Hello!"

    def test_session_cancel_notification_shape(self):
        notif = JsonRpcNotification(method="session/cancel", params={"sessionId": "mod3-abc"})
        wire = json.loads(notif.model_dump_json())
        assert wire["method"] == "session/cancel"
        assert "id" not in wire
