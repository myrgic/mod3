"""Tests for the ACP session/update spec-compliant nested wire shape.

Fix 2 in feat/session-auto-create-and-acp-wire-fix:

Per [[reference/acp-protocol-spec]] (schema.json SessionNotification), the
session/update params must be:

    {"sessionId": "...", "update": {"sessionUpdate": "...", "content": {...}}}

mod3 previously emitted a FLAT shape:

    {"sessionId": "...", "sessionUpdate": "...", "content": {...}}

Both server (schemas/acp/notifications.py) and client (dashboard/acp-transport.js)
agreed on the wrong shape, so mod3 worked internally but any compliant external
ACP client (e.g. Zed) would misparse the notifications.

These tests pin the corrected nested form:
  - Server-side: SessionUpdateNotification emits nested params.
  - Client-side: acp-transport.js _handleNotification reads nested params.
  - Integration: end-to-end notification shape matches spec.

Run with: PYTHONPATH=. .venv/bin/python -m pytest tests/test_acp_wire_shape.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.acp.notifications import (
    SessionUpdateNotification,
    SessionUpdateParams,
    SessionUpdatePayload,
)
from schemas.acp.content import TextContent


# ---------------------------------------------------------------------------
# Server-side emission: SessionUpdateNotification
# ---------------------------------------------------------------------------


class TestServerSideEmission:
    """SessionUpdateNotification must emit the spec-compliant nested shape."""

    def test_text_chunk_params_has_update_key(self):
        notif = SessionUpdateNotification.text_chunk(session_id="sid1", text="hello")
        data = notif.model_dump()
        params = data["params"]
        assert "update" in params, "params must contain 'update' key per ACP spec"

    def test_text_chunk_session_id_at_params_root(self):
        notif = SessionUpdateNotification.text_chunk(session_id="sid2", text="x")
        data = notif.model_dump()
        assert data["params"]["sessionId"] == "sid2"

    def test_text_chunk_update_contains_session_update_kind(self):
        notif = SessionUpdateNotification.text_chunk(session_id="s", text="t")
        data = notif.model_dump()
        update = data["params"]["update"]
        assert update["sessionUpdate"] == "agent_message_chunk"

    def test_text_chunk_update_contains_content(self):
        notif = SessionUpdateNotification.text_chunk(session_id="s", text="chunk text")
        data = notif.model_dump()
        content = data["params"]["update"]["content"]
        assert content["type"] == "text"
        assert content["text"] == "chunk text"

    def test_flat_keys_absent_from_params_root(self):
        """The old flat shape must not appear — guards against regression."""
        notif = SessionUpdateNotification.text_chunk(session_id="s", text="t")
        data = notif.model_dump()
        params = data["params"]
        assert "sessionUpdate" not in params, (
            "sessionUpdate must be inside params.update, not at params root (old flat shape)"
        )
        assert "content" not in params, (
            "content must be inside params.update, not at params root (old flat shape)"
        )

    def test_json_serialization_round_trip(self):
        notif = SessionUpdateNotification.text_chunk(session_id="rt-sid", text="round trip")
        raw = notif.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["method"] == "session/update"
        assert parsed["params"]["sessionId"] == "rt-sid"
        assert parsed["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        assert parsed["params"]["update"]["content"]["text"] == "round trip"

    def test_custom_update_kind_preserved(self):
        """Non-default sessionUpdate kinds are preserved in nested shape."""
        notif = SessionUpdateNotification(
            params=SessionUpdateParams(
                sessionId="s",
                update=SessionUpdatePayload(sessionUpdate="thought_chunk"),
            )
        )
        data = notif.model_dump()
        assert data["params"]["update"]["sessionUpdate"] == "thought_chunk"

    def test_notification_has_no_id(self):
        """session/update is a notification — it must not carry an id."""
        notif = SessionUpdateNotification.text_chunk(session_id="s", text="t")
        data = notif.model_dump()
        assert "id" not in data


# ---------------------------------------------------------------------------
# Client-side parsing: acp-transport.js _handleNotification
#
# We test the JavaScript parsing logic indirectly by verifying that the exact
# JSON shape that _handleNotification now reads (params.update.sessionUpdate,
# params.update.content) is produced by the server and parseable.
# ---------------------------------------------------------------------------


class TestClientSideParseShape:
    """Verify the JSON shape the client's _handleNotification now reads."""

    def test_server_produces_shape_client_expects(self):
        """Server emission produces params.update.{sessionUpdate,content}
        which matches the updated _handleNotification: update = params.update."""
        notif = SessionUpdateNotification.text_chunk(session_id="s", text="Hi")
        wire = json.loads(notif.model_dump_json())

        # Simulate _handleNotification logic:
        params = wire.get("params", {})
        update = params.get("update", {})  # the updated client reads params.update
        update_kind = update.get("sessionUpdate")
        content = update.get("content", {})

        assert update_kind == "agent_message_chunk"
        assert content.get("type") == "text"
        assert content.get("text") == "Hi"

    def test_old_flat_shape_would_fail_new_client(self):
        """Guard test: a message with the OLD flat shape must not yield
        update_kind via the new client read path."""
        old_flat_msg = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s",
                "sessionUpdate": "agent_message_chunk",  # old flat key
                "content": {"type": "text", "text": "stale"},
            },
        }
        # New client reads params.update (not params directly)
        params = old_flat_msg["params"]
        update = params.get("update", {})  # will be {} for old shape
        update_kind = update.get("sessionUpdate")

        # The new client would not find the update_kind in the old shape,
        # confirming the old messages would be silently ignored (no-op,
        # not a crash — which is the correct behavior for unknown shapes).
        assert update_kind is None


# ---------------------------------------------------------------------------
# Integration: full message envelope
# ---------------------------------------------------------------------------


class TestFullNotificationEnvelope:
    def test_complete_wire_message_matches_spec(self):
        """Full JSON-RPC notification envelope matches the ACP spec exactly."""
        notif = SessionUpdateNotification.text_chunk(session_id="mod3-abc123", text="streaming response ")
        wire = json.loads(notif.model_dump_json())

        expected_shape = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "mod3-abc123",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "streaming response "},
                },
            },
        }
        # Check structural equivalence (id is absent from both)
        assert wire["jsonrpc"] == expected_shape["jsonrpc"]
        assert wire["method"] == expected_shape["method"]
        assert wire["params"]["sessionId"] == expected_shape["params"]["sessionId"]
        assert wire["params"]["update"]["sessionUpdate"] == expected_shape["params"]["update"]["sessionUpdate"]
        assert wire["params"]["update"]["content"] == expected_shape["params"]["update"]["content"]
        assert "id" not in wire
