"""Regression tests for Claude Code session_id ↔ mod3 seat binding.

The "close the loop" wiring landed 2026-05-19. Two paths bind a mod3 seat
to a real Claude Code session_id instead of collapsing into the legacy
"main" sentinel:

1. Manual launch: ``claude --dangerously-load-development-channels server:mod3``
   uses the repo's ``mcp.channel.json``. The ``env`` block substitutes
   ``${CLAUDE_CODE_SESSION_ID}`` so each Claude Code session gets a distinct
   ``MOD3_SESSION_ID`` and channel_client.py registers its seat under that id.

2. Dashboard spawn: ``sessions.html`` posts to ``/v1/claude-code/spawn`` and on
   success ``window.opener.postMessage`` notifies the dashboard tab, which
   calls ``acpTransport.sessionResume(session_id)`` to bind its ACP connection
   to the new session's seats.

These tests guard the static artefacts (mcp.channel.json + the three dashboard
files) — server-side ACP ``session/resume`` semantics already have coverage in
``test_acp_session_management.py``.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_claude_session_id_binding.py -v
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Path 1 — mcp.channel.json env-pass
# ---------------------------------------------------------------------------


class TestMcpChannelEnvPass:
    @pytest.fixture(scope="class")
    def cfg(self):
        return json.loads((REPO_ROOT / "mcp.channel.json").read_text())

    def test_mod3_server_entry_exists(self, cfg):
        assert "mcpServers" in cfg
        assert "mod3" in cfg["mcpServers"]

    def test_env_block_present(self, cfg):
        mod3 = cfg["mcpServers"]["mod3"]
        assert "env" in mod3, "mcp.channel.json mod3 entry must declare an env block"

    def test_session_id_pulled_from_harness(self, cfg):
        """MOD3_SESSION_ID must substitute from CLAUDE_CODE_SESSION_ID with a
        'main' fallback so manually-launched Claude Code sessions are bound to
        their harness session_id rather than colliding on 'main'."""
        env = cfg["mcpServers"]["mod3"]["env"]
        assert "MOD3_SESSION_ID" in env
        value = env["MOD3_SESSION_ID"]
        # Accept either ${VAR:-default} (preferred) or ${VAR} forms; both flow
        # through Claude Code's env-substitution. The 'main' fallback is
        # important for backward compatibility when the harness env isn't set.
        assert value.startswith("${CLAUDE_CODE_SESSION_ID"), (
            f"MOD3_SESSION_ID must reference CLAUDE_CODE_SESSION_ID, got: {value!r}"
        )
        assert ":-main" in value or "}" == value[-1], (
            f"MOD3_SESSION_ID should fall back to 'main' if the harness env is unset, got: {value!r}"
        )


# ---------------------------------------------------------------------------
# Path 2 — dashboard postMessage bridge
# ---------------------------------------------------------------------------


class TestDashboardPostMessageBridge:
    @pytest.fixture(scope="class")
    def acp_transport_js(self):
        return (REPO_ROOT / "dashboard" / "acp-transport.js").read_text()

    @pytest.fixture(scope="class")
    def sessions_html(self):
        return (REPO_ROOT / "dashboard" / "sessions.html").read_text()

    @pytest.fixture(scope="class")
    def index_html(self):
        return (REPO_ROOT / "dashboard" / "index.html").read_text()

    def test_acp_transport_exposes_session_resume(self, acp_transport_js):
        """AcpTransport.sessionResume must exist and issue a session/resume
        JSON-RPC request, then store the sessionId for subsequent prompts."""
        assert re.search(r"\basync\s+sessionResume\s*\(", acp_transport_js), (
            "AcpTransport class must define an async sessionResume(sessionId) method"
        )
        assert "'session/resume'" in acp_transport_js or '"session/resume"' in acp_transport_js
        # The method must persist _sessionId so sessionPrompt() finds it.
        assert re.search(
            r"sessionResume[\s\S]{0,400}this\._sessionId\s*=",
            acp_transport_js,
        ), "sessionResume must assign this._sessionId from the result"

    def test_sessions_html_posts_to_opener(self, sessions_html):
        """sessions.html must postMessage to window.opener after a successful
        spawn so the dashboard tab can attach via ACP."""
        assert "window.opener" in sessions_html
        assert "postMessage" in sessions_html
        assert "mod3:claude-code-spawned" in sessions_html, (
            "sessions.html must use the 'mod3:claude-code-spawned' message type (matched by index.html's listener)"
        )
        # Same-origin guard — must scope the postMessage to window.location.origin
        # rather than '*' to avoid leaking the spawn details to unrelated frames.
        assert "window.location.origin" in sessions_html

    def test_index_html_listens_for_spawn_message(self, index_html):
        """index.html must register a message listener that filters on
        origin + message type and binds the ACP transport to the new session."""
        assert "mod3:claude-code-spawned" in index_html, (
            "index.html must listen for the 'mod3:claude-code-spawned' message type"
        )
        # Same-origin check is required on the listener side too.
        assert "ev.origin !== window.location.origin" in index_html or (
            "event.origin !== window.location.origin" in index_html
        )
        # Must call sessionResume (via the helper) to actually attach.
        assert "sessionResume" in index_html, (
            "index.html must call acpTransport.sessionResume after receiving the spawn message"
        )

    def test_attach_helper_waits_for_seat(self, index_html):
        """The attach helper must poll for the channel-client seat to register
        before issuing session/resume, since the spawned subprocess needs a
        moment to launch and POST its seat. Otherwise session/prompt would
        fail with 'No channel-client seats attached' immediately after spawn."""
        # Match the fetch against the seats endpoint with the session_id.
        assert re.search(
            r"/v1/sessions/\$\{[^}]+\}/seats|/v1/sessions/.+seats",
            index_html,
        ), "attach helper must poll /v1/sessions/<id>/seats before calling sessionResume"
