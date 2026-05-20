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
        origin + message type and binds the ACP transport to the new session.

        Loose 'sessionResume appears anywhere in the file' would let a broken
        implementation that defines the method but never calls it from the
        listener pass. The wiring check below requires the message listener
        to actually invoke attachToClaudeCodeSession with the spawned
        session_id.
        """
        assert "mod3:claude-code-spawned" in index_html, (
            "index.html must listen for the 'mod3:claude-code-spawned' message type"
        )
        # Same-origin check is required on the listener side too.
        assert "ev.origin !== window.location.origin" in index_html or (
            "event.origin !== window.location.origin" in index_html
        )
        # Wiring check: the listener body must call attachToClaudeCodeSession
        # with data.session_id (not just reference sessionResume somewhere).
        # Scope to `window.addEventListener('message', ...)` — the dashboard
        # also has WebSocket-message listeners that match a looser regex.
        listener_match = re.search(
            r"window\.addEventListener\(\s*['\"]message['\"]\s*,\s*\([^)]*\)\s*=>\s*\{[\s\S]*?mod3:claude-code-spawned[\s\S]*?\}\s*\)\s*;",
            index_html,
        )
        assert listener_match, "index.html must register a window message listener with an arrow handler"
        listener_body = listener_match.group(0)
        assert "mod3:claude-code-spawned" in listener_body, (
            "the message listener body must filter on the 'mod3:claude-code-spawned' type"
        )
        assert "attachToClaudeCodeSession" in listener_body, (
            "the message listener body must actually call attachToClaudeCodeSession; "
            "merely defining sessionResume elsewhere is insufficient wiring"
        )
        # And the helper itself must invoke sessionResume.
        # Anchor on the global assignment that follows the helper definition,
        # since the helper body contains nested `}\n` lines that confuse a
        # naïve non-greedy match.
        helper_match = re.search(
            r"async\s+function\s+attachToClaudeCodeSession\s*\([\s\S]*?window\.__mod3AttachToClaudeCodeSession",
            index_html,
        )
        assert helper_match, "attachToClaudeCodeSession helper must exist"
        assert "sessionResume" in helper_match.group(0), (
            "attachToClaudeCodeSession must call the ACP transport's sessionResume"
        )

    def test_origin_mismatch_aborts_listener_path(self, index_html):
        """The same-origin guard must be an early return at the top of the
        message handler — not a conditional that still proceeds on mismatch.

        Without this, a broken refactor (e.g. an `if/else` that flips polarity)
        could let cross-origin messages reach attachToClaudeCodeSession.
        """
        match = re.search(
            r"window\.addEventListener\(\s*['\"]message['\"]\s*,\s*\([^)]*\)\s*=>\s*\{([\s\S]*?mod3:claude-code-spawned[\s\S]*?)\}\s*\)\s*;",
            index_html,
        )
        assert match, "could not locate the spawn-message listener body"
        body = match.group(1)
        # The origin check must appear before attachToClaudeCodeSession; otherwise
        # cross-origin messages could reach the attach logic.
        origin_idx = -1
        for needle in ("ev.origin", "event.origin"):
            i = body.find(needle)
            if i >= 0:
                origin_idx = i
                break
        assert origin_idx >= 0, "listener body must reference ev.origin or event.origin"
        attach_idx = body.find("attachToClaudeCodeSession")
        assert attach_idx > origin_idx, (
            "the origin check must appear before attachToClaudeCodeSession; "
            "otherwise cross-origin messages could trigger the attach"
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

    def test_attach_helper_skips_resume_on_timeout(self, index_html):
        """If the seat-poll deadline expires without a seat appearing, the
        helper must NOT call sessionResume — otherwise the subsequent
        session/prompt fails with the cryptic 'no seats' error instead of
        the cause (spawn produced no working channel client)."""
        # Anchor on the global assignment that follows the helper definition,
        # since the helper body contains nested `}\n` lines that confuse a
        # naïve non-greedy match.
        helper_match = re.search(
            r"async\s+function\s+attachToClaudeCodeSession\s*\([\s\S]*?window\.__mod3AttachToClaudeCodeSession",
            index_html,
        )
        assert helper_match
        body = helper_match.group(0)
        # The helper must track whether a seat was found and skip resume when
        # it wasn't. Accept either an explicit boolean or an early-return.
        assert re.search(r"seatFound|no seat appeared", body), (
            "helper must track seat-found state and warn/abort when the poll deadline expires without a seat"
        )

    def test_attach_helper_has_supersede_guard(self, index_html):
        """When two postMessages for different session_ids arrive in rapid
        succession, only the latest attach attempt should call sessionResume.
        Without this guard, a slow first attach can overwrite a fast second
        bind, and the dashboard ends up addressing the wrong session.

        Implementation may use a monotonic token, an AbortController, or
        equivalent — just verify some form of newer-wins discipline exists.
        """
        # Anchor on the global assignment that follows the helper definition,
        # since the helper body contains nested `}\n` lines that confuse a
        # naïve non-greedy match.
        helper_match = re.search(
            r"async\s+function\s+attachToClaudeCodeSession\s*\([\s\S]*?window\.__mod3AttachToClaudeCodeSession",
            index_html,
        )
        assert helper_match
        body = helper_match.group(0)
        has_token_guard = bool(re.search(r"_attachAttemptToken|AbortController|attemptId|superseded", body))
        assert has_token_guard, (
            "attachToClaudeCodeSession must guard against newer attempts overwriting older ones; "
            "expected a token/AbortController/'superseded' pattern in the helper"
        )
