"""Mod³ Channel Client — standalone stdio MCP server for Claude Code channel integration.

Architecture
------------
This process is spawned by Claude Code as a child process (stdio MCP transport).
It authenticates with the Mod³ HTTP daemon (localhost:7860) and registers a
"seat" in a mod³ session.  Events from the session are forwarded to Claude Code
via MCP notifications; Claude Code can call mod3_dashboard_post and mod3_speak
to send outbound messages.

        Claude Code
           │ spawns (stdio)
           ▼
  channel_client.py  ← this file
           │ HTTP/SSE
           ▼
  mod3 HTTP daemon (port 7860)
           │ session with N seats (one per channel_client.py child)

Auth
----
Bearer token read from ~/.mod3/channel-client.token.
On first run (no token file) a UUID token is generated, written to that file,
and registered with the mod3 daemon via POST /v1/channel-tokens.

Seat lifecycle
--------------
  startup  → POST /v1/sessions/{session_id}/seats  → get seat_id
  running  → GET  /v1/sessions/{session_id}/seats/{seat_id}/events  (SSE)
             each event → notifications/claude/channel
  shutdown → DELETE /v1/sessions/{session_id}/seats/{seat_id}

MCP surface
-----------
  capabilities:
    experimental["claude/channel"] = {}
    experimental["claude/channel/permission"] = {}
    tools = {}
  instructions: channel-tag shape + outbound tools description
  tools:
    mod3_dashboard_post(text, role?)  → POST /v1/dashboard-chat
    mod3_speak(text, voice?, speed?)  → POST /v1/speak

Usage
-----
  # Normal (spawned by Claude Code via mcp.channel.json):
  python3 clients/channel_client.py [--session SESSION_ID] [--server URL]

  # Connectivity smoke-test (exits 0 on success):
  python3 clients/channel_client.py --test [--server URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

logger = logging.getLogger("mod3.channel_client")

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

_DEFAULT_SERVER_URL = "http://localhost:7860"
_DEFAULT_SESSION_ID = "main"
_TOKEN_PATH = Path.home() / ".mod3" / "channel-client.token"


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------


def _load_or_create_token() -> str:
    """Return the persistent bearer token, generating one on first run."""
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _TOKEN_PATH.exists():
        token = _TOKEN_PATH.read_text().strip()
        if token:
            return token
    token = str(uuid.uuid4())
    _TOKEN_PATH.write_text(token + "\n")
    logger.info("Generated new channel-client token at %s", _TOKEN_PATH)
    return token


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register_seat(
    client: httpx.AsyncClient,
    server_url: str,
    session_id: str,
    token: str,
    device_uuid: str,
) -> dict[str, Any]:
    """POST /v1/sessions/{session_id}/seats — register a seat, create session if needed."""
    url = f"{server_url}/v1/sessions/{session_id}/seats"
    body = {"client_type": "claude-code-channel", "device_uuid": device_uuid}
    resp = await client.post(url, json=body, headers=_auth_headers(token), timeout=10.0)
    resp.raise_for_status()
    return resp.json()


async def _delete_seat(
    server_url: str,
    session_id: str,
    seat_id: str,
    token: str,
) -> None:
    """DELETE /v1/sessions/{session_id}/seats/{seat_id} — clean up on exit."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{server_url}/v1/sessions/{session_id}/seats/{seat_id}"
            await client.delete(url, headers=_auth_headers(token), timeout=5.0)
            logger.info("Seat %s deleted", seat_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Seat delete failed (already gone?): %s", exc)


# ---------------------------------------------------------------------------
# Channel client core
# ---------------------------------------------------------------------------


class ChannelClient:
    """Coordinates seat registration, SSE subscription, and MCP server."""

    def __init__(self, server_url: str, session_id: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.token = _load_or_create_token()
        self.device_uuid = self._load_or_create_device_uuid()
        self.seat_id: str | None = None
        self._notification_sender: Any = None  # set when MCP session starts

    def _load_or_create_device_uuid(self) -> str:
        uuid_path = _TOKEN_PATH.parent / "device-uuid"
        uuid_path.parent.mkdir(parents=True, exist_ok=True)
        if uuid_path.exists():
            val = uuid_path.read_text().strip()
            if val:
                return val
        val = str(uuid.uuid4())
        uuid_path.write_text(val + "\n")
        return val

    def set_notification_sender(self, sender: Any) -> None:
        self._notification_sender = sender

    async def send_channel_notification(self, content: str, meta: dict | None = None) -> None:
        """Forward a seat event to Claude Code as notifications/claude/channel."""
        if self._notification_sender is None:
            logger.debug("No notification sender yet; dropping event: %s", content[:80])
            return
        params: dict[str, Any] = {"content": content}
        if meta:
            params["meta"] = meta
        notification = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/claude/channel",
            params=params,
        )
        try:
            await self._notification_sender(notification)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send channel notification: %s", exc)

    async def run_sse_subscription(self) -> None:
        """Stream events from GET /v1/sessions/{session_id}/seats/{seat_id}/events."""
        if not self.seat_id:
            logger.error("No seat_id — cannot subscribe to events")
            return
        url = f"{self.server_url}/v1/sessions/{self.session_id}/seats/{self.seat_id}/events"
        headers = {**_auth_headers(self.token), "Accept": "text/event-stream"}
        logger.info("Subscribing to seat events at %s", url)
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    event_type: str | None = None
                    data_lines: list[str] = []
                    async for line in resp.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:") :].strip())
                        elif line == "":
                            # End of SSE event — process it
                            if data_lines:
                                raw = "\n".join(data_lines)
                                await self._handle_sse_event(event_type, raw)
                            event_type = None
                            data_lines = []
        except httpx.RemoteProtocolError as exc:
            logger.info("SSE stream closed: %s", exc)
        except asyncio.CancelledError:
            logger.debug("SSE subscription cancelled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SSE subscription error: %s", exc)

    async def _handle_sse_event(self, event_type: str | None, raw_data: str) -> None:
        """Parse an SSE event and forward to Claude Code."""
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = {"raw": raw_data}

        etype = event_type or payload.get("type", "event")

        if etype == "user_message":
            text = payload.get("content", "")
            meta = {
                "session_id": self.session_id,
                "seat_id": self.seat_id,
                "source": "mod3",
                "input_type": payload.get("input_type", "text"),
            }
            channel_content = (
                f'<channel source="mod3" session_id="{self.session_id}" '
                f'seat_id="{self.seat_id}" input_type="{meta["input_type"]}">'
                f"{text}</channel>"
            )
            await self.send_channel_notification(channel_content, meta)

        elif etype == "pairing_request":
            code = payload.get("code", "")
            identifier = payload.get("identifier", "")
            meta = {"session_id": self.session_id, "seat_id": self.seat_id, "source": "mod3"}
            channel_content = (
                f'<channel source="mod3" pairing_request="true" '
                f'identifier="{identifier}" code="{code}">'
                f"Pairing request from identifier '{identifier}'. "
                f"Run `/mod3:access pair {code}` to approve.</channel>"
            )
            await self.send_channel_notification(channel_content, meta)

        elif etype == "permission_request":
            meta = {"session_id": self.session_id, "seat_id": self.seat_id, "source": "mod3"}
            channel_content = (
                f'<channel source="mod3" permission_request="true" '
                f'session_id="{self.session_id}">'
                f"{json.dumps(payload)}</channel>"
            )
            await self.send_channel_notification(channel_content, meta)

        else:
            logger.debug("Unhandled SSE event type %r: %s", etype, raw_data[:200])


# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

_CHANNEL_INSTRUCTIONS = """
Mod³ Channel Client — you are connected to the Mod³ TTS daemon as a channel seat.

## Inbound channel tags

Dashboard messages arrive as MCP notifications (method: notifications/claude/channel)
with a content field shaped like:

  <channel source="mod3" session_id="main" seat_id="seat-xxxx" input_type="text">
  User's message here
  </channel>

Pairing requests look like:

  <channel source="mod3" pairing_request="true" identifier="uuid" code="abcde">
  Pairing request from identifier 'uuid'. Run `/mod3:access pair abcde` to approve.
  </channel>

## Outbound tools

  mod3_dashboard_post(text, role?) — send text to the dashboard chat panel
  mod3_speak(text, voice?, speed?) — synthesize text to speech and play it

Always reply to user messages using mod3_dashboard_post so your response appears
in the dashboard.  Use mod3_speak when voice output is appropriate.
"""


def build_mcp_server(client: ChannelClient) -> FastMCP:
    """Build the FastMCP server with channel capabilities and tools."""
    mcp = FastMCP(
        "mod3-channel",
        instructions=_CHANNEL_INSTRUCTIONS,
    )

    # Declare channel capabilities (Anthropic Channels spec)
    mcp.experimental_capabilities = {
        "claude/channel": {},
        "claude/channel/permission": {},
    }

    @mcp.tool()
    async def mod3_dashboard_post(text: str, role: str = "assistant") -> str:
        """Send text to the Mod³ dashboard chat panel.

        Args:
            text: The message text to display in the dashboard.
            role: Message role — "assistant" (default) or "user".
        """
        url = f"{client.server_url}/v1/dashboard-chat"
        body = {
            "text": text,
            "role": role,
            "session_id": client.session_id,
            "seat_id": client.seat_id,
        }
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(url, json=body, headers=_auth_headers(client.token), timeout=10.0)
                if resp.status_code in (200, 404):
                    # 404 = endpoint not yet implemented on server; degrade gracefully
                    if resp.status_code == 404:
                        logger.debug("/v1/dashboard-chat not found — falling back to /ws broadcast")
                    return "ok"
                resp.raise_for_status()
                return "ok"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @mcp.tool()
    async def mod3_speak(text: str, voice: str = "eng_uk_m_davids", speed: float = 1.0) -> dict:
        """Synthesize text to speech and play it through the Mod³ daemon.

        Hits POST /v1/speak (queue-aware endpoint). Returns immediately with a
        job token — the daemon's drain thread owns all audio playback.

        Args:
            text: Text to synthesize and speak aloud.
            voice: Voice preset name (use list_voices to discover options).
                Default: eng_uk_m_davids (Chatterbox-Turbo, British male).
            speed: Playback speed multiplier (0.5–2.0). Default: 1.0.

        Returns:
            {"job_id": str, "queue_position": int, "status": "speaking" | "queued"}
            Poll GET /v1/jobs/{job_id} for completion. Stop via POST /v1/stop.
        """
        url = f"{client.server_url}/v1/speak"
        body: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "speed": speed,
        }
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(url, json=body, headers=_auth_headers(client.token), timeout=30.0)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    return mcp


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------


async def _run_channel(server_url: str, session_id: str) -> None:
    """Full channel client: register seat, subscribe SSE, serve MCP over stdio."""
    client = ChannelClient(server_url=server_url, session_id=session_id)
    mcp = build_mcp_server(client)

    # Register seat with mod3 HTTP daemon
    async with httpx.AsyncClient() as http:
        try:
            result = await _register_seat(
                http,
                server_url=client.server_url,
                session_id=client.session_id,
                token=client.token,
                device_uuid=client.device_uuid,
            )
            client.seat_id = result["seat_id"]
            logger.info("Registered seat %s in session %s", client.seat_id, client.session_id)
        except Exception as exc:
            logger.error("Failed to register seat: %s", exc)
            sys.exit(1)

    # Run SSE subscription in background
    sse_task = asyncio.create_task(client.run_sse_subscription())

    # Cleanup on exit
    async def _cleanup(signum=None):
        sse_task.cancel()
        try:
            await sse_task
        except asyncio.CancelledError:
            pass
        if client.seat_id:
            await _delete_seat(
                server_url=client.server_url,
                session_id=client.session_id,
                seat_id=client.seat_id,
                token=client.token,
            )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_cleanup()))

    # Run MCP server over stdio, wiring the notification sender
    async with stdio_server() as (read_stream, write_stream):
        # Capture the write stream so we can send notifications
        async def _send_notification(notification: JSONRPCNotification) -> None:
            msg = SessionMessage(message=JSONRPCMessage(notification))
            await write_stream.send(msg)

        client.set_notification_sender(_send_notification)

        try:
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
        finally:
            await _cleanup()


async def _run_test(server_url: str) -> int:
    """Smoke test: check mod3 health and seat registration, exit 0 on success."""
    token = _load_or_create_token()
    device_uuid = str(uuid.uuid4())  # ephemeral for test

    print(f"Testing connection to {server_url} ...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            resp = await client.get(f"{server_url}/health")
            if resp.status_code != 200:
                print(f"FAIL: /health returned {resp.status_code}")
                return 1
            print(f"  /health OK ({resp.status_code})")

            # Seat registration
            result = await _register_seat(
                client,
                server_url=server_url,
                session_id="test",
                token=token,
                device_uuid=device_uuid,
            )
            seat_id = result.get("seat_id")
            if not seat_id:
                print(f"FAIL: no seat_id in response: {result}")
                return 1
            print(f"  seat registered: {seat_id}")

            # Cleanup
            await _delete_seat(
                server_url=server_url,
                session_id="test",
                seat_id=seat_id,
                token=token,
            )
            print(f"  seat deleted: {seat_id}")

        print("PASS: connectivity test succeeded")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("MOD3_LOG_LEVEL", "WARNING"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Mod³ Channel Client — stdio MCP server for Claude Code channel integration",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("MOD3_SERVER_URL", _DEFAULT_SERVER_URL),
        help="Mod³ HTTP server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--session",
        default=os.environ.get("MOD3_SESSION_ID") or _DEFAULT_SESSION_ID,
        help="Mod³ session ID to join (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run connectivity smoke test and exit (does not start stdio MCP server)",
    )
    args = parser.parse_args()

    # Belt-and-suspenders: if MOD3_SESSION_ID env-substitution produced an empty
    # string (the bash :- fallback semantics aren't formally guaranteed by every
    # MCP loader), fall back to the default rather than registering a seat under
    # session_id="" which would land at the malformed path /v1/sessions//seats.
    if not args.session or not args.session.strip():
        logger.warning(
            "channel_client: --session resolved to empty; falling back to %r",
            _DEFAULT_SESSION_ID,
        )
        args.session = _DEFAULT_SESSION_ID

    if args.test:
        rc = asyncio.run(_run_test(args.server))
        sys.exit(rc)
    else:
        asyncio.run(_run_channel(server_url=args.server, session_id=args.session))


if __name__ == "__main__":
    main()
