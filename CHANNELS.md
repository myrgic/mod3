# Mod³ as a Claude Code Channel

Mod³ implements the [Anthropic Channels primitive](https://code.claude.com/docs/en/channels)
as a **separated client process** — the canonical architecture for clean decoupling
between a service daemon and a per-session Claude Code link.

## Architecture

```
Claude Code session A         Claude Code session B
   | spawn child                  | spawn child
   v                              v
channel_client.py (stdio)      channel_client.py (stdio)
   | authenticated link           | authenticated link
   +------------------+-----------+
                      v
        mod3 HTTP daemon (port 7860)
                      |
        session has N "seats" — one per attached channel client
                      |
        dashboard input -> fans to seats per session policy
```

Each Claude Code session spawns its own `clients/channel_client.py` child process.
That child registers a seat in the mod3 session, subscribes to the seat's SSE event
stream, and forwards events to Claude Code as `notifications/claude/channel`.

This is structurally different from the old `server.py --channel` mode (removed in
this PR — see issue #11 and PR #39 supersession). The daemon stays HTTP-only; the
stdio interface lives entirely in the channel client.

## Starting as a Claude Code Channel (development)

```bash
# From the mod3 repo directory — Claude Code spawns channel_client.py per mcp.channel.json:
claude --dangerously-load-development-channels server:mod3
```

Once mod3 is in the Anthropic plugin marketplace:
```bash
claude --channels plugin:mod3@myrgic-plugins
```

The `mcp.channel.json` file at the repo root configures the channel client entrypoint.
Claude Code spawns `python3 clients/channel_client.py` as the child process.

## Two modes of operation

| Mode | How to start | What happens |
|------|-------------|--------------|
| **Standalone** (default) | `python3 server.py --http` | Dashboard text routes to the local AgentLoop |
| **Claude Code Channel** | `claude --dangerously-load-development-channels server:mod3` | Dashboard text arrives in Claude Code as `<channel>` tags via the channel client |

## Enabling channel mode in the dashboard

Configure via the `/mod3:configure` skill inside your Claude Code session, or manually:

```bash
mkdir -p ~/.claude/channels/mod3
cat > ~/.claude/channels/mod3/config.json <<'EOF'
{
  "channel_mode": "claude-code",
  "server_url": "http://localhost:7860",
  "voice": "bm_lewis",
  "speed": 1.25
}
EOF
```

## Message flow

### Inbound (dashboard -> Claude Code)

When the user types in the dashboard text box, the channel client receives a seat SSE
event, then delivers it to Claude Code as:

```xml
<channel source="mod3" session_id="main" seat_id="seat-xxxx" input_type="text">
Hello, Claude
</channel>
```

Voice transcripts use the same path with `input_type="voice"`.

### Outbound (Claude Code -> dashboard)

Claude replies using the channel client's tools:

```python
# Post text to the dashboard chat panel
mod3_dashboard_post(text="Here is the answer...")

# Play audio through the user's speakers (non-blocking, proxied to HTTP daemon)
mod3_speak(text="Working on it, give me a moment.", voice="bm_lewis")
```

Both can be called together. `mod3_speak` routes to `POST /v1/synthesize` on the daemon.
`mod3_dashboard_post` routes to `POST /v1/dashboard-chat`.

## Seat lifecycle

Each channel client subprocess goes through:

1. `POST /v1/sessions/{session_id}/seats` — register seat, access check (access.py)
2. `GET  /v1/sessions/{session_id}/seats/{seat_id}/events` — SSE subscription
3. Events received -> `notifications/claude/channel` forwarded to Claude Code
4. `DELETE /v1/sessions/{session_id}/seats/{seat_id}` — on exit or signal

Multiple seats can attach to the same session (multiple Claude Code windows, one mod3 session).
Fan-out policy in v0: broadcast to all seats.

## Connectivity test

```bash
python3 clients/channel_client.py --test
# or against a remote server:
python3 clients/channel_client.py --test --server http://mod3.local:7860
```

## Access control

Access is governed by `~/.claude/channels/mod3/access.json` via `access.py`.

**Localhost connections (same machine) are auto-allowed** regardless of policy.

**Remote clients** must pair:

1. Client connects; the seat registration fails with HTTP 403 and a pairing code.
   The daemon fans a `pairing_request` event to existing seats in the session:
   ```xml
   <channel source="mod3" pairing_request="true" identifier="uuid" code="abcde">
   Pairing request from identifier 'uuid'. Run /mod3:access pair abcde to approve.
   </channel>
   ```

2. Operator approves from their Claude Code terminal (not through the channel):
   ```
   /mod3:access pair abcde
   ```

3. Future connections from that identifier pass `access.is_allowed()`.

See the `/mod3:access` skill for full subcommand reference.

## New HTTP endpoints (added alongside this channel implementation)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/sessions/{id}/seats` | Register a channel-client seat |
| `DELETE` | `/v1/sessions/{id}/seats/{seat_id}` | Revoke a seat |
| `GET` | `/v1/sessions/{id}/seats/{seat_id}/events` | SSE event stream |
| `GET` | `/v1/sessions/{id}/seats` | List seats in session |
| `POST` | `/v1/sessions/{id}/messages` | Fan dashboard text to all seats |
| `POST` | `/v1/dashboard-chat` | REST outbound (used by mod3_dashboard_post tool) |

## Skills

| Skill | Purpose |
|-------|---------|
| `/mod3:configure` | Set channel mode, server URL, voice defaults |
| `/mod3:access` | Manage access.json: pair, policy, allow, list |

## What changed from PR #39

PR #39 added `--channel` mode to `server.py`, conflating service and client in the
daemon process. That approach was structurally incorrect (see issue #11 — stdio
deprecation was intentional). This implementation supersedes PR #39 with:

- `clients/channel_client.py` — separate stdio MCP process (correct shape)
- `seats.py` — in-memory seat registry for the HTTP daemon
- `access.py` — transport-agnostic access control (ported from PR #39 as-is)
- `server.py` — cleaned up: no `--channel` mode, no monkey-patch, HTTP-only
