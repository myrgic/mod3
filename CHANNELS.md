# Mod³ as a Claude Code Channel

Mod³ implements the [Anthropic Channels primitive](https://code.claude.com/docs/en/channels),
making it a first-class bidirectional channel between the dashboard and a Claude Code session.

## Two modes of operation

| Mode | How to start | What happens |
|------|-------------|--------------|
| **Standalone** (default) | `python3 server.py --http` | Dashboard text routes to the local AgentLoop |
| **Claude Code Channel** | `claude --dangerously-load-development-channels server:mod3` | Dashboard text arrives in Claude Code as `<channel>` tags |

## Starting as a Claude Code Channel (development)

```bash
# From the mod3 repo directory:
claude --dangerously-load-development-channels server:mod3
```

Claude Code spawns `python3 server.py --channel` as a subprocess using the
config in `mcp.channel.json`. The server declares `capabilities.experimental['claude/channel'] = {}`
at init, which registers the notification listener in Claude Code.

Once Anthropic accepts mod3 into their plugin marketplace, the command becomes:
```bash
claude --channels plugin:mod3@myrgic-plugins
```

## Enabling channel mode in the dashboard

By default the dashboard still routes to the local AgentLoop even when started
from Claude Code. To route text input to Claude Code instead:

```bash
# Option 1: env var (no config file needed)
export MOD3_CHANNEL_MODE=claude-code
python3 server.py --channel

# Option 2: config file
mkdir -p ~/.claude/channels/mod3
cat > ~/.claude/channels/mod3/config.json <<'EOF'
{
  "channel_mode": "claude-code",
  "server_url": "http://localhost:7860"
}
EOF
```

Or use the `/mod3:configure` skill from inside your Claude Code session.

## Message flow

### Inbound (dashboard → Claude Code)

When the user types in the dashboard text box, Claude Code receives:

```xml
<channel source="mod3" session_id="browser:a1b2c3d4" input_type="text">
Hello, Claude
</channel>
```

Voice transcripts use the same mechanism with different meta attributes:

```xml
<channel source="mod3" speaker="user" confidence="0.95" input_type="voice">
What is the weather like today
</channel>
```

### Outbound (Claude Code → dashboard)

Claude replies using one or both MCP tools:

```python
# Post text to the dashboard chat panel
mod3_dashboard_post(text="Here is the answer...")

# Play audio through the user's speakers (non-blocking)
speak(text="Working on it, give me a moment.")

# Both simultaneously (the power move)
speak(text="Found three issues.")          # user hears immediately
mod3_dashboard_post(text="Issue list: ...") # structured output appears in chat
```

## Access control

Access is governed by `~/.claude/channels/mod3/access.json`.

**Localhost connections (same machine) are auto-allowed** regardless of policy.
This is the common case: browser dashboard on the same laptop as Claude Code.

**Remote clients** (future: mobile app, LAN browser) must pair:

1. Client connects; Mod³ emits a pairing request to Claude Code:
   ```xml
   <channel source="mod3" pairing_request="true" identifier="uuid" code="abcde">
   Pairing request from identifier 'uuid'. Run /mod3:access pair abcde to approve.
   </channel>
   ```

2. Operator approves from their Claude Code terminal (not through the channel):
   ```
   /mod3:access pair abcde
   ```

3. Future connections from that identifier are allowed without re-pairing.

See the `/mod3:access` skill for full subcommand reference.

## Skills

| Skill | Purpose |
|-------|---------|
| `/mod3:configure` | Set channel mode, server URL, voice defaults |
| `/mod3:access` | Manage access.json: pair, policy, allow, list |
