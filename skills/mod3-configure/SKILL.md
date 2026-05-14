---
name: mod3-configure
description: Set up the mod3 Claude Code Channel — saves config to ~/.claude/channels/mod3/config.json. Use when the user wants to enable claude-code channel mode, set the server URL, configure voice defaults, or check channel status. Triggers on "configure mod3 channel", "enable channel mode", "set mod3 URL", "mod3 setup".
---

# /mod3:configure — Configure the Mod³ Claude Code Channel

Mod³ can operate in two modes:

| Mode | Description |
|------|-------------|
| `local-agent` | Default. Mod³ routes dashboard text to its own local AgentLoop. No Claude Code involvement. |
| `claude-code` | Channel mode. Dashboard text arrives in Claude Code as `<channel source="mod3" ...>` tags. Claude replies via `mod3_dashboard_post` and/or `speak`. |

## Config file

`~/.claude/channels/mod3/config.json` — created/updated by this skill.

```json
{
  "channel_mode": "claude-code",
  "server_url": "http://localhost:7860",
  "voice": "bm_lewis",
  "speed": 1.25
}
```

Fields:

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `channel_mode` | `"claude-code"` / `"local-agent"` | `"local-agent"` | Routing mode |
| `server_url` | URL string | `"http://localhost:7860"` | Mod³ HTTP base URL |
| `voice` | Any `list_voices()` result | `"bm_lewis"` | Default TTS voice |
| `speed` | float | `1.25` | Default TTS speed |

You can also set `channel_mode` via env var without a config file:

```bash
export MOD3_CHANNEL_MODE=claude-code   # enable
export MOD3_CHANNEL_MODE=local-agent   # disable
```

## Starting Claude Code with the Mod³ channel

**Development (local):**
```bash
claude --dangerously-load-development-channels server:mod3
```

Claude Code will spawn mod³'s MCP server (`python3 server.py --channel`) as a subprocess per the `.mcp.json` / `mcp.channel.json` config. The `claude/channel` capability is declared at server init; Claude Code registers the notification listener automatically.

**Future (once mod3 is submitted to the Anthropic plugin marketplace):**
```bash
claude --channels plugin:mod3@myrgic-plugins
```

## Applying the config

When the user asks to configure mod³:

1. Read the current config if it exists:
   ```
   ~/.claude/channels/mod3/config.json
   ```

2. Merge the requested changes.

3. Write the updated config back.

4. Confirm what changed. If `channel_mode` was set to `claude-code`, remind the operator to restart Claude Code with:
   ```
   claude --dangerously-load-development-channels server:mod3
   ```

## Checking channel status

The mod³ HTTP API exposes channel health at `/capabilities`:

```bash
curl http://localhost:7860/capabilities | python3 -m json.tool
```

The response includes `channel_mode`, `mcp_session_active`, and `version`.

## Resetting to defaults

Delete the config file to return to standalone mode:

```bash
rm ~/.claude/channels/mod3/config.json
```

Or explicitly set `channel_mode` back to `local-agent`.
