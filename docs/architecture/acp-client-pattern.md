# ACP-Client Pattern: mod3 as Claude Code Driver

## What this is

The ACP-client pattern makes mod3 the **front-door** for Claude Code rather
than a peripheral surface attached to an already-running session. mod3 lists
projects and sessions, spawns Claude Code subprocesses on demand, and the
spawned process registers back as a seat on the mod3 dashboard.

This is distinct from the **channel pattern** (PR #40), where Claude Code
starts first and mod3 is attached as a channel client.

| Pattern | Driver | Backend | Entry point |
|---------|--------|---------|-------------|
| Channel | Claude Code session | mod3 = voice/text surface | `claude --channels server:mod3` |
| ACP-client | mod3 dashboard | Claude Code = inference backend | `http://localhost:7860/dashboard/sessions.html` |

Both patterns converge at the same seat-registration mechanism
(`POST /v1/sessions/{id}/seats`). The difference is who initiates.

---

## Kernel endpoints (CogOS)

The kernel (port 6931) exposes three endpoints that the ACP-client pattern depends on:

```
GET  /v1/claude-code/projects
GET  /v1/claude-code/projects/{project}/sessions
POST /v1/claude-code/spawn
```

See the cogos `internal/engine/serve_claude_code.go` handler for implementation
details and the `CLAUDE_PROJECTS_DIR` env-var override.

---

## mod3 surface

### Dashboard page

`http://localhost:7860/dashboard/sessions.html` — served via
`GET /dashboard/{filename}` in `http_api.py`. Shows:

- Project grid: all project directories under `~/.claude/projects/`, sorted by
  last activity. Click to drill into sessions.
- Session list: per-project `.jsonl` files with `first_prompt_summary`,
  turn count, message count, and `last_modified` timestamp.
- **Resume** button: calls `POST /v1/claude-code/spawn` with the selected
  `session_id`.
- **Mount New Session** / **New Session in Project** buttons: call
  `POST /v1/claude-code/spawn` without a `session_id`.

### Spawn proxy

`POST /v1/claude-code/spawn` in mod3 proxies to the kernel endpoint.
This avoids browser CORS issues (dashboard at port 7860 calling kernel at 6931).
The `COGOS_KERNEL_URL` env-var overrides the default `http://localhost:6931`.

---

## Spawn lifecycle

```
1. Operator clicks "Resume" on sessions.html
2. Browser POST /v1/claude-code/spawn → mod3 (port 7860)
3. mod3 proxies → kernel POST /v1/claude-code/spawn (port 6931)
4. Kernel writes temp .mcp.json (points channel_client.py at mod3)
5. Kernel calls ClaudeCodeProvider.SpawnBackground(...)
6. Kernel returns { process_id, status: "spawned", spawned_at }
7. mod3 returns the kernel response verbatim (201)
8. Dashboard banner: "Session resumed (process <id>)"
9. Claude Code subprocess starts, loads .mcp.json
10. channel_client.py connects: POST /v1/sessions/{id}/seats
11. Seat appears in Dashboard tab participants panel
```

Steps 9-11 happen asynchronously. The spawn response confirms the subprocess
was started; observe seat registration on the Dashboard tab to confirm it is live.

---

## Testing

```bash
# Unit + integration tests covering the spawn proxy and dashboard static delivery
cd mod3
PYTHONPATH=. pytest tests/test_acp_client_flow.py -v

# Curl smoke tests (kernel must be running on port 6931)
curl http://localhost:6931/v1/claude-code/projects
curl http://localhost:6931/v1/claude-code/projects/-Users-slowbro/sessions

# Spawn a new session (kernel must be running)
curl -X POST http://localhost:7860/v1/claude-code/spawn \
  -H 'Content-Type: application/json' \
  -d '{"project": "-Users-slowbro"}'

# Resume a specific session
curl -X POST http://localhost:7860/v1/claude-code/spawn \
  -H 'Content-Type: application/json' \
  -d '{"project": "-Users-slowbro", "session_id": "<session_uuid>"}'
```

---

## Operator Resume flow (manual verification)

After the three PRs are merged and both services are running:

1. Open `http://localhost:7860/dashboard/sessions.html`
2. Confirm project list loads (requires kernel on port 6931)
3. Click a project — confirm session list with preview cards
4. Click **Resume** on any session
5. Banner shows "Session resumed (process ...)"
6. Switch to `http://localhost:7860/dashboard`
7. Watch the Participants pill — count should increment as the channel client registers its seat

**Do not click Resume from an automated context** — it starts a real Claude
Code subprocess with real token spend.

---

## Env-var reference

| Var | Default | Effect |
|-----|---------|--------|
| `COGOS_KERNEL_URL` | `http://localhost:6931` | mod3 spawn proxy target |
| `CLAUDE_PROJECTS_DIR` | `~/.claude/projects` | kernel project listing root |
