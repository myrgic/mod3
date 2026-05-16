# G2: B+ Executed Decision Record

**Status:** Resolved 2026-05-16; B+ selected by operator.
**Supersedes:** `G2-adoption-decision.md` (proposed-options record; not modified in place).
**References:** `G1-protocol-delta-scoping.md`, `docs/architecture/audio-sidecar.md`
**Authorization:** Operator-confirmed. Implementation work authorized per this record.

---

## 1. Decision

**Option B+ is selected.** This is Option B from G2 (inbound audio framing) plus three
scoped additions resolved by the operator:

| Question | Resolution |
|---|---|
| Q3 — Handshake | Implement: `client-ready` → `bot-ready` exchange on `/ws/audio/{session_id}` accept |
| Q4 — Transcript duplication | Implement: emit RTVI transcript and speaking-lifecycle messages on the audio WS surface alongside existing ACP delivery |
| Q5 — Chunk-by-chunk TTS framing | Defer: the three-frame envelope (`bot-tts-started` / `bot-tts-audio` / `bot-tts-stopped`) is already implemented and sufficient |
| Q6 — Client transport config | Client-side: `pipecat-client-js` and `pipecat-client-react` callers must configure explicit WebSocket transport; no default behavior change on the mod3 side |
| Q7 — Session ID minting | Client-minted UUID: Pipecat clients pass their own UUID as the `/ws/audio/{session_id}` path parameter; mod3 uses it as an opaque key; no ACP pre-handshake required |

**What does NOT change:** ACP remains the control plane. Config stays on HTTP. The audio
sidecar is the scope of this work. RTVI topology is not inherited.

---

## 2. Verified Ground Truth

**Upstream source:** `pipecat-ai/pipecat:main/src/pipecat/processors/frameworks/rtvi/`

### RTVI 1.3.0 canonical message catalog (from `models.py` + `processor.py`)

**Inbound (client → server) — relevant to this work:**

| Wire type string | Handler in `processor.py` | Notes |
|---|---|---|
| `client-ready` | `_handle_client_ready` | Parsed via `ClientReadyData(version, about)` |
| `raw-audio` | `_handle_audio_buffer` | Base64 audio payload; same handler as `raw-audio-batch` |
| `raw-audio-batch` | `_handle_audio_buffer` | Batch variant |
| `disconnect-bot` | pushes `EndTaskFrame` upstream | Graceful close trigger |
| `client-message` | `_handle_client_message` | Custom message relay |
| `ui-event` | UI Agent Protocol handler | Out of scope |
| `send-text` | `_handle_send_text` | Out of scope |

**Naming correction:** G1 and the earlier G2 draft used `client-audio` for the inbound
audio frame type. The actual RTVI 1.3.0 wire type is **`raw-audio`** (and `raw-audio-batch`
for batched delivery). This is confirmed in `processor.py:case "raw-audio" | "raw-audio-batch"`.
All implementation tasks must use `raw-audio`, not `client-audio`.

**Outbound (server → client) — relevant to this work (T4 transcript duplication):**

| Wire type | Class | Purpose |
|---|---|---|
| `bot-ready` | `BotReady` | Handshake reply after `client-ready` |
| `bot-tts-started` | `BotTTSStartedMessage` | Already implemented |
| `bot-tts-audio` | `BotTTSAudioMessage` | Already implemented |
| `bot-tts-stopped` | `BotTTSStoppedMessage` | Already implemented |
| `bot-llm-started` | `BotLLMStartedMessage` | T4: emit on LLM start |
| `bot-llm-stopped` | `BotLLMStoppedMessage` | T4: emit on LLM stop |
| `bot-transcription` | `BotTranscriptionMessage` | T4: duplicate bot STT/response text |
| `user-transcription` | `UserTranscriptionMessage` | T4: duplicate user STT result |
| `user-started-speaking` | `UserStartedSpeakingMessage` | T3 inbound + T4 outbound echo |
| `user-stopped-speaking` | `UserStoppedSpeakingMessage` | T3 inbound + T4 outbound echo |

**`BotReady` data shape** (from `BotReadyData` in `models.py`):
```json
{
  "label": "rtvi-ai",
  "type": "bot-ready",
  "id": "<mirrors client-ready id>",
  "data": {
    "version": "1.3.0",
    "about": { ... }
  }
}
```

**RTVI surface framing:** T2 (handshake) and T4 (transcript duplication) are two facets
of the same RTVI surface addition, not two parallel tracks. They compose: the handshake
gates the surface open; transcript duplication populates it with events. Implement in
dependency order (T2 first, then T4) but treat them as one conceptual unit.

---

## 3. Implementation Task List

Six tasks. Wave 0 tasks are parallel-safe. Wave 1 and Wave 2 depend on earlier waves.

### Wave 0 — Parallel (4 tasks)

**T1 — Schema: add `rtvi-client` to `VALID_CLIENT_TYPES`**
- File: `mod3/seats.py:41`
- Change: `VALID_CLIENT_TYPES = frozenset({"claude-code-channel", "generic", "rtvi-client"})`
- Also update the corresponding test fixture that validates seat creation.
- Effort: trivial (1 line + test update).

**T2 — Handshake: `client-ready` → `bot-ready` on `/ws/audio/{session_id}`**
- File: `mod3/http_api.py` (`ws_audio` handler at line 1900)
- Current state: the handler accepts the WS connection and immediately drains client
  frames as a no-op while loop (lines 1934-1941). Client → server frames are ignored.
- Change: before entering the drain loop, read the first JSON text frame, expect
  `{"type": "client-ready", ...}`, validate the major version component against
  `PROTOCOL_VERSION = "1.3.0"`, then reply with a `bot-ready` JSON text frame.
- On version mismatch: send an RTVI `error` frame and close.
- On non-JSON or missing `type`: tolerate (treat as legacy binary-only client, skip handshake).
- `bot-ready.data.about` should carry `{"server": "mod3", "version": "<mod3 version>"}`.

**T4 — RTVI surface: outbound transcript and speaking-lifecycle emission**
- Files: `mod3/audio_subscribers.py` (new emit methods), `mod3/inbound.py` (transcript
  hook), `mod3/agent_loop.py` (LLM lifecycle hooks)
- Pattern: model on existing `emit_wav` in `audio_subscribers.py:150`. Add:
  - `emit_user_transcription(session_id, text, is_final)` → `user-transcription` frame
  - `emit_bot_transcription(session_id, text, is_final)` → `bot-transcription` frame
  - `emit_user_started_speaking(session_id)` → `user-started-speaking` frame
  - `emit_user_stopped_speaking(session_id)` → `user-stopped-speaking` frame
  - `emit_bot_llm_started(session_id)` → `bot-llm-started` frame
  - `emit_bot_llm_stopped(session_id)` → `bot-llm-stopped` frame
- Wire points: `inbound.py` publishes the STT transcript via `ModalityBus.perceive()`
  — add the RTVI emit call at the same site. `agent_loop.py` manages the LLM call
  lifecycle — add `emit_bot_llm_started` before the LLM call and `emit_bot_llm_stopped`
  after response completion.
- Delivery: best-effort, same thread-safety pattern as `emit_wav` (RLock + `run_coroutine_threadsafe`).
- Both ACP and audio WS surfaces receive transcript events; ACP path is unchanged.

**T5 — `disconnect-bot` handling in `/ws/audio/{session_id}`**
- File: `mod3/http_api.py` (`ws_audio` handler)
- Change: in the drain loop, if the received JSON frame has `"type": "disconnect-bot"`,
  break the loop and fall through to the `finally` unregister path.
- This is a graceful close initiated by the client; the WebSocket close handshake fires
  normally via the `finally` block.

### Wave 1 — After T2 (1 task)

**T3 — Inbound `raw-audio` JSON parsing in `/ws/audio/{session_id}`**
- Files: `mod3/http_api.py` (ws_audio drain loop), `mod3/inbound.py` (STT routing)
- Current state: `AudioCapture` is the only STT source path; it reads from the
  microphone via `sounddevice`. WebSocket-sourced audio has no route to STT.
- Change: in the drain loop, detect JSON text frames with `"type": "raw-audio"`. Decode
  the base64 `data.audio` field to `bytes`, interpret as int16 PCM, and push to the
  existing VAD/STT pipeline. Also handle `user-started-speaking` and
  `user-stopped-speaking` as explicit VAD override signals (bypass Silero VAD; treat as
  utterance boundary markers).
- The WS-source audio path must not conflict with a simultaneously active mic path.
  Guard with a per-session flag or a session-scoped source selector.
- Depends on T2 (handshake) because the drain loop structure established by T2 is the
  integration point for audio frame dispatch. The `ws_audio` handler itself does not
  check `client_type` — T1 (`VALID_CLIENT_TYPES`) is independent of this handler.

### Wave 2 — After T3 (1 task)

**T6 — Integration test with `@pipecat-ai/client-js`**
- Location: `mod3/tests/test_rtvi_integration.py` (new file) + fixture in
  `mod3/tests/fixtures/rtvi_client/`
- Vendor or pin `@pipecat-ai/client-js` in the fixture (npm package, explicit WS
  transport configured).
- Drive a full session: `client-ready` → `bot-ready` → `raw-audio` upload (synthetic
  PCM) → `bot-tts-started/audio/stopped` reception → `disconnect-bot`.
- Assert: handshake succeeds, audio frames arrive in correct RTVI envelope, graceful
  close observed.
- Use `pytest-asyncio` + a test WebSocket server fixture wrapping the real `ws_audio`
  handler.

---

## 4. Compatibility Matrix

| SDK | Transport default | Works after B+ lands? | Notes |
|---|---|---|---|
| `pipecat-client-js` (web) | WebRTC (default) or WebSocket (explicit config) | Yes, with explicit WS transport | Caller must pass `transport: new WebSocketTransport(url)` |
| `pipecat-client-react` | WebRTC (default) or WebSocket (explicit config) | Yes, with explicit WS transport | Same as above |
| `pipecat-client-ios` | WebRTC | No | Out of scope; WebRTC not implemented |
| `pipecat-client-android` | WebRTC | No | Out of scope |
| `pipecat-client-cpp` | WebRTC | No | Out of scope |
| `pipecat-client-react-native` | WebRTC | No | Out of scope |

mod3's existing dashboard client and channel-pattern clients (ACP + binary audio) continue
to work without modification. The handshake in T2 is tolerant of legacy binary-only
clients that skip the `client-ready` frame.

---

## 5. Out of Scope (Explicit)

The following RTVI features are present in upstream Pipecat but are explicitly excluded
from B+:

- **Chunk-by-chunk TTS framing** (Q5): the current three-frame-per-utterance envelope
  is sufficient. Sub-utterance streaming is a latency optimization deferred to a future
  iteration.
- **LLM function-call lifecycle** (`llm-function-call-started/in-progress/stopped`): the
  deprecated `llm-function-call` and its replacements carry pipecat pipeline internals
  not relevant to mod3's architecture.
- **UI Agent Protocol** (`ui-event`, `ui-command`, `ui-snapshot`, `ui-cancel-task`):
  this is an entirely separate vertical within RTVI. mod3 has no UI automation surface.
- **Audio-level events** (`user-audio-level`, `bot-audio-level`): optional telemetry;
  not required for basic interoperability.
- **`metrics` and `system-log`** messages: pipecat pipeline internals.
- **WebRTC transport**: all four non-web SDK targets default to WebRTC. mod3 does not
  implement WebRTC. Out of scope for this batch and for the foreseeable future.
- **ACP topology changes**: transcripts, config, and action protocol remain on ACP.
  RTVI does not expand into the control plane.

---

## 6. Foundation Operational Mode Test

Before authorizing implementation: is B+ driven by a concrete use case or by protocol
completeness for its own sake?

The forcing function is concrete: `pipecat-client-js` and `pipecat-client-react` are the
target client surfaces for a Pipecat pipeline integration. Those clients emit `client-ready`
on connect and send audio as `raw-audio` frames. Without T2 + T3, they cannot connect to
mod3's audio sidecar. Without T4, they receive no transcript feedback on the RTVI surface
— the session is a one-way audio stream with no confirmation of speech recognition.

B+ is the minimal set that makes those clients functional. Everything in the "out of scope"
list above was tested against this same question and failed it.

---

## 7. Related Files

- `docs/rtvi/G1-protocol-delta-scoping.md` — full delta analysis and trade-offs
- `docs/rtvi/G2-adoption-decision.md` — proposed-options record (do not modify)
- `docs/architecture/audio-sidecar.md` — canonical control/audio plane split
- `mod3/audio_subscribers.py` — existing `emit_wav` pattern (T4 reference implementation)
- `mod3/http_api.py` — `ws_audio` handler at line 1900 (T2, T3, T5 target)
- `mod3/inbound.py` — STT transcript publish point (T3, T4 wire point)
- `mod3/agent_loop.py` — LLM lifecycle (T4 wire point)
- `mod3/seats.py:41` — `VALID_CLIENT_TYPES` (T1 target)
