# G1: RTVI Protocol Delta Scoping Report

**Status:** Scoping document (Wave 0 output). G2 decision document references this.
**Date:** 2026-05-16
**Scope:** Identify what mod3 already implements from the RTVI wire protocol,
what the delta is vs. full RTVI conformance, and what the cost/risk of
closing that delta would be — WITHOUT committing to full implementation.

---

## 1. What mod3 already implements (RTVI 1.3.0 subset)

mod3's current implementation touches RTVI at one point: the TTS audio
sidecar via `/ws/audio/{session_id}`.

### 1.1 Implemented message types

| Message type | Direction | Status |
|---|---|---|
| `bot-tts-started` | server → client | Implemented (audio_subscribers.py) |
| `bot-tts-audio` | server → client | Implemented with mod3 extension fields |
| `bot-tts-stopped` | server → client | Implemented (audio_subscribers.py) |

Wire shape (current):
```json
{"label":"rtvi-ai","type":"bot-tts-audio","id":"<uuid>","data":{
  "audio":"<base64 int16 PCM>",
  "sample_rate":24000,
  "num_channels":1
}}
```

Extension fields in `data` (mod3-specific, documented in audio-sidecar.md):
`engine`, `voice`, `chunk_index`, `sentence_index`, `is_final`, `rtf`, `gen_time_sec`.

### 1.2 What is NOT implemented

The following RTVI features exist in the pipecat ecosystem but have no
counterpart in mod3:

| Feature | RTVI Message/Concept | mod3 gap |
|---|---|---|
| Client audio input framing | `client-audio` (raw PCM upstream) | mod3 receives raw WebSocket binary frames; no RTVI envelope on inbound |
| VAD signals | `client-vad-activity` | mod3 does VAD server-side; no client-VAD message type |
| Bot STT transcript | `bot-transcription` | mod3 surfaces transcript via ACP session, not RTVI |
| User STT transcript | `user-transcription` | Same — ACP, not RTVI |
| Bot LLM response | `bot-llm-stopped`, `bot-llm-text` | Not in mod3 (LLM response is ACP, not RTVI) |
| Session lifecycle | `connected`, `disconnected`, `error` | mod3 uses HTTP 200/WebSocket close, not RTVI lifecycle messages |
| Config messages | `config`, `config-update` | Not implemented |
| Action protocol | `action`, `action-response` | Not implemented |
| RTVI metrics | `metrics` | Not implemented |
| Pipeline control | `update-config`, `get-config` | Not implemented |

---

## 2. What RTVI conformance would require

Full RTVI conformance means implementing the entire RTVI message taxonomy on
the audio sidecar WebSocket — both inbound (client → server) and outbound
(server → client).

### 2.1 Inbound (client → server) — currently unframed

mod3 currently receives raw audio bytes from the client WebSocket with no
RTVI envelope. Full conformance would require:
- Wrapping inbound audio in `client-audio` frames.
- Parsing RTVI-shaped inbound messages on the WebSocket.
- Adding RTVI session lifecycle messages (`connected`, `disconnected`).

**Effort estimate:** Medium. Requires changes to WebSocket handling in
`http_api.py` and potentially `audio_subscribers.py`. Client-side (dashboard,
channels) must also send RTVI-shaped frames.

### 2.2 Transcript messages

mod3 produces STT transcripts and delivers them via ACP (`ContentBlock`). RTVI
delivers them as `user-transcription` / `bot-transcription` on the same
WebSocket. Full conformance would mean duplicating delivery (once to ACP, once
to RTVI) OR migrating transcript delivery from ACP to RTVI.

**Effort estimate:** Medium-High. ACP is the load-bearing control plane; the
mod3 architecture doc explicitly separates control (ACP) from audio
(RTVI-shaped sidecar). Migrating transcript delivery to RTVI would blur that
separation. Duplication is simpler but adds maintenance surface.

### 2.3 Config and action protocol

RTVI's config and action messages allow clients to update pipeline
configuration at runtime. mod3 has a settings API via HTTP (`/v1/settings/*`).
Exposing that via RTVI would require a new translation layer.

**Effort estimate:** High. mod3's settings surface is richer than RTVI's
config model; mapping would require semantic decisions about what maps to what.

---

## 3. Key observations

### 3.1 The structural fit

mod3's architecture is **already shaped by RTVI** at the audio sidecar
boundary. The three-frame envelope (`started` / `audio` / `stopped`) is RTVI
1.3.0. The `label:"rtvi-ai"` field is present. The mod3 extension fields in
`data` are additive and compatible with RTVI clients that ignore unknown fields.

The inbound side (client → server audio) is the main gap. mod3 currently
receives raw binary WebSocket frames — this is simpler for the dashboard
client but is not RTVI-conformant.

### 3.2 The architectural boundary

The audio-sidecar architecture doc explicitly positions ACP as the control
plane and RTVI as the audio plane. This is a deliberate split. Full RTVI
conformance (adding transcript, config, action messages to the RTVI channel)
would expand the RTVI boundary into ACP territory. That is a topology decision,
not just a protocol conformance decision.

### 3.3 The pipecat ecosystem value

The primary value of RTVI conformance is **pipecat client-library
compatibility**: a Pipecat-native voice client could connect to mod3's audio
sidecar without a custom adapter. The current three-frame TTS output is
already compatible. The gap is inbound audio framing — a Pipecat client that
sends `client-audio` frames would need mod3 to parse them.

---

## 4. Decision surface for G2

G2 must decide among three postures:

**Option A — Status quo (current subset, no expansion)**
Keep the current RTVI 1.3.0 TTS output subset. Do not add inbound framing.
Accept that inbound audio is unframed (raw binary).

Trade-off: no pipecat client compatibility for inbound audio. No maintenance
burden. The current subset is sufficient for mod3's dashboard client and
channel pattern.

**Option B — Inbound framing only (narrow RTVI conformance)**
Add `client-audio` inbound message parsing to the audio sidecar WebSocket.
Keep all other boundaries as-is (transcripts via ACP, config via HTTP).

Trade-off: enables pipecat-native clients to send audio. Small, bounded change.
Dashboard client must be updated to send framed audio. Does not commit to
full topology change.

**Option C — Full RTVI topology inheritance**
Migrate transcripts, config, and action protocol to the RTVI channel. Full
pipecat client library compatibility.

Trade-off: significant engineering effort. Blurs the ACP/RTVI boundary. Not
recommended unless mod3 is targeting pipecat-pipeline-as-a-service use cases.

---

## 5. Recommendation (scoping-only, not a decision)

Option B is the natural next step IF the goal is pipecat ecosystem
interoperability. It is bounded, reversible, and does not require topology
changes.

Option A is correct if mod3's client base remains the dashboard and
channel-pattern clients (which mod3 controls and can keep unframed).

The G2 decision document should record which option the operator selects and
why, and whether RTVI topology inheritance is on the roadmap at all.

---

## 6. Files surveyed

- `mod3/audio_subscribers.py` — RTVI TTS output implementation (current)
- `mod3/docs/architecture/audio-sidecar.md` — architectural intent and topology
- `mod3/tests/test_audio_subscribers.py` — RTVI shape compliance tests
- `pipecat-ai/pipecat:src/pipecat/audio/vad/` — RTVI upstream reference (v1.2.1)
- RTVI protocol taxonomy inferred from pipecat source + audio_subscribers.py wire shapes
