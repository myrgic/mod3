# G2: RTVI Adoption Decision

**Status:** Decision document. Operator decision required.
**Date:** 2026-05-16
**References:** G1 scoping report (`docs/rtvi/G1-protocol-delta-scoping.md`)
**Authorization:** Operator decision — do not auto-merge.

---

## Decision surface

Three options were scoped in G1. This document presents them for operator
decision and captures the rationale once decided.

---

## Option A — Status quo (current TTS subset, no expansion)

**What it means:** Keep the current RTVI 1.3.0 TTS output subset
(`bot-tts-started` / `bot-tts-audio` / `bot-tts-stopped`). Inbound audio
remains raw binary WebSocket frames — no RTVI framing on the client → server
path.

**When this is correct:**
- mod3's client base is the dashboard and channel-pattern clients (which mod3
  controls).
- No requirement to interoperate with pipecat-native client libraries that
  expect `client-audio` framing.
- Minimal maintenance surface is the priority.

**Trade-offs:**
- A pipecat pipeline client attempting to send audio would need an adapter.
- The architectural boundary (ACP control / RTVI audio) remains clean.
- No change to any existing code.

---

## Option B — Inbound framing only (narrow RTVI conformance)

**What it means:** Add `client-audio` inbound message parsing to the audio
sidecar WebSocket (`/ws/audio/{session_id}`). The server accepts RTVI-shaped
`{"type":"client-audio","data":{"audio":"<base64>"}}` frames in addition to
(or replacing) raw binary frames.

All other boundaries unchanged: transcripts via ACP, config via HTTP.

**When this is correct:**
- mod3 should accept pipecat-native client connections for inbound audio.
- The dashboard client can be updated to send framed audio without a large
  architectural change.
- This is the natural next step if ecosystem interop is a goal.

**Trade-offs:**
- Bounded engineering effort (primarily `http_api.py` WebSocket handler).
- Dashboard client must be updated to send RTVI-framed audio (rather than
  raw binary).
- Does not commit to full topology change.
- Reversible — removing inbound framing is low-cost if requirements change.

**Implementation shape (if selected):**
1. Update `/ws/audio/{session_id}` handler to parse inbound WebSocket messages
   as JSON text frames with RTVI envelope.
2. Accept both raw binary (legacy) and JSON text (RTVI) for a transition period.
3. Update dashboard client to send RTVI-shaped inbound audio.
4. Add tests for RTVI inbound parsing.

---

## Option C — Full RTVI topology inheritance

**What it means:** Migrate transcripts, config, and action protocol to the RTVI
channel. Full pipecat client library compatibility.

**When this is correct:**
- mod3 is targeting pipecat-pipeline-as-a-service use cases where third-party
  clients drive the full conversation lifecycle via RTVI.
- The ACP/RTVI boundary split is acceptable to collapse.

**Trade-offs:**
- Significant engineering effort.
- Blurs the ACP/RTVI boundary — the audio-sidecar architecture doc explicitly
  designed against this.
- Not recommended unless the use-case requirement is clear.

---

## Foundation operational mode test (per CLAUDE.md)

Before committing to any option: is the decision driven by a concrete use case
or by completeness-for-its-own-sake?

- If no current client is blocked by the inbound framing gap → Option A.
- If a specific client (pipecat pipeline, external integration) needs inbound
  framing → Option B.
- If a full-RTVI client is the stated target → Option C.

The default answer is **Option A** unless a forcing function is present.

---

## Operator decision

**[ ] Option A — Status quo.** No RTVI changes in this cycle.

**[ ] Option B — Inbound framing only.** Add `client-audio` parsing to
`/ws/audio/{session_id}`. Wave 2 scope addition.

**[ ] Option C — Full topology.** Out of scope for this batch; requires
dedicated planning.

---

## Topology inheritance declaration

**RTVI topology is NOT inherited in this batch unless Option B or C is
selected.** The G1+G2 structure produces a scoping doc and a decision doc. No
RTVI implementation work occurs in Wave 0/1/2 unless the operator selects
Option B or C in this document.

F5 (Wave 2) wires Smart Turn into the turn-taking pipeline — this is internal
pipeline work independent of RTVI adoption. F5 does not require a RTVI
topology decision.

---

## Related files

- `docs/rtvi/G1-protocol-delta-scoping.md` — full delta analysis and trade-off
  details
- `docs/architecture/audio-sidecar.md` — architectural intent and topology
- `audio_subscribers.py` — current RTVI TTS output implementation
