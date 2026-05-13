# Audio Sidecar Architecture

## Overview

This document describes how mod3's real-time audio plane composes with the
ACP control plane. The pattern is called the **audio sidecar**: a parallel
WebSocket channel that carries time-sensitive audio data alongside, but
separate from, the ACP session that carries agent messages and control events.

## 1. Motivation

The Agent Client Protocol (ACP) models content as typed `ContentBlock` records.
Its current `ContentBlock::Audio` type carries audio as a complete blob: encode
the audio, put it in the block, send it. That is the right model for
asynchronous, document-shaped work. It is the wrong model for two real-time
audio use cases:

**Low-latency TTS playback.** When mod3's TTS engine finishes a sub-sentence
chunk (typically 0.5 to 2 seconds of audio), the user should hear it within
tens of milliseconds. Batching a complete utterance into an ACP `Audio` block
and sending it once synthesis finishes adds 1 to 4 seconds of end-to-end
latency. That is unacceptable for conversational voice interfaces.

**Continuous microphone input.** VAD and STT operate on a growing audio buffer
that is updated every 20 to 100 milliseconds. Sending each increment as an ACP
content block would flood the control channel with hundreds of small messages
per second. The control plane is structured for intent, not for raw sensor data.

A parallel audio channel decouples the two concerns. ACP carries structured
messages: agent responses, tool calls, session state. The audio sidecar carries
raw or lightly-framed PCM: TTS chunks going down to the client, mic frames
going up from the client.

## 2. Topology

```
  Client (dashboard / voice IDE / Pipecat pipeline)
    |                             |
    | ACP control plane           | RTVI-shaped audio sidecar
    | (stdio or Streamable HTTP)  | (WebSocket)
    |                             |
    v                             v
  mod3 ACP session handler    mod3 /ws/audio/{session_id}
    |                             |
    +-------- session_id ----------+
    |
    +-- agent kernel (CogOS)
    +-- TTS engine (kokoro/voxtral/chatterbox/spark)
    +-- STT (WhisperDecoder)
    +-- VAD (Silero)
```

The two connections share a single `session_id`. The client opens an ACP
session (which mints or reuses a `session_id`) and then opens a WebSocket to
`/ws/audio/{session_id}`. Both connections are identified by the same opaque
string. The audio channel has no independent session lifecycle: it follows the
ACP session.

## 3. Wire Envelope for the Audio Sidecar

### Server to client (TTS audio)

The server sends TTS output as RTVI 1.3.0 `bot-tts-audio` messages extended
with mod3-specific metadata fields.

RTVI base shape:

```json
{
  "type": "bot-tts-audio",
  "data": {
    "audio": "<base64-encoded int16 PCM>",
    "sample_rate": 24000,
    "num_channels": 1
  }
}
```

Mod3 extension fields carried inside `data`:

| Field | Type | Description |
|---|---|---|
| `engine` | string | TTS engine that produced this chunk (e.g. `"kokoro"`) |
| `voice` | string | Voice identifier (e.g. `"bm_lewis"`) |
| `chunk_index` | int | 0-based chunk counter within the utterance |
| `sentence_index` | int | 0-based sentence counter within the utterance |
| `is_final` | bool | True on the last chunk of the utterance |
| `rtf` | float | Real-time factor for this chunk (gen_time / audio_duration) |
| `gen_time_sec` | float | Wall-clock seconds spent generating this chunk |

This maps directly to `TTSChunkEvent` in `schemas/operations.py:158` (committed
at `5be113c`). The wire encoding is: base64 the int16 PCM bytes into the
`audio` field; copy all other `TTSChunkEvent` fields into `data`. No
re-encoding or shape transformation is needed between the D2 worker wire and
the WebSocket wire.

The current `/ws/audio/{session_id}` implementation in `http_api.py:1269`
sends a two-frame sequence: a JSON header frame followed by a binary WAV blob.
Section 5 (Migration Plan) describes how to evolve that to the RTVI-shaped
envelope.

### Client to server (microphone input)

The client sends raw audio data and VAD signals using standard RTVI client
messages:

| Message type | Payload | Description |
|---|---|---|
| `user-started-speaking` | `{}` | Client VAD detected voice onset |
| `user-stopped-speaking` | `{}` | Client VAD detected end of utterance |
| `user-audio-level` | `{"level": 0.0-1.0}` | RMS energy level (for UI meters) |
| binary frame | raw int16 PCM at 16kHz | Raw microphone audio for server-side VAD/STT |

Mic audio is 16 kHz, mono, int16 PCM. Binary frames are sent without a JSON
wrapper. The server dispatches them to the VAD and STT pipeline.

## 4. Session Coupling

A single conceptual session has one ACP session ID and one parallel WebSocket
audio connection. The lifecycle is:

1. **Client opens ACP session.** The ACP session handler creates or resumes a
   `session_id` (e.g. `"mod3-8a3f"`). For new sessions this also registers the
   session with `session_registry.py`.

2. **Client opens audio WebSocket.** The client connects to
   `/ws/audio/{session_id}`. The `ws_audio` handler (`http_api.py:1269`) calls
   `AudioSubscriberRegistry.register()` (`audio_subscribers.py:64`) to attach
   the WebSocket to the session bucket.

3. **TTS output is routed to the session.** When the agent produces a TTS
   response, the server looks up the session's audio subscribers via
   `AudioSubscriberRegistry.has_subscribers()` and emits the audio frames over
   the WebSocket. If no subscriber is attached, it falls back to local playback
   (`afplay` or `sounddevice`).

4. **Microphone frames flow in.** Binary frames arriving on the WebSocket are
   dispatched to the VAD pipeline. VAD output gates the STT decoder. The STT
   transcript is forwarded to the agent as a `CognitiveEvent` (user speech).

5. **Cleanup.** On WebSocket disconnect, `ws_audio` calls
   `AudioSubscriberRegistry.unregister()` (`audio_subscribers.py`). The session
   remains registered in the session registry; only the audio subscriber is
   removed. If the ACP session is also closed, the session registry deregisters
   it via `/v1/sessions/{session_id}/deregister` (`http_api.py:946`).

This mirrors the existing Wave 4.3 pattern in `audio_subscribers.py`. The
subscriber registry is already the coupling point; the ACP session ID just
becomes the canonical `session_id` that both connections use.

## 5. Migration Plan

Mod3 currently has two WebSocket endpoints:

- `/ws/chat` (`http_api.py:1313`): a hybrid control-and-audio channel. The
  dashboard sends voice input and receives both agent responses and audio
  output over one WebSocket. It also hosts the `AgentLoop`, `BrowserChannel`,
  and `PipelineState` directly in the handler.

- `/ws/audio/{session_id}` (`http_api.py:1269`): the Wave 4.3 audio fan-out
  channel. The server emits WAV blobs; the client plays them. Client-to-server
  frames are currently ignored.

The migration proceeds in three steps:

**Step 1: Evolve `/ws/audio/{session_id}` to RTVI envelope.**
Change the server-side emit path in `audio_subscribers.py` from the current
two-frame (JSON header + binary WAV) format to the `bot-tts-audio` RTVI shape
described in Section 3. This is a wire-format change only; the subscriber
registry, session coupling, and fan-out logic are unchanged. The dashboard
WebSocket client must be updated to match.

**Step 2: Add microphone input handling to `/ws/audio/{session_id}`.**
Currently `ws_audio` reads and discards client frames (`http_api.py:1303`).
Add a dispatch path: binary frames go to the VAD/STT pipeline; JSON frames
matching `user-started-speaking` / `user-stopped-speaking` update the inbound
VAD state.

**Step 3: Deprecate `/ws/chat`.**
Once `/ws/audio/{session_id}` carries both directions and the ACP session
handler covers the control plane, `/ws/chat` is redundant. It should be marked
deprecated (a `DeprecationWarning` and a `Deprecation` header on connection
accept) and removed in mod3 v0.5. The two successor paths are:

- ACP session transport for all control-plane work (agent messages, tool calls,
  session lifecycle).
- Pipecat client (`Mod3TTSService` in `integrations/pipecat/tts_service.py`)
  for inference testing and pipeline integration.

The deprecation follows the pattern already established for `/v1/mcp` in
`mcp_shim.py` (commit `f78fea4`): a `DeprecationWarning` on connect, a
migration guide in the docstring, and a hard removal date tied to a version.

## 6. Composition with the Schemas at 5be113c

`TTSChunkEvent` in `schemas/operations.py:158` is the natural payload shape for
server-to-client audio frames. Every field maps cleanly to the RTVI
`bot-tts-audio` extension described in Section 3:

- `audio_b64`: the base64-encoded int16 PCM audio, placed in `data.audio`.
- `sample_rate`, `num_channels`, `dtype`: wire geometry, placed in `data`.
- `chunk_index`, `sentence_index`, `is_final`: sequencing fields, carried as
  mod3 extension fields.
- `gen_time_sec`, `rtf`, `peak_memory_gb`, `tokens`: performance metrics,
  carried as mod3 extension fields (optional; clients may ignore).
- `engine`, `voice`: provenance fields useful for UI display.

No schema change is needed. The D2 worker subprocess emits `TTSChunkEvent`
records on the D2 wire (kernel-to-worker); the HTTP layer re-frames them as
RTVI `bot-tts-audio` messages for WebSocket clients. The data is the same; only
the outer envelope changes.

## 7. Future Work: ACP Streaming Audio Extension

The audio sidecar is an interim pattern that works today. The correct long-term
solution is a streaming audio content type in ACP itself: a
`ContentBlock::AudioStream` that carries audio chunks inline on the ACP session
transport, eliminating the separate WebSocket.

The rough shape of such an extension:

```json
{
  "type": "message",
  "content": [
    {
      "type": "audio_stream",
      "chunk_index": 0,
      "is_final": false,
      "audio": "<base64 int16 PCM>",
      "sample_rate": 24000,
      "num_channels": 1
    }
  ]
}
```

Contributing this upstream to ACP requires a working implementation (the sidecar
gives us one), benchmark data on latency vs. blob approach, and consensus from
ACP maintainers on the streaming model. The detailed design is RFC-0001's scope
(`docs/rfcs/0001-mod3-as-cog-native-modality-node.md`). This document notes the
dependency; the sidecar is the bridge until that upstream work lands.
