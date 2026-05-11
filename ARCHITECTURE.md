# Mod³ Dashboard — Process Architecture

## Intended Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER                               │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐              │
│  │ Silero   │    │ Text     │    │ Audio     │              │
│  │ VAD v5   │    │ Input    │    │ Playback  │              │
│  │ (ONNX)   │    │          │    │ (Web Audio│              │
│  └────┬─────┘    └────┬─────┘    └─────▲─────┘              │
│       │               │               │                     │
│       │ onSpeechEnd   │ sendControl    │ enqueueWav          │
│       │ (Int16 PCM)   │ (JSON)         │ (base64 WAV)       │
│       ▼               ▼               │                     │
│  ┌────────────────────────────────────┐│                     │
│  │      VoiceTransport (WebSocket)    ││                     │
│  │  binary frames ──►  ──► JSON      ││                     │
│  │  JSON frames   ──►  ◄── JSON/b64  ││                     │
│  └────────────────┬───────────────────┘│                     │
│                   │                    │                     │
└───────────────────┼────────────────────┼─────────────────────┘
                    │ WebSocket /ws/chat │
                    ▼                    │
┌───────────────────┼────────────────────┼─────────────────────┐
│                   │  MOD³ SERVER       │                     │
│                   │  (single process)  │                     │
│                   ▼                    │                     │
│  ┌─────────────────────────────────────────┐                │
│  │         BrowserChannel                   │                │
│  │                                          │                │
│  │  _handle_audio(pcm) → buffer            │                │
│  │  _handle_json(msg)  → dispatch           │                │
│  │  _deliver_async()   → send to browser    │                │
│  └──────┬─────────┬──────────▲──────────────┘                │
│         │         │          │                               │
│    PCM audio   text msg   encoded output                    │
│         │         │          │                               │
│         ▼         │          │                               │
│  ┌──────────┐     │          │                               │
│  │ STT      │     │          │                               │
│  │ (mlx_    │     │          │                               │
│  │ whisper) │     │          │                               │
│  │ temp WAV │     │          │                               │
│  └────┬─────┘     │          │                               │
│       │           │          │                               │
│       │ transcript│          │                               │
│       ▼           ▼          │                               │
│  ┌─────────────────────┐     │                               │
│  │  CognitiveEvent     │     │                               │
│  │  {content: "text"}  │     │                               │
│  └──────────┬──────────┘     │                               │
│             │                │                               │
│             ▼                │                               │
│  ┌──────────────────────┐    │                               │
│  │     AgentLoop         │    │                               │
│  │                       │    │                               │
│  │  conversation[]       │    │                               │
│  │  provider.chat()      │    │                               │
│  │  → tool_calls         │    │                               │
│  │                       │    │                               │
│  │  DISPATCH:            │    │                               │
│  │  speak(text)          │    │                               │
│  │    → send_response_text ──────► channel (text to chat)    │
│  │    → bus.act(VOICE)   │    │                               │
│  │       ▼               │    │                               │
│  │  send_text(text)      │    │                               │
│  │    → send_response_text ──────► channel (text to chat)    │
│  │                       │    │                               │
│  │  think(reasoning)     │    │                               │
│  │    → (internal only)  │    │                               │
│  └──────────┬────────────┘    │                               │
│             │                │                               │
│     bus.act(VOICE intent)    │                               │
│             │                │                               │
│             ▼                │                               │
│  ┌──────────────────────┐    │                               │
│  │   ModalityBus        │    │                               │
│  │                      │    │                               │
│  │   OutputQueue        │    │                               │
│  │   (per-channel FIFO) │    │                               │
│  │         │            │    │                               │
│  │         ▼            │    │                               │
│  │   VoiceEncoder       │    │                               │
│  │   (Kokoro TTS)       │    │                               │
│  │   → WAV bytes        │    │                               │
│  │         │            │    │                               │
│  │   ch.deliver(output) ─────┘                               │
│  │   (base64 JSON)      │                                    │
│  └──────────────────────┘                                    │
│                                                              │
│  ┌──────────────────────┐                                    │
│  │  InferenceProvider   │                                    │
│  │  (mlx-lm / Ollama)  │                                    │
│  │                      │                                    │
│  │  model resident in   │                                    │
│  │  memory (in-process) │                                    │
│  └──────────────────────┘                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Current Problems

### 1. ~~Agent blocks on TTS delivery~~ — RESOLVED

```
agent_loop._process():
  await send_response_text(text)    # ← fast, JSON to browser
  await asyncio.to_thread(bus.act)  # ← BLOCKS until TTS generates + delivers
  # agent can't process next event until TTS finishes
```

**Status (2026-05-11):** Resolved. `agent_loop.py:259` and `:283` call `self.bus.act(intent, channel=self.channel_id)` synchronously without `await asyncio.to_thread`, relying on `bus.act`'s default `blocking=False` which returns `QueuedJob` immediately. The PR-#22 review explicitly verified this; the snippet above describes a problem state that no longer matches the code.

### 2. ~~Kokoro cold start blocks OutputQueue drain thread~~ — RESOLVED

```
OutputQueue drain thread:
  _do_encode() → VoiceEncoder.encode() → engine.synthesize()
    → Kokoro first-time init: ~60s blocking
    → All other queued jobs wait
    → _deliver_sync timeout (10s) fires on older jobs
```

**Status (2026-05-11):** Resolved by PR #22. `server.py::_prewarm_tts_if_enabled()` fires a daemon thread on startup that calls `engine.synthesize("warmup", voice="bm_lewis", speed=1.25)` once, paying the Kokoro cold-start cost up front. Env-gated by `MOD3_PREWARM_TTS=1` (default on). Known caveat: if the user's configured default is a non-Kokoro engine (e.g., Voxtral), Kokoro still cold-starts on the first real call — follow-up is to read the configured default at pre-warm time.

### 3. WebSocket lifecycle fragility

```
Browser page reload → new WebSocket → new BrowserChannel
  Old channel's deliver callback still referenced by bus
  Old OutputQueue drain thread still running
  → sends to dead WebSocket → timeout → error cascade
```

**Should be:** channel cleanup on disconnect should cancel all queued jobs
for that channel.

### 4. STT blocks the event loop context

```
_process_utterance():
  await asyncio.to_thread(_transcribe)  # blocks a thread pool thread
    → mlx_whisper.transcribe()          # 1-2s CPU-bound
    → blocks one thread pool slot
```

This is fine for one user. But the thread pool is shared with bus.act().

### 5. No separation between thinking and acting

The agent loop processes ONE event at a time (_processing flag).
If bus.act() blocks, no new events can be processed.
The agent should be able to think about the next input while 
TTS is generating for the current one.

## Intended Architecture (what we should build toward)

```
Browser ──WebSocket──► BrowserChannel
                           │
                      ┌────▼────┐
                      │  INPUT  │  (fast, non-blocking)
                      │  QUEUE  │  CognitiveEvents
                      └────┬────┘
                           │
                      ┌────▼────┐
                      │  AGENT  │  (owns conversation, calls LLM)
                      │  LOOP   │  processes events sequentially
                      │         │  but NEVER blocks on output
                      └────┬────┘
                           │
                    tool calls (non-blocking)
                           │
              ┌────────────┼────────────┐
              │            │            │
         speak(text)  send_text()  think()
              │            │            │
              ▼            ▼            │
        ┌──────────┐ ┌──────────┐     (log)
        │ OUTPUT   │ │ channel  │
        │ QUEUE    │ │ .deliver │
        │ (async)  │ │ (JSON)   │
        └────┬─────┘ └──────────┘
             │
        ┌────▼─────┐
        │ TTS      │  (background thread)
        │ Kokoro   │  
        └────┬─────┘
             │
        ch.deliver(base64 WAV)
             │
             ▼
         Browser playback
```

Key principle: **the agent never waits for output delivery.**
speak() queues a TTS job and returns immediately.
The bus handles encoding and delivery asynchronously.

## Files

| File | Role | Lines | Status |
|------|------|-------|--------|
| `providers.py` | InferenceProvider: MLX, Ollama, CogOS | ~450 | Working |
| `channels.py` | BrowserChannel: WebSocket ↔ bus | ~260 | Working (fragile) |
| `agent_loop.py` | Event → LLM → tool dispatch | ~160 | Working (blocks on TTS) |
| `dashboard/index.html` | UI: chat, VAD, settings | ~700 | Working |
| `dashboard/transport.js` | WebSocket framing | ~100 | Working |
| `dashboard/playback.js` | Web Audio playback | ~113 | Working |
| `http_api.py` | WebSocket endpoint, static serving | +70 | Working |
| `server.py` | --dashboard startup mode | +12 | Working |
| `modules/voice.py` | VoiceGate, WhisperDecoder, VoiceEncoder | 309 | Working (not used for dashboard STT) |
| `bus.py` | ModalityBus: perceive/act, OutputQueue | 318 | Working |
