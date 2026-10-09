# RFC-0002: A Shared Ear, and Music as a Modality

| Field | Value |
|---|---|
| Status | Draft |
| Author | @chazmaniandinkle (idea), Cog (draft) |
| Date | 2026-10-09 |
| Relates | RFC-0001 (mod3 as a Cog-native modality node), `modality.py` (Gate / Decoder / Encoder / ModalityModule), `vad.py` (Silero VAD + Bag of Hallucinations), `adaptive_player.py` (chunk-deficit buffer policy), `output_queue.py`, Oscine (browser DAW; its own engines are pluggable providers) |

## Summary

Mod3 hears and speaks, but it has one ear per job and no ear at all on its own
output. This RFC proposes two things:

1. **A shared ear.** One analysis layer that runs on every audio stream mod3
   touches: what comes in from a microphone or a chat attachment, and what mod3
   itself generates **before it is queued to play**. Because the ear runs about
   ten times faster than real time on short chunks, mod3 can treat audio as a
   stream of chunks it inspects and edits ahead of the listener, not a file it
   fires and forgets.
2. **Music as a modality.** Oscine plugs into mod3 as a `music` modality module:
   Oscine's renderer is the encoder, the ear is the decoder, and the gate routes
   each chunk to speech or music. Oscine's own engine providers (chip models,
   synths) sit under that encoder. The same provider shape repeats at each level.

## 1. Motivation

### 1.1 A real failure this would have caught

On 2026-10-09 a Discord user replied to a message carrying a two-minute
instrumental. The gateway treated the referenced attachment as the user's own
voice message and ran speech-to-text on it. Whisper produced a classic
hallucination ("PLEASE SUB … let us know your comment"), which reached the agent
as the user's words and interrupted a running turn.

Mod3 already has the parts that stop this (`vad.py`'s Silero gate and its Bag of
Hallucinations filter), but they sit on mod3's own inbound path. The gateway's
path never went through them. The underlying gap is that **nothing asks "what
kind of sound is this?" before choosing a decoder.** A gate that can tell music
from speech would have routed the file to a music decoder, which hears a key and
a tempo, not words.

### 1.2 Mod3 does not hear itself

Today a TTS chunk goes from the engine to `output_queue.py` to the player. The
only measurement on that path is timing (`adaptive_player.py`'s chunk-deficit
EMA). Nothing checks the audio itself: a clipped or silent chunk, a mispronounced
word, a level jump between chunks, or speech running over a music bed all reach
the listener unexamined.

### 1.3 The two systems already have the same shape

| | mod3 | Oscine |
|---|---|---|
| Plug-in unit | modality module (gate, decoder, encoder) | instrument / engine provider |
| Output path | intent → encoder → `EncodedOutput` (audio bytes) | project → renderer → audio |
| Input path | raw audio → gate → decoder → `CognitiveEvent` | audio → ear → notes, key, sections |
| Speed | synthesis ahead of playback (see §4) | offline render ahead of playback |

## 2. Design

### 2.1 The shared ear

One analysis function, called per chunk, returning a structured reading:

```
EarReading {
  kind:        speech | music | noise | silence   # with a confidence each
  level:       rms_db, peak_db, true_peak_db
  pitch:       f0 track, voiced fraction
  timbre:      spectral flatness, band energy (sub / low / mid / high)
  music:       key + confidence, tempo + confidence, onsets   # when kind=music
  speech:      voiced segments, end-of-turn probability       # when kind=speech
  anomalies:   [clipping, dropout, level_jump, hallucination_risk, ...]
}
```

The first version reuses what exists: Silero VAD and Smart Turn for speech,
librosa-style features for music, and the BoH list as a known-phantom check. It
is a library inside mod3 that both the inbound and outbound paths call, not a
separate service.

### 2.2 Inbound: the ear in front of every decoder

`raw → ear → gate decision → decoder`

- `kind=speech` → the STT decoder, as today.
- `kind=music` → the music decoder (§2.4). It returns notes, key and sections as
  a `CognitiveEvent` whose `content` is a short description and whose `metadata`
  carries the structured reading.
- `kind=noise|silence` → no decoder. The event says so rather than inventing text.

Every `CognitiveEvent` records which decoder produced it and the ear's
confidence, so an agent can tell "the user said X" from "a music file was
attached".

### 2.3 Outbound: the ear between the encoder and the queue

`intent → encoder → chunk → ear → (accept | fix | regenerate) → output_queue`

Because synthesis and analysis both run faster than playback, the pipeline can
stay ahead of the listener by a lookahead window (initially the player's existing
startup buffer, up to `HEAVY_BUFFER_MS_CAP` = 2 s). Within that window, per chunk:

- **Accept** when the reading is clean.
- **Fix** cheap problems in place: level-match to the previous chunk, trim a
  click at a boundary, duck a music bed under speech.
- **Regenerate** a chunk whose reading is wrong (dropout, clipping, or, once an
  STT round-trip check exists, words that don't match the intended text). This
  costs one more synthesis call and must fit inside the lookahead; if it can't,
  the chunk plays as-is and the anomaly is logged.

Barge-in already cancels queued output. With the ear on the outbound path, the
queue also knows *what* it is about to say, so a cancelled chunk can be reported
precisely ("stopped before: …").

### 2.4 Music as a modality

Add `ModalityType.MUSIC` and a `MusicModule(ModalityModule)`:

- **Encoder:** a score or Oscine project goes in, audio comes out. Oscine renders
  in Node or the browser, so the module calls it in another process, through
  Oscine's MCP endpoint (`/mcp`) or its offline render command, rather than
  importing it.
- **Decoder:** the ear, in music mode: notes, key, tempo, section boundaries,
  each with a confidence.
- **Gate:** the shared ear's `kind` decision.

`CognitiveIntent.content` is a string today. For music that is not enough: the
intent is a structured score. Proposal: keep `content` as the human-readable
summary and carry the score in `metadata["score"]` in the first slice, then
promote it to a typed field if a second structured modality needs the same thing.

Oscine's own providers (chip models, synths, later real sound-driver emulation)
stay inside Oscine. Mod3 sees one music encoder; Oscine decides which engine
renders each part. The provider interface nests: mod3 → music module → Oscine's
engine providers.

## 3. Slices, each with a gate that is a command

1. **Ear library + inbound gate.** `ear.read(chunk)` with `kind` + confidence.
   Gate: a fixture set (speech, music, noise, silence clips) classified correctly;
   the fixture that caused §1.1 (an instrumental) yields `kind=music` and no
   transcript.
2. **Outbound ear, observe-only.** Every TTS chunk is read before queueing and the
   reading is logged next to the existing playback metrics. No behaviour change.
   Gate: readings appear for 100% of chunks, and end-to-end latency (TTFA) does
   not regress beyond noise on the existing metrics.
3. **Outbound fixes.** Level-match and boundary-click trim. Gate: measured
   inter-chunk level jumps fall below a threshold on a fixed text set.
4. **Music module.** Oscine render as encoder, ear as decoder. Gate: a short
   project renders through mod3 and the decoder recovers its key and tempo.
5. **Hum to notes.** Mic → ear (`kind=music`, voiced pitch track) → notes into an
   Oscine project → rendered back. Gate: a hummed scale comes back as the same
   scale.

## 4. What exists (measured 2026-10-09 on the darkstar node)

| Measurement | Value |
|---|---|
| Kokoro synthesis through `/v1/synthesize` | 6.55 s of speech in 3.51 s wall time, about **1.9x real time** including the HTTP round trip |
| Ear prototype (RMS, pYIN pitch, chroma, flatness), speech | 53 ms per 0.5 s chunk (**9.5x**), 171 ms per 2 s chunk (**11.7x**) |
| Same, music (a 174 BPM chiptune mix) | 51 ms per 0.5 s chunk (**9.8x**), 170 ms per 2 s chunk (**11.8x**) |
| Player lookahead cap already in code | `HEAVY_BUFFER_MS_CAP = 2000.0` |

So for each second of speech, synthesis takes about 0.54 s and the ear about
0.1 s, which leaves roughly a third of real time for fixes or a regeneration.
The two measurements came from different processes on one machine, under
whatever else it was running; they are a feasibility check, not a benchmark.
Even the crude ear separates the cases: the speech clip was 39% voiced with
median flatness 0.014, the music 89% voiced with flatness 0.001.

## 5. Open questions (each with a proposed default)

1. **Where does the ear live?** Default: a module inside mod3 (`ear/`), with
   Oscine calling mod3 for analysis rather than keeping its own copy.
2. **Music decoder output type.** Default: `metadata["reading"]` for slice 4;
   decide on a typed field after slice 5.
3. **Regeneration budget.** Default: at most one regeneration per chunk, only
   when it fits in the remaining lookahead; otherwise play and log.
4. **STT round-trip check on our own speech** (does the chunk say the intended
   words?). Default: off until slice 3 shows the latency budget allows it.
5. **Does the gateway's attachment path route through mod3?** Default: yes. Any
   audio a chat surface wants transcribed goes through the shared ear first.

## 6. Non-goals

- Replacing Silero VAD or Smart Turn. The ear wraps them.
- Moving Oscine's renderer into Python.
- Judging taste. The ear reports what it measured; the person listening decides.
