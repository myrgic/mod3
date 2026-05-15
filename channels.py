"""Browser channel — WebSocket adapter for the Mod³ dashboard.

Wraps a FastAPI WebSocket connection as a ChannelDescriptor on the bus.
Knows the WebSocket protocol (binary PCM / JSON control frames),
knows nothing about LLMs or agent logic.

Includes three-tier adaptive STT scheduler:
  T1 (Whisper Base, ~31ms): per-chunk during speech
  T2 (Whisper Large, ~470ms): on natural pause
  T3 (Whisper Large, ~470ms): on end-of-utterance (final)

Server→client WebSocket message types:
  audio, response_text, response_complete, interrupted,
  partial_transcript, transcript,
  trace_event  — kernel cycle-trace events (ADR-083), fanned out via
                 BrowserChannel.broadcast_trace_event().

Text input from the dashboard is forwarded to channel-client seats via
the seat registry (POST /v1/sessions/{id}/messages), so Claude Code
receives it as notifications/claude/channel. Responses flow back through
mod3_dashboard_post and speak tools on the channel client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from bus import ModalityBus
from chat_flow_log import CHAT_MESSAGE_RECEIVED, CHAT_RESPONSE_GENERATED, get_chat_flow_log, phase_timer
from modality import CognitiveEvent, EncodedOutput, ModalityType
from modules.voice import WhisperDecoder
from pipeline_state import PipelineState
from schemas.ws_chat import (
    AudioFrame,
    ConfigFrame,
    EndOfSpeechFrame,
    InboundFrame,
    InterruptedFrame,
    InterruptFrame,
    PartialTranscriptFrame,
    ResponseCompleteFrame,
    ResponseTextFrame,
    TextMessageFrame,
    TraceEventFrame,
    TranscriptFrame,
    WsErrorDetail,
    WsErrorFrame,
)

logger = logging.getLogger("mod3.channels")

# Module-level TypeAdapter for the InboundFrame discriminated union.
# Built once at import time; used in BrowserChannel._handle_json for
# validated dispatch without per-call TypeAdapter construction overhead.
_INBOUND_FRAME_ADAPTER: TypeAdapter[InboundFrame] = TypeAdapter(InboundFrame)

# ---------------------------------------------------------------------------
# Dedicated STT executor — isolated from the asyncio default thread pool.
#
# mlx_whisper.transcribe() is 1-2s CPU-bound on the ANE/GPU.  Running it on
# asyncio.to_thread() consumes a slot from the default pool, which is also
# used by bus.act() drain threads and every other to_thread call in the
# server.  Under concurrent load that starves those callers.
#
# Fix (§4 of ARCHITECTURE.md): a single-worker ThreadPoolExecutor dedicated
# to STT.  STT jobs serialise here (only one MLX graph in flight at a time
# anyway), leaving the default pool free for everything else.
#
# Lifecycle: created at module import; shut down by shutdown_stt_executor()
# which is called from the FastAPI lifespan teardown in http_api.py.
# ---------------------------------------------------------------------------

_STT_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mod3-stt",
)


def shutdown_stt_executor(wait: bool = True) -> None:
    """Shut down the module-level STT executor.

    Called from the FastAPI lifespan teardown so in-flight STT jobs are
    allowed to finish before the process exits.  Pass ``wait=False`` for
    tests that need a non-blocking teardown.
    """
    _STT_EXECUTOR.shutdown(wait=wait)


# ---------------------------------------------------------------------------
# Whisper repetition dedup — backstop for phrase-doubling hallucinations.
#
# Even with condition_on_previous_text=False and temperature=0.0, Whisper
# occasionally emits the same content twice in one transcript, e.g.:
#   "How are you? Are you okay? How are you? Are you okay?"
#
# This function detects the pattern and trims to the first occurrence.
# It runs BEFORE is_hallucination() so the BoH check sees the cleaned text.
#
# Three strategies applied in order (first match wins):
#   C — Sentence-level dedup: split on .!? and remove consecutive near-dup
#       sentences.  Catches aligned repetitions most reliably.
#   A — N-way chunk dedup: split text into N equal parts (N=2,3,4) and
#       return the first part when all N are near-identical.
#   B — Longest-repeating-suffix: Z-function O(n) scan for the longest
#       suffix s such that text ends with s repeated ≥2 times; strips the
#       trailing copies while preserving any non-repeating tail.
#
# A diagnostic INFO log is emitted when NO strategy fires, so real-world
# near-miss cases can be studied from the service log.
# ---------------------------------------------------------------------------

_DEDUP_SIMILARITY_THRESHOLD = 0.95  # chunk similarity threshold (Strategy A)
_DEDUP_SENTENCE_THRESHOLD = 0.85  # sentence-level threshold (Strategy C)
# Minimum unit size guards.  Low values let us catch single-token repetitions
# ("okay okay", "OK OK OK OK") while exact-equality in _near_equal ensures we
# don't false-positive on legitimate short phrases.
_DEDUP_MIN_UNIT_CHARS = 4  # minimum repeating unit character length
_DEDUP_MIN_UNIT_WORDS = 1  # minimum repeating unit word count
# Near-match floor: segments shorter than this use exact equality only (no
# fuzzy matching), preventing false positives on short but valid sentences.
_DEDUP_NEAR_MATCH_FLOOR = 10  # chars below which only exact match applies


def _char_similarity(a: str, b: str) -> float:
    """Character-level similarity between two strings (prefix-aligned)."""
    if not a or not b:
        return 0.0
    shorter = min(len(a), len(b))
    matches = sum(x == y for x, y in zip(a, b))
    return matches / shorter


def _word_count(s: str) -> int:
    return len(s.split())


def _near_equal(a: str, b: str, threshold: float) -> bool:
    """True when the two strings are near-equal at the given char threshold.

    Length-aware floor: if either string is shorter than _DEDUP_NEAR_MATCH_FLOOR
    we require exact equality to avoid false positives on very short segments.
    Exact equality always passes regardless of length.
    """
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    # Short segments: only exact match counts (already handled above if equal).
    if min(len(a), len(b)) < _DEDUP_NEAR_MATCH_FLOOR:
        return False
    return _char_similarity(a, b) >= threshold


# ---------------------------------------------------------------------------
# Strategy C — sentence-level dedup
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strategy_c(text: str) -> str | None:
    """Remove consecutive near-duplicate sentences.

    Splits on sentence boundaries, then walks the list removing any sentence
    that is near-identical to its predecessor.  Returns the cleaned text if
    at least one duplicate was removed, else None.

    Also handles the no-punctuation case: if there is no .!? in the text,
    this strategy yields to A/B.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    if len(sentences) < 2:
        return None

    kept: list[str] = [sentences[0]]
    removed = 0
    for sent in sentences[1:]:
        if _near_equal(sent, kept[-1], _DEDUP_SENTENCE_THRESHOLD):
            removed += 1
        else:
            kept.append(sent)

    if removed == 0:
        return None

    result = " ".join(kept)
    logger.info(
        "STT dedup [C/sentence]: removed %d duplicate sentence(s), result: %r",
        removed,
        result[:80],
    )
    return result


# ---------------------------------------------------------------------------
# Strategy A — N-way equal-chunk dedup
# ---------------------------------------------------------------------------


def _strategy_a(text: str) -> str | None:
    """Split text into N equal chunks; if all N are near-identical, keep first.

    Tries N=4,3,2 in descending order so higher-repetition patterns (4-way,
    3-way) are collapsed before falling back to 2-way.  Uses the stricter
    _DEDUP_SIMILARITY_THRESHOLD (0.95) since we're comparing char-aligned
    chunks.
    """
    stripped = text.strip()
    n_total = len(stripped)

    for n in (4, 3, 2):
        if n_total < n * _DEDUP_MIN_UNIT_CHARS:
            continue
        chunk_size = n_total // n
        chunks = [stripped[i * chunk_size : (i + 1) * chunk_size].strip() for i in range(n)]
        # Include any remainder in the last chunk.
        if n_total % n:
            chunks[-1] = stripped[(n - 1) * chunk_size :].strip()

        if (
            all(_near_equal(chunks[0], chunks[i], _DEDUP_SIMILARITY_THRESHOLD) for i in range(1, n))
            and _word_count(chunks[0]) >= _DEDUP_MIN_UNIT_WORDS
        ):
            logger.info(
                "STT dedup [A/%d-way]: kept first of %d chunks, result: %r",
                n,
                n,
                chunks[0][:80],
            )
            return chunks[0]

    return None


# ---------------------------------------------------------------------------
# Strategy B — longest-repeating-suffix via Z-function (O(n))
# ---------------------------------------------------------------------------


def _z_function(s: str) -> list[int]:
    """Compute the Z-array for string s.

    z[i] = length of the longest substring starting at s[i] that is also a
    prefix of s.  z[0] is conventionally 0 (or len(s) depending on
    convention; we set it to 0 here since it is unused in the suffix search).
    """
    n = len(s)
    z = [0] * n
    lo, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - lo])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            lo, r = i, i + z[i]
    return z


def _strategy_b(text: str) -> str | None:
    """Detect and strip a repeated trailing suffix using Z-function.

    Finds the shortest unit U (≥ _DEDUP_MIN_UNIT_CHARS) such that
    z[len(U)] >= n - len(U), meaning the entire text beyond U is a prefix
    match of U repeated.  This is true iff text = U * k (or U * k with a
    truncated final copy due to stripping a trailing space).

    The >= condition (rather than ==) gracefully handles cases where the
    last repetition is one character shorter than the unit due to .strip()
    removing a trailing space (e.g. unit="go home now " stripped to
    "go home now go home now go home now go home now" len=47).

    Example:
      "Hello world. Hello world." → unit="Hello world." reps=2 → "Hello world."
      "Hello world. Hello world. Yes." → z[13] < 30-13; B does NOT fire
        (correct — trailing content preserved, Strategy C handles this case).
      "go home now go home now go home now go home now" → z[12]=35=47-12 → fires.
    """
    stripped = text.strip()
    n = len(stripped)

    z = _z_function(stripped)

    best_unit: str | None = None
    best_reps = 1

    # Start from 2 (single-char units are noise) so single short tokens like
    # "OK OK OK OK" (unit = "OK ", 3 chars) are detected.
    _b_min_chars = max(2, _DEDUP_MIN_UNIT_CHARS // 2)
    for unit_len in range(_b_min_chars, n // 2 + 1):
        # z[unit_len] >= n - unit_len means stripped[unit_len:] is fully
        # covered by the prefix stripped[:unit_len], i.e. entire text is
        # a repetition of that unit.
        if z[unit_len] >= n - unit_len:
            unit = stripped[:unit_len]
            if _word_count(unit) >= _DEDUP_MIN_UNIT_WORDS:
                reps = (n + unit_len - 1) // unit_len  # ceiling estimate
                best_unit = unit
                best_reps = reps
                break  # shortest unit_len comes first; take it

    if best_unit is not None:
        best_unit = best_unit.strip()
        logger.info(
            "STT dedup [B/suffix]: unit=%r reps~=%d, result: %r",
            best_unit[:40],
            best_reps,
            best_unit[:80],
        )
        return best_unit

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _dedup_repeated_transcript(text: str) -> str:
    """Return text with trailing repeated content removed.

    Applies three strategies in order (C → A → B); first match wins.
    Logs at INFO level when dedup fires (with which strategy) and when it
    does NOT fire (diagnostic — lets operators study real failure modes).
    """
    stripped = text.strip()
    if not stripped:
        return text

    n = len(stripped)
    if n < 4:  # too short to be a meaningful repetition
        return text

    # Strategy C — sentence-level dedup (sentence-aligned transcripts).
    result = _strategy_c(stripped)
    if result is not None:
        return result

    # Strategy B — Z-function exact-repetition (O(n), word-boundary safe).
    # Run before A because character-chunk splitting in A is unreliable at
    # word boundaries; B handles exact k-way repetition precisely.
    result = _strategy_b(stripped)
    if result is not None:
        return result

    # Strategy A — N-way near-duplicate chunk (catches fuzzy repetitions that
    # B misses because the copies differ by a word or punctuation mark).
    result = _strategy_a(stripped)
    if result is not None:
        return result

    # No strategy fired — log for diagnostics.
    logger.info("STT dedup: no match — transcript: %r", stripped[:200])
    return text


class BrowserChannel:
    """WebSocket-backed channel for the browser dashboard."""

    # Registry of currently-active dashboard channels. Used by
    # broadcast_trace_event() to fan kernel cycle-trace events out to every
    # connected dashboard client (see ADR-083). Populated in __init__,
    # pruned in _cleanup.
    _active_channels: "set[BrowserChannel]" = set()

    def __init__(
        self,
        ws: WebSocket,
        bus: ModalityBus,
        pipeline_state: PipelineState,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[CognitiveEvent], Awaitable[None]] | None = None,
    ):
        self.ws = ws
        self.bus = bus
        self.pipeline_state = pipeline_state
        self._loop = loop
        self._on_event = on_event
        self.channel_id = f"browser:{uuid.uuid4().hex[:8]}"
        self.config: dict[str, Any] = {
            "voice": "bm_lewis",
            "speed": 1.25,
            "model": "kokoro",
        }
        self._audio_buffer = bytearray()
        self._active = True

        # Three-tier STT state
        self._streaming_decoder = WhisperDecoder(load_base=True)
        self._streaming_audio = bytearray()  # Growing buffer for streaming STT
        self._last_t1_time = 0.0  # Last T1 transcription time
        self._last_speech_time = 0.0  # Last time we received speech audio
        self._t1_interval = 0.3  # Run T1 every 300ms
        self._t2_pause_threshold = 0.6  # Run T2 after 600ms pause
        self._is_speaking = False  # Whether user is currently speaking
        self._t2_scheduled = False  # Whether T2 is already scheduled

        # Register on the bus with a delivery callback
        bus.register_channel(
            self.channel_id,
            modalities=[ModalityType.VOICE, ModalityType.TEXT],
            deliver=self._deliver_sync,
        )
        BrowserChannel._active_channels.add(self)
        logger.info("BrowserChannel registered: %s", self.channel_id)

    # ------------------------------------------------------------------
    # Delivery (bus → browser)
    # ------------------------------------------------------------------

    def _deliver_sync(self, output: EncodedOutput) -> None:
        """Called from the sync OutputQueue drain thread. Bridges to async."""
        if not self._active:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._deliver_async(output), self._loop)
            future.result(timeout=10.0)
        except (WebSocketDisconnect, RuntimeError, TimeoutError):
            logger.debug("deliver failed (client disconnected?), deactivating channel")
            self._active = False

    async def _deliver_async(self, output: EncodedOutput) -> None:
        """Send encoded output over the WebSocket."""
        import base64

        logger.info(
            "deliver: modality=%s format=%s bytes=%d duration=%.1fs",
            output.modality.value if output.modality else "none",
            output.format,
            len(output.data) if output.data else 0,
            output.duration_sec,
        )

        if output.modality == ModalityType.VOICE and output.data:
            # Send audio as base64 JSON (avoids binary frame issues)
            audio_b64 = base64.b64encode(output.data).decode("ascii")
            logger.info("deliver: sending base64 audio JSON (%d chars)", len(audio_b64))
            frame = AudioFrame(
                type="audio",
                data=audio_b64,
                format=output.format or "wav",
                duration_sec=round(output.duration_sec, 2),
                sample_rate=output.metadata.get("sample_rate", 24000),
            )
            await self.ws.send_json(frame.model_dump(exclude_none=True))
            logger.info("deliver: audio sent OK")
        elif output.modality == ModalityType.TEXT:
            text = output.data.decode("utf-8") if isinstance(output.data, bytes) else str(output.data)
            logger.info("deliver: sending text response (%d chars)", len(text))
            frame = ResponseTextFrame(type="response_text", text=text)
            await self.ws.send_json(frame.model_dump(exclude_none=True))
        else:
            logger.warning("deliver: unhandled modality %s, dropping", output.modality)

    # ------------------------------------------------------------------
    # Receive loop (browser → server)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main receive loop — runs until WebSocket disconnects."""
        try:
            while True:
                message = await self.ws.receive()
                msg_type = message.get("type", "")
                if msg_type == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"]:
                    self._handle_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    await self._handle_json(json.loads(message["text"]))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("BrowserChannel error: %s", e)
        finally:
            self._cleanup()

    def _handle_audio(self, pcm_bytes: bytes) -> None:
        """Binary frame: raw Int16 PCM at 16kHz from browser Silero VAD.

        A5: Receives streaming audio during speech (from onFrameProcessed)
        AND the final complete buffer (from onSpeechEnd). Both accumulate
        for the final T3 utterance processing.

        During speech, audio also accumulates in _streaming_audio for T1/T2
        partial transcription.
        """
        self._audio_buffer.extend(pcm_bytes)
        self._streaming_audio.extend(pcm_bytes)
        self._last_speech_time = time.monotonic()
        self._is_speaking = True

        # T1: Fast Whisper Base transcription every _t1_interval
        now = time.monotonic()
        if now - self._last_t1_time >= self._t1_interval and len(self._streaming_audio) > 6400:
            self._last_t1_time = now
            asyncio.ensure_future(self._run_t1())

        # Schedule T2 check on pause detection
        if not self._t2_scheduled:
            asyncio.ensure_future(self._schedule_t2_on_pause())

    async def _handle_json(self, msg: dict) -> None:
        """JSON frame: control message dispatch via discriminated InboundFrame union."""
        msg_type = msg.get("type", "")
        logger.info("Received JSON: type=%s", msg_type)

        try:
            frame: InboundFrame = _INBOUND_FRAME_ADAPTER.validate_python(msg)  # type: ignore[assignment]
        except (ValidationError, KeyError) as exc:
            logger.warning("Unrecognised inbound frame (type=%r): %s", msg_type, exc)
            await self._send_error("invalid_frame", f"unrecognised frame type: {msg_type!r}")
            return

        try:
            if isinstance(frame, EndOfSpeechFrame):
                await self._process_utterance()
            elif isinstance(frame, TextMessageFrame):
                text = frame.text.strip()
                if text:
                    await self._process_text(text)
            elif isinstance(frame, InterruptFrame):
                await self._handle_interrupt()
            elif isinstance(frame, ConfigFrame):
                if frame.model is not None:
                    self.config["model"] = frame.model
                if frame.voice is not None:
                    self.config["voice"] = frame.voice
                if frame.speed is not None:
                    self.config["speed"] = frame.speed
        except Exception as exc:  # noqa: BLE001
            logger.error("handler error for frame type=%r: %s", msg_type, exc)
            await self._send_error("handler_error", str(exc))

    # ------------------------------------------------------------------
    # Three-Tier STT
    # ------------------------------------------------------------------

    async def _run_t1(self) -> None:
        """T1: Fast Whisper Base transcription on growing audio buffer (~31ms).

        Runs every ~300ms during speech. Emits partial_transcript with
        confirmed/tentative text at 30% opacity.
        """
        if not self._streaming_audio:
            return

        pcm_data = bytes(self._streaming_audio)

        def _transcribe_t1():
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 4800:  # <300ms
                return None
            return self._streaming_decoder.decode_streaming(audio, tier="t1")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_STT_EXECUTOR, _transcribe_t1)
            if result and result.get("changed") and not result.get("filtered"):
                frame = PartialTranscriptFrame(
                    type="partial_transcript",
                    confirmed=result["confirmed"],
                    tentative=result["tentative"],
                    tier="t1",
                    elapsed_ms=result["elapsed_ms"],
                )
                await self.ws.send_json(frame.model_dump(exclude_none=True))
        except Exception as e:
            logger.debug("T1 error: %s", e)

    async def _run_t2(self) -> None:
        """T2: Large model transcription on natural pause (~470ms).

        Runs when speech pauses for >600ms but hasn't ended. Emits
        partial_transcript with higher confidence (60% opacity).
        """
        if not self._streaming_audio:
            return

        pcm_data = bytes(self._streaming_audio)

        def _transcribe_t2():
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 8000:  # <500ms
                return None
            return self._streaming_decoder.decode_streaming(audio, tier="t2")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_STT_EXECUTOR, _transcribe_t2)
            if result and not result.get("filtered"):
                frame = PartialTranscriptFrame(
                    type="partial_transcript",
                    confirmed=result["confirmed"],
                    tentative=result["tentative"],
                    tier="t2",
                    elapsed_ms=result["elapsed_ms"],
                )
                await self.ws.send_json(frame.model_dump(exclude_none=True))
        except Exception as e:
            logger.debug("T2 error: %s", e)
        finally:
            self._t2_scheduled = False

    async def _schedule_t2_on_pause(self) -> None:
        """Check if speech has paused long enough for T2."""
        await asyncio.sleep(self._t2_pause_threshold)
        if not self._is_speaking:
            return
        # Check if there's been a pause since last audio
        silence = time.monotonic() - self._last_speech_time
        if silence >= self._t2_pause_threshold and not self._t2_scheduled:
            self._t2_scheduled = True
            await self._run_t2()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def _process_utterance(self) -> None:
        """T3: PCM audio buffer → WhisperDecoder STT → CognitiveEvent → agent loop.

        This is the final tier — end-of-utterance. Uses the Large model for
        maximum accuracy. Everything is confirmed (100% opacity).

        Skips the server-side VoiceGate (Silero VAD) because the browser
        already ran Silero VAD client-side — no need to validate again,
        and it avoids the torchaudio dependency for resampling.
        """
        pcm_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        # Reset streaming state
        self._streaming_audio.clear()
        self._streaming_decoder.reset_streaming()
        self._is_speaking = False

        if len(pcm_data) < 6400:  # <200ms at 16kHz Int16
            return

        # stt_capture: time from PCM buffer ready (EndOfSpeech received) to
        # just before we hand off to the STT executor.  Measures the in-process
        # overhead between VAD speech-end and the transcription job starting.
        _stt_capture_t0 = time.perf_counter()
        _msg_id = ""  # voice path has no message_id until after STT

        # Transcribe via mlx_whisper — needs a temp WAV file
        def _transcribe():
            import io
            import os
            import struct
            import tempfile

            import mlx_whisper
            import numpy as np

            from vad import is_hallucination

            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Skip silence
            if len(audio) < 16000 * 0.3:
                return None
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.005:
                return None

            # mlx_whisper needs a file path — write temp WAV
            buf = io.BytesIO()
            buf.write(b"RIFF")
            buf.write(struct.pack("<I", 36 + len(pcm_data)))
            buf.write(b"WAVE")
            buf.write(b"fmt ")
            buf.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
            buf.write(b"data")
            buf.write(struct.pack("<I", len(pcm_data)))
            buf.write(pcm_data)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(buf.getvalue())
                tmp_path = f.name

            try:
                result = mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    language="en",
                    # Prevent context-driven repetition loops — each window
                    # decodes independently rather than being primed by the
                    # prior window's output.
                    condition_on_previous_text=False,
                    # Deterministic decoding: single temperature, no fallback
                    # sampling.  Reduces the stochastic paths that produce
                    # phantom phrase doubling.
                    temperature=0.0,
                )
                transcript = result.get("text", "").strip()
                transcript = _dedup_repeated_transcript(transcript)
                logger.info("STT: '%s' (%.1fs, rms=%.3f)", transcript[:80], len(audio) / 16000, rms)

                if not transcript or is_hallucination(transcript):
                    return None

                return CognitiveEvent(
                    modality=ModalityType.VOICE,
                    content=transcript,
                    source_channel=self.channel_id,
                    confidence=1.0,
                )
            except Exception as e:
                logger.error("STT failed: %s", e)
                return None
            finally:
                os.unlink(tmp_path)

        loop = asyncio.get_event_loop()

        # Emit stt_capture: covers the overhead from EndOfSpeech to executor handoff.
        _stt_capture_ms = int((time.perf_counter() - _stt_capture_t0) * 1000)
        try:
            get_chat_flow_log().emit_phase(
                phase_name="stt_capture",
                session_id=self.channel_id,
                message_id=_msg_id,
                duration_ms=_stt_capture_ms,
            )
        except Exception:  # noqa: BLE001
            pass

        try:
            async with phase_timer("stt_transcribe", self.channel_id, _msg_id):
                event = await loop.run_in_executor(_STT_EXECUTOR, _transcribe)
        except Exception as exc:  # noqa: BLE001
            logger.error("STT executor error: %s", exc)
            await self._send_error("stt_failed", str(exc))
            return

        # stt_ms is used only for the TranscriptFrame display; the authoritative
        # wall-time is captured in the chat.phase.stt_transcribe event above.
        stt_ms = 0.0

        if event and event.content:
            # Send transcript to browser
            frame = TranscriptFrame(
                type="transcript",
                text=event.content,
                stt_ms=round(stt_ms, 1),
                source="voice",
            )
            await self.ws.send_json(frame.model_dump(exclude_none=True))
            # Forward to agent loop
            event.metadata["stt_ms"] = stt_ms
            if self._on_event:
                await self._on_event(event)

    async def _process_text(self, text: str) -> None:
        """Text message → CognitiveEvent → agent loop."""
        _msg_id = str(uuid.uuid4())[:8]
        try:
            get_chat_flow_log().emit(
                CHAT_MESSAGE_RECEIVED,
                self.channel_id,
                _msg_id,
                "ws",
                [],
                text,
                "inbound",
            )
        except Exception:  # noqa: BLE001
            pass

        event = CognitiveEvent(
            modality=ModalityType.TEXT,
            content=text,
            source_channel=self.channel_id,
            confidence=1.0,
        )
        frame = TranscriptFrame(type="transcript", text=text, source="text")
        await self.ws.send_json(frame.model_dump(exclude_none=True))
        if self._on_event:
            await self._on_event(event)

        try:
            get_chat_flow_log().emit(
                CHAT_RESPONSE_GENERATED,
                self.channel_id,
                _msg_id,
                "ws",
                [],
                "",
                "outbound",
            )
        except Exception:  # noqa: BLE001
            pass

    async def _handle_interrupt(self) -> None:
        """Interrupt in-flight speech."""
        if self.pipeline_state.is_speaking:
            self.pipeline_state.interrupt(reason="browser_interrupt")
        frame = InterruptedFrame(type="interrupted")
        await self.ws.send_json(frame.model_dump(exclude_none=True))

    async def _send_error(self, code: str, message: str, data: Any = None) -> None:
        """Emit a structured WsErrorFrame to the browser.

        Called from exception paths in _handle_json and _process_utterance.
        Never raises — a broken WebSocket at this point is a no-op.
        """
        try:
            frame = WsErrorFrame(
                type="error",
                error=WsErrorDetail(code=code, message=message, data=data),
            )
            await self.ws.send_json(frame.model_dump(exclude_none=True))
        except Exception as exc:  # noqa: BLE001
            logger.debug("_send_error failed (WS closed?): %s", exc)

    # ------------------------------------------------------------------
    # Helper methods (called by agent loop)
    # ------------------------------------------------------------------

    async def send_response_text(self, text: str) -> None:
        """Send response text for display in chat panel."""
        if self._active:
            try:
                logger.info("send_response_text: %s", text[:100])
                frame = ResponseTextFrame(type="response_text", text=text)
                await self.ws.send_json(frame.model_dump(exclude_none=True))
            except Exception:
                self._active = False

    async def send_response_complete(self, metrics: dict | None = None) -> None:
        """Signal response is complete."""
        if self._active:
            try:
                frame = ResponseCompleteFrame(type="response_complete", metrics=metrics or {})
                await self.ws.send_json(frame.model_dump(exclude_none=True))
            except Exception:
                self._active = False

    # ------------------------------------------------------------------
    # Trace event broadcast (kernel cycle-trace → dashboards)
    # ------------------------------------------------------------------

    @classmethod
    def broadcast_trace_event(cls, event: dict) -> None:
        """Fan a kernel cycle-trace event out to every connected dashboard.

        Per ADR-083, `event` is a pre-parsed CycleEvent dict
        (id, ts, source, cycle_id, kind, payload). Wrapped in the
        `{"type": "trace_event", "event": ...}` envelope and sent to each
        active BrowserChannel's WebSocket. Clients whose send fails are
        skipped silently (they will be pruned by their own disconnect path).
        """
        tf = TraceEventFrame(type="trace_event", event=event)
        frame = tf.model_dump(exclude_none=True)
        for ch in list(cls._active_channels):
            if not ch._active:
                continue
            try:
                asyncio.run_coroutine_threadsafe(ch.ws.send_json(frame), ch._loop)
            except Exception as exc:  # noqa: BLE001 — disconnected clients are expected
                logger.debug("trace_event send failed for %s: %s", ch.channel_id, exc)

    @classmethod
    def broadcast_response_text(cls, text: str, session_id: str | None = None) -> None:
        """Push an agent-reply text frame to dashboard WebSocket clients.

        The frame matches the existing text-response shape emitted by
        `_deliver_async` and `send_response_text`:
        `{"type": "response_text", "text": <text>}`.

        If `session_id` is None (default) the frame is broadcast to every
        active dashboard channel. When provided with a `mod3:<channel_id>`
        prefix, only the matching channel receives the frame.

        Thread-safe: dispatches each WS send via `run_coroutine_threadsafe`
        on the channel's own loop, matching `broadcast_trace_event`.
        """
        rf = ResponseTextFrame(type="response_text", text=text)
        frame = rf.model_dump(exclude_none=True)
        expected_channel = None
        if session_id and session_id.startswith("mod3:"):
            expected_channel = session_id[len("mod3:") :]
        for ch in list(cls._active_channels):
            if not ch._active:
                continue
            if expected_channel and ch.channel_id != expected_channel:
                continue
            try:
                asyncio.run_coroutine_threadsafe(ch.ws.send_json(frame), ch._loop)
            except Exception as exc:  # noqa: BLE001 — disconnected clients are expected
                logger.debug("response_text send failed for %s: %s", ch.channel_id, exc)

    @classmethod
    def broadcast_response_complete(
        cls,
        metrics: dict | None = None,
        session_id: str | None = None,
    ) -> None:
        """Push a `response_complete` frame to dashboard WebSocket clients.

        Companion to :meth:`broadcast_response_text`. Routing and threading
        match `broadcast_response_text` 1:1 — pass the same `session_id` so
        the completion frame lands on the same channel that received the text
        frames for this turn.
        """
        rcf = ResponseCompleteFrame(type="response_complete", metrics=metrics or {})
        frame = rcf.model_dump(exclude_none=True)
        expected_channel = None
        if session_id and session_id.startswith("mod3:"):
            expected_channel = session_id[len("mod3:") :]
        for ch in list(cls._active_channels):
            if not ch._active:
                continue
            if expected_channel and ch.channel_id != expected_channel:
                continue
            try:
                asyncio.run_coroutine_threadsafe(ch.ws.send_json(frame), ch._loop)
            except Exception as exc:  # noqa: BLE001 — disconnected clients are expected
                logger.debug("response_complete send failed for %s: %s", ch.channel_id, exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Deactivate channel, cancel queued TTS jobs, and detach from bus.

        Fixes §3 of ARCHITECTURE.md: on browser page reload the old
        WebSocket disconnects and a new one connects with a new channel_id.
        Without full teardown, the bus retains:

          * the ChannelDescriptor's ``deliver`` callback — a bound method
            on this dead BrowserChannel that, if the drain thread fires
            against it, writes to a closed WebSocket and triggers the
            10s ``future.result(timeout=...)`` cascade in
            ``_deliver_sync``;
          * the per-channel ``ChannelQueue`` in the OutputQueueManager,
            potentially with a drain thread mid-job for stale work;
          * the channel's slot in ``_active_channels`` (used by the
            trace-event broadcast fan-out).

        We cancel queued jobs first, then unregister from the bus —
        ``unregister_channel`` severs ``ch.deliver`` so any drain-thread
        job already past the cancel point finds no callback to call when
        it reaches the ``if ch and ch.deliver`` guard in ``bus.act``'s
        ``_do_encode``.
        """
        if not self._active:
            # Already cleaned up; idempotent.
            return
        self._active = False
        BrowserChannel._active_channels.discard(self)
        cancelled = self.bus._queue_manager.cancel_channel(self.channel_id)
        self.bus.unregister_channel(self.channel_id)
        logger.info(
            "BrowserChannel disconnected: %s (cancelled %d pending jobs, detached from bus)",
            self.channel_id,
            cancelled,
        )
