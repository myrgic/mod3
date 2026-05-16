"""Inbound voice pipeline — continuous mic → VAD → STT → channel notification.

Runs a background thread that listens to the microphone via AudioCapture,
gates on Silero VAD to avoid waking Whisper on silence, accumulates speech
until an utterance boundary (silence window), then sends the complete
utterance through ModalityBus.perceive() for STT and BoH filtering.
Transcripts are emitted as MCP channel notifications to Claude Code.

Reflex arc: if TTS is playing when the user speaks, the pipeline calls
PipelineState.interrupt() to flush playback within ~50ms — no LLM
round-trip needed.

No side effects on import.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time

import numpy as np

from bus import ModalityBus
from capture import AudioCapture
from pipeline_state import PipelineState
from server import emit_channel_event, emit_permission_verdict
from vad import VADResult, detect_speech

# Matches verbal permission verdicts like "yes abcde", "n fghij" (case insensitive).
# The request_id is 5 lowercase letters (a-z excluding 'l').
PERMISSION_VERDICT_RE = re.compile(r"^\s*(y|yes|n|no)\s+([a-km-z]{5})\s*$", re.IGNORECASE)

logger = logging.getLogger("mod3.inbound")


class InboundPipeline:
    """Continuous voice input: mic → VAD → STT → channel notification.

    Runs in a background thread. Uses AudioCapture for mic input,
    ModalityBus.perceive() for the VAD→STT→BoH pipeline, and
    emit_channel_event() to send notifications to Claude Code.
    """

    # Default silence threshold for utterance endpointing.
    # Override per-session via MOD3_VAD_SILENCE_MS (e.g. MOD3_VAD_SILENCE_MS=700).
    # 600ms is the sweet spot: long enough for mid-sentence pauses, short enough
    # that the user doesn't feel lag at turn-end.
    _DEFAULT_SILENCE_MS: int = 600

    def __init__(
        self,
        bus: ModalityBus,
        pipeline_state: PipelineState,
        capture: AudioCapture | None = None,
        chunk_duration_sec: float = 2.0,
        vad_threshold: float = 0.5,
        speaker: str = "user",
        sample_rate: int = 16000,
        min_silence_duration_sec: float | None = None,
        loop_sleep_sec: float = 0.05,
        bargein_registry=None,
        use_smart_turn: bool | None = None,
    ):
        self._bus = bus
        self._pipeline_state = pipeline_state
        self._capture = capture or AudioCapture(sample_rate=sample_rate)
        self._chunk_sec = chunk_duration_sec
        self._vad_threshold = vad_threshold
        self._speaker = speaker
        self._sample_rate = sample_rate
        # Resolve silence duration: caller > env var > class default (600ms).
        # MOD3_VAD_SILENCE_MS is in milliseconds; stored internally as seconds.
        if min_silence_duration_sec is not None:
            self._min_silence_sec = min_silence_duration_sec
        else:
            env_ms_raw = os.environ.get("MOD3_VAD_SILENCE_MS", "").strip()
            if env_ms_raw:
                try:
                    self._min_silence_sec = int(env_ms_raw) / 1000.0
                except ValueError:
                    logger.warning(
                        "MOD3_VAD_SILENCE_MS=%r is not an integer; using default %dms",
                        env_ms_raw,
                        self._DEFAULT_SILENCE_MS,
                    )
                    self._min_silence_sec = self._DEFAULT_SILENCE_MS / 1000.0
            else:
                self._min_silence_sec = self._DEFAULT_SILENCE_MS / 1000.0
        logger.debug("VAD silence threshold: %.3fs", self._min_silence_sec)
        self._loop_sleep_sec = loop_sleep_sec
        # Optional BargeinRegistry — when provided, VAD trigger dispatches a
        # ``user_speaking_start`` event directly (in addition to the
        # ``pipeline_state.interrupt()`` reflex), removing dependence on the
        # ``/tmp/mod3-barge-in.json`` file watcher for in-process consumers.
        self._bargein_registry = bargein_registry

        # F5: Smart Turn end-of-utterance detector (optional).
        # Enabled when use_smart_turn=True or MOD3_SMART_TURN=1 env var.
        # When enabled, the Smart Turn ONNX model runs after the VAD silence
        # window closes to confirm the user has finished speaking. If Smart
        # Turn predicts incomplete (still speaking), accumulation continues.
        # When unavailable (weight absent, onnxruntime missing), falls back
        # to VAD-only endpointing transparently.
        if use_smart_turn is None:
            use_smart_turn = os.environ.get("MOD3_SMART_TURN", "").strip() in ("1", "true", "yes")
        self._use_smart_turn = use_smart_turn
        self._smart_turn_detector = None  # Lazily initialised in start()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the listening loop in a background thread."""
        if self._running:
            return

        if not self._capture.is_active():
            self._capture.start()

        # F5: Lazily initialise Smart Turn detector if enabled.
        if self._use_smart_turn and self._smart_turn_detector is None:
            try:
                from turn_detector import SmartTurnDetector

                self._smart_turn_detector = SmartTurnDetector()
                if self._smart_turn_detector.is_available():
                    logger.info("Smart Turn end-of-utterance detector enabled")
                else:
                    logger.warning(
                        "Smart Turn enabled but unavailable (weight absent or onnxruntime "
                        "missing); falling back to VAD-only endpointing"
                    )
                    self._smart_turn_detector = None
            except Exception as exc:  # noqa: BLE001
                logger.warning("Smart Turn init failed: %s; falling back to VAD-only", exc)
                self._smart_turn_detector = None

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="inbound-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("inbound pipeline started")

    def stop(self) -> None:
        """Stop the listening loop and mic capture."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._capture.stop()
        self._running = False
        logger.info("inbound pipeline stopped")

    @property
    def is_running(self) -> bool:
        """Whether the listening loop is currently active."""
        return self._running and not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Listening loop (runs in background thread)
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Main loop: chunk → VAD pre-check → accumulate → STT → notify."""
        logger.debug("listen loop entered")

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("error in listen loop tick")
                # Brief pause on error to avoid tight spin
                self._stop_event.wait(0.5)

        logger.debug("listen loop exited")

    def _tick(self) -> None:
        """Single iteration of the listen loop."""

        # 1. Grab a chunk of audio from the ring buffer
        chunk = self._capture.get_audio(self._chunk_sec)
        if chunk is None:
            # Not enough data accumulated yet — wait and retry
            self._stop_event.wait(self._loop_sleep_sec)
            return

        # 2. Fast VAD pre-check (Silero, no Whisper)
        vad_result = detect_speech(
            chunk,
            sample_rate=self._sample_rate,
            threshold=self._vad_threshold,
        )

        if not vad_result.has_speech:
            # No speech — sleep briefly and loop
            self._stop_event.wait(self._loop_sleep_sec)
            return

        # 3. Speech detected — reflex arc: interrupt TTS if speaking
        if self._pipeline_state.is_speaking:
            interrupt_info = self._pipeline_state.interrupt("vad_reflex")
            if interrupt_info is not None:
                logger.info(
                    "reflex interrupt: spoken_pct=%.1f%% reason=%s",
                    interrupt_info.spoken_pct * 100,
                    interrupt_info.reason,
                )

        # 3b. Dispatch a barge-in start event into the registry (if wired),
        # so in-process subscribers and the file mirror see the same signal
        # the legacy /tmp/mod3-barge-in.json watcher produces. Lower latency
        # than waiting for an external producer to write the signal file.
        if self._bargein_registry is not None:
            try:
                from bargein.providers.base import BargeinEvent

                self._bargein_registry._dispatch(
                    BargeinEvent(
                        source="mic_vad",
                        event_type="user_speaking_start",
                        metadata={"speech_ratio": round(vad_result.speech_ratio, 2)},
                    )
                )
            except Exception:
                logger.exception("failed to dispatch barge-in event from inbound pipeline")

        # 4. Accumulate audio until utterance boundary (silence window)
        utterance, final_vad = self._accumulate_utterance(chunk, vad_result)
        if utterance is None:
            return

        # 5. Send complete utterance through the bus pipeline (Gate → Whisper → BoH)
        audio_bytes = utterance.astype(np.float32).tobytes()
        event = self._bus.perceive(
            audio_bytes,
            modality="voice",
            channel="mod3-voice",
        )

        if event is None:
            # Gate rejected or hallucination filtered
            logger.debug("utterance filtered by bus pipeline")
            return

        # 6. Emit channel notification to Claude Code
        logger.info("transcript: %s (confidence=%.2f)", event.content[:80], event.confidence)
        self._emit_notification(event, final_vad)

    # ------------------------------------------------------------------
    # Speech accumulation
    # ------------------------------------------------------------------

    def _accumulate_utterance(
        self,
        initial_chunk: np.ndarray,
        initial_vad: VADResult,
    ) -> tuple[np.ndarray | None, VADResult]:
        """Read chunks until silence exceeds the silence window.

        Starts with the initial chunk that triggered speech detection,
        then keeps reading while VAD still reports speech. Once silence
        persists for min_silence_duration_sec, considers the utterance
        complete.

        Returns:
            (accumulated_audio, last_vad_result) or (None, last_vad) if
            the pipeline was stopped during accumulation.
        """
        chunks: list[np.ndarray] = [initial_chunk]
        last_speech_time = time.monotonic()
        last_vad = initial_vad

        while not self._stop_event.is_set():
            # Brief pause before grabbing the next chunk
            self._stop_event.wait(self._loop_sleep_sec)
            if self._stop_event.is_set():
                return None, last_vad

            chunk = self._capture.get_audio(self._chunk_sec)
            if chunk is None:
                continue

            vad_result = detect_speech(
                chunk,
                sample_rate=self._sample_rate,
                threshold=self._vad_threshold,
            )
            last_vad = vad_result

            if vad_result.has_speech:
                chunks.append(chunk)
                last_speech_time = time.monotonic()
            else:
                # Silence detected — check if we've exceeded the silence window
                silence_elapsed = time.monotonic() - last_speech_time
                if silence_elapsed >= self._min_silence_sec:
                    # VAD silence window closed — possible utterance boundary.
                    # F5: If Smart Turn is wired, run it on the accumulated audio
                    # to confirm the user has finished speaking. If Smart Turn
                    # predicts incomplete, extend the accumulation window and
                    # continue reading. Falls back transparently when unavailable.
                    if self._smart_turn_detector is not None:
                        candidate = np.concatenate(chunks)
                        # Smart Turn expects float32 at 16kHz; ensure dtype.
                        if candidate.dtype != np.float32:
                            candidate = candidate.astype(np.float32)
                        prediction = self._smart_turn_detector.predict(candidate, sample_rate=self._sample_rate)
                        if prediction.skipped:
                            # Model unavailable this call — accept the boundary
                            logger.debug("Smart Turn skipped (unavailable); accepting VAD boundary")
                            break
                        if prediction.is_complete:
                            logger.debug(
                                "Smart Turn: complete (prob=%.3f) — utterance boundary accepted",
                                prediction.probability,
                            )
                            break
                        else:
                            # User likely still speaking — extend the silence window.
                            # Reset the last-speech timer so the window re-opens
                            # from the current moment rather than stalling forever.
                            logger.debug(
                                "Smart Turn: incomplete (prob=%.3f) — extending accumulation",
                                prediction.probability,
                            )
                            last_speech_time = time.monotonic()
                            chunks.append(chunk)
                            continue
                    else:
                        # VAD-only endpointing — accept the boundary
                        break
                # Still within the grace period — keep accumulating
                # (include the silent tail so Whisper has context)
                chunks.append(chunk)

        if self._stop_event.is_set():
            return None, last_vad

        utterance = np.concatenate(chunks)
        duration = len(utterance) / self._sample_rate
        logger.debug(
            "utterance accumulated: %.1fs, %d chunks",
            duration,
            len(chunks),
        )
        return utterance, last_vad

    # ------------------------------------------------------------------
    # Notification delivery
    # ------------------------------------------------------------------

    def _emit_notification(self, event, vad_result: VADResult) -> None:
        """Send the transcript to Claude Code as a channel notification.

        If the transcript matches a permission verdict pattern (e.g. "yes abcde"),
        emits a permission verdict notification instead of a normal channel event.

        emit_channel_event() / emit_permission_verdict() are async; we run
        them synchronously from the background thread via asyncio.run().
        """
        # Check if this transcript is a permission verdict
        match = PERMISSION_VERDICT_RE.match(event.content)
        if match:
            request_id = match.group(2).lower()
            behavior = "allow" if match.group(1).lower().startswith("y") else "deny"
            logger.info(
                "permission verdict detected: %s %s (from: %r)",
                behavior,
                request_id,
                event.content,
            )
            try:
                asyncio.run(emit_permission_verdict(request_id, behavior))
            except RuntimeError as exc:
                logger.warning("failed to emit permission verdict: %s", exc)
            except Exception:
                logger.exception("unexpected error emitting permission verdict")
            return

        # Normal channel notification path
        try:
            asyncio.run(
                emit_channel_event(
                    content=event.content,
                    meta={
                        "source": "mod3-voice",
                        "speaker": self._speaker,
                        "confidence": str(round(event.confidence, 2)),
                        "speech_ratio": str(round(vad_result.speech_ratio, 2)),
                    },
                )
            )
        except RuntimeError as exc:
            # MCP session not active — log but don't crash the loop
            logger.warning("failed to emit channel event: %s", exc)
        except Exception:
            logger.exception("unexpected error emitting channel event")
