"""Smart Turn end-of-utterance detector (F5).

Wraps the vendored Smart Turn v3 inference (vendor/smart_turn/) as a drop-in
end-of-utterance (EOU) detector for the mod3 inbound pipeline.

Smart Turn is a Whisper-feature-extractor + ONNX classifier trained by
pipecat-ai to predict whether an audio segment is a complete turn
(prediction=1) or an incomplete one (prediction=0). It provides a
complementary signal to VAD silence-window endpointing:

  VAD silence window:   cheap, fast, reactive to pause duration
  Smart Turn:           semantic, handles short pauses, reduces false triggers

Usage in InboundPipeline:

    detector = SmartTurnDetector()
    if detector.is_available():
        # after accumulating utterance via VAD:
        is_complete = detector.predict(utterance_float32_16khz)
        if not is_complete:
            # keep accumulating — user likely still speaking
            continue

Integration strategy (F5):
  The SmartTurnDetector is an optional gating layer applied AFTER the VAD
  silence window closes. If Smart Turn is not available (weight file absent
  or onnxruntime not installed), the pipeline falls back to VAD-only
  endpointing — behaviour is identical to the pre-F5 state.

  The detector is disabled by default and must be explicitly enabled via:
    MOD3_SMART_TURN=1  (env var)
  or by passing use_smart_turn=True to InboundPipeline.

No side effects on import. Weight file is NOT auto-fetched;
run scripts/fetch_smart_turn_weights.py to populate vendor/smart_turn/data/.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("mod3.turn_detector")


@dataclass
class TurnPrediction:
    """Result of one Smart Turn inference call."""

    is_complete: bool
    probability: float  # sigmoid confidence of completion
    skipped: bool = False  # True if inference was skipped (model unavailable)


class SmartTurnDetector:
    """End-of-utterance detector using the vendored Smart Turn v3 ONNX model.

    Thread-safe: a single session is shared across calls; the session is
    created with ORT_SEQUENTIAL execution and a single inter-op thread,
    so concurrent calls are serialised internally.

    Lazy: the model is not loaded until the first call to predict(). If the
    weight file is absent, is_available() returns False and predict() returns
    a "skipped" result without raising.

    Configurable threshold:
      By default, prediction=1 if probability > 0.5.
      Set MOD3_SMART_TURN_THRESHOLD=0.6 to raise the bar (fewer interruptions
      from mid-sentence pauses at the cost of slightly higher latency).
    """

    _DEFAULT_THRESHOLD: float = 0.5

    def __init__(self, threshold: float | None = None):
        """Initialise the detector.

        Args:
            threshold: Completion probability threshold. Defaults to
                MOD3_SMART_TURN_THRESHOLD env var, then 0.5.
        """
        if threshold is not None:
            self._threshold = threshold
        else:
            env_val = os.environ.get("MOD3_SMART_TURN_THRESHOLD", "").strip()
            try:
                self._threshold = float(env_val) if env_val else self._DEFAULT_THRESHOLD
            except ValueError:
                logger.warning(
                    "MOD3_SMART_TURN_THRESHOLD=%r is not a float; using %.1f",
                    env_val,
                    self._DEFAULT_THRESHOLD,
                )
                self._threshold = self._DEFAULT_THRESHOLD

        self._lock = threading.Lock()
        self._loaded: bool | None = None  # None = not yet attempted
        self._predict_endpoint = None  # callable once loaded

    def is_available(self) -> bool:
        """Return True if the Smart Turn model is ready for inference."""
        if self._loaded is None:
            self._try_load()
        return bool(self._loaded)

    def _try_load(self) -> None:
        """Attempt to import the vendor smart_turn module and build the session."""
        with self._lock:
            if self._loaded is not None:
                return
            try:
                from vendor.smart_turn.inference import predict_endpoint, ONNX_MODEL_PATH

                if not os.path.exists(ONNX_MODEL_PATH):
                    logger.warning(
                        "Smart Turn weight file not found at %s. "
                        "Run scripts/fetch_smart_turn_weights.py to download.",
                        ONNX_MODEL_PATH,
                    )
                    self._loaded = False
                    return

                # Warm up the session (loads ONNX graph into memory)
                dummy = np.zeros(16000, dtype=np.float32)
                predict_endpoint(dummy)
                self._predict_endpoint = predict_endpoint
                self._loaded = True
                logger.info("Smart Turn v3 detector loaded from %s", ONNX_MODEL_PATH)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Smart Turn unavailable: %s", exc)
                self._loaded = False

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> TurnPrediction:
        """Predict whether the audio represents a complete turn.

        Args:
            audio: Float32 numpy array of speech audio at 16 kHz.
                If sample_rate != 16000, the audio is NOT resampled —
                Smart Turn requires 16 kHz input. Resample upstream if needed.
            sample_rate: Sample rate of audio (must be 16000).

        Returns:
            TurnPrediction with is_complete=True if the turn appears finished.
            is_complete=False means the user is likely still speaking.
            skipped=True if the model is unavailable (callers should fall back
            to VAD-only endpointing).
        """
        if not self.is_available():
            return TurnPrediction(is_complete=True, probability=1.0, skipped=True)

        if sample_rate != 16000:
            logger.debug(
                "Smart Turn requires 16kHz input; got %dHz — returning skipped", sample_rate
            )
            return TurnPrediction(is_complete=True, probability=1.0, skipped=True)

        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            result = self._predict_endpoint(audio)
            prob = float(result["probability"])
            is_complete = prob > self._threshold
            logger.debug(
                "Smart Turn: probability=%.3f threshold=%.2f is_complete=%s",
                prob,
                self._threshold,
                is_complete,
            )
            return TurnPrediction(is_complete=is_complete, probability=prob)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Smart Turn inference failed: %s", exc)
            # Fail-open: treat as complete so the pipeline doesn't stall
            return TurnPrediction(is_complete=True, probability=1.0, skipped=True)


# ─── Process-global default detector ─────────────────────────────────────────

_default_detector: SmartTurnDetector | None = None
_default_detector_lock = threading.Lock()


def get_default_smart_turn_detector() -> SmartTurnDetector:
    """Return (or lazily create) the process-global SmartTurnDetector."""
    global _default_detector
    if _default_detector is None:
        with _default_detector_lock:
            if _default_detector is None:
                _default_detector = SmartTurnDetector()
    return _default_detector


def reset_default_smart_turn_detector() -> None:
    """For tests — reset the process-global detector."""
    global _default_detector
    with _default_detector_lock:
        _default_detector = None


__all__ = [
    "SmartTurnDetector",
    "TurnPrediction",
    "get_default_smart_turn_detector",
    "reset_default_smart_turn_detector",
]
