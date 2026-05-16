#
# Vendored from pipecat-ai/smart-turn HEAD (4786657e242d)
# Source: inference.py
# Vendor rung: 1 (pinned copy + MANIFEST; no local patches)
# See mod3/vendor/MANIFEST.toml for provenance record.
#
# Load path adjusted: ONNX_MODEL_PATH is set relative to vendor dir.
# HuggingFace weight pin: see MANIFEST.toml [smart_turn] section.
#

"""Smart Turn v3 inference — end-of-utterance detector.

Predicts whether an audio segment represents a complete turn
(user has finished speaking) or an incomplete one (still speaking).
Uses a fine-tuned Whisper feature extractor + ONNX classifier.
"""

import os

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from vendor.smart_turn.audio_utils import truncate_audio_to_last_n_seconds

# Vendor-relative model path — resolved at import time.
_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(_VENDOR_DIR, "data", "smart-turn-v3.1.onnx")


def build_session(onnx_path: str = ONNX_MODEL_PATH) -> ort.InferenceSession:
    """Build an ONNX InferenceSession with deterministic single-threaded config."""
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)


feature_extractor = WhisperFeatureExtractor(chunk_length=8)
# Lazy session — loaded on first call to predict_endpoint so import doesn't
# fail if the weight file hasn't been placed yet.
_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = build_session(ONNX_MODEL_PATH)
    return _session


def predict_endpoint(audio_array: np.ndarray) -> dict:
    """Predict whether an audio segment is a complete turn.

    Args:
        audio_array: Float32 numpy array of audio at 16 kHz.

    Returns:
        dict with:
          - prediction: 1 = complete (end-of-turn), 0 = incomplete (still speaking)
          - probability: sigmoid probability of completion
    """
    session = _get_session()

    # Truncate/pad to 8 seconds (keeping the end).
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )

    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # batch dim

    outputs = session.run(None, {"input_features": input_features})
    probability = float(outputs[0][0])
    prediction = 1 if probability > 0.5 else 0

    return {"prediction": prediction, "probability": probability}
