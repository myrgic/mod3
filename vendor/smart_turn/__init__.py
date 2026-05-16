# Vendored smart-turn package — rung 1.
# See MANIFEST.toml at vendor/ root for provenance.
# HuggingFace weight file (smart-turn-v3.1.onnx) must be placed in
# vendor/smart_turn/data/ before inference runs.
# Use scripts/fetch_smart_turn_weights.sh or see MANIFEST.toml [smart_turn].
from vendor.smart_turn.audio_utils import truncate_audio_to_last_n_seconds
from vendor.smart_turn.inference import predict_endpoint

__all__ = ["predict_endpoint", "truncate_audio_to_last_n_seconds"]
