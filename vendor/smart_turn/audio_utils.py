#
# Vendored from pipecat-ai/smart-turn HEAD (4786657e242d)
# Source: audio_utils.py
# Vendor rung: 1 (pinned copy + MANIFEST; no local patches)
# See mod3/vendor/MANIFEST.toml for provenance record.
#

"""Audio utility functions for Smart Turn inference."""

import numpy as np


def truncate_audio_to_last_n_seconds(
    audio_array: np.ndarray, n_seconds: int = 8, sample_rate: int = 16000
) -> np.ndarray:
    """Truncate audio to the last n seconds, or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array
