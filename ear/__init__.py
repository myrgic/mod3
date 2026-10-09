"""The shared ear: one analysis layer for every audio stream mod3 touches.

    from ear import read
    reading = read(samples, sr)
    reading.kind          # "speech" | "music" | "noise" | "silence"
    reading.confidences   # {"speech": 0.01, "music": 0.97, ...}

See docs/rfcs/0002-shared-ear-and-music-modality.md.
"""

from .core import read, read_clip
from .reading import KINDS, MUSIC, NOISE, SILENCE, SPEECH, EarReading

__all__ = ["read", "read_clip", "EarGate", "read_clip", "EarReading", "KINDS", "SPEECH", "MUSIC", "NOISE", "SILENCE"]
