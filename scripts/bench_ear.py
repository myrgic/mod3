#!/usr/bin/env python3
"""Time ear.read() on generated fixtures: ms per chunk and x real time.

    python scripts/bench_ear.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import ear  # noqa: E402
import ear_fixtures as fx  # noqa: E402


def bench(name: str, x, sr: int, seconds: float, reps: int = 5) -> None:
    n = int(seconds * sr)
    chunks = [x[s : s + n] for s in range(0, len(x) - n + 1, n)]
    ear.read(chunks[0], sr)  # warm
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for c in chunks:
            ear.read(c, sr)
        per.append((time.perf_counter() - t0) / len(chunks))
    ms = statistics.median(per) * 1000
    print(f"{name:22s} {seconds:>4.1f}s chunk: {ms:7.2f} ms   {seconds * 1000 / ms:7.1f}x real time   ({len(chunks)} chunks x {reps})")


def main() -> None:
    clips = fx.speech_clips()
    sets = {
        "speech (TTS)": next(iter(clips.values())),
        "music (174 BPM chiptune)": (fx.chiptune_instrumental(), fx.SR),
        "noise (pink)": (fx.pink_noise(seconds=12), fx.SR),
    }
    for name, (x, sr) in sets.items():
        for sec in (0.5, 2.0):
            bench(name, x, sr, sec)


if __name__ == "__main__":
    main()
