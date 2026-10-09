#!/usr/bin/env python3
"""Regenerate the speech fixtures used by tests/test_ear.py.

Renders a few neutral sentences with a running mod3 (Kokoro engine), resamples
to 16 kHz mono and writes small WAVs to tests/fixtures/ear/. The committed files
are what the tests use; run this only to refresh them.

    python scripts/make_ear_speech_fixtures.py [--url http://127.0.0.1:7860]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
from ear_fixtures import FIXTURE_DIR, SR, to_wav_bytes  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

CLIPS = {
    "speech_a": (
        "af_heart",
        "The meeting is at three o'clock tomorrow, so please bring the printed reports and a pen.",
    ),
    "speech_b": ("bm_lewis", "I was walking down to the harbor this morning when the fog finally started to lift."),
    "speech_c": (
        "bf_emma",
        "Could you tell me which platform the next train to the city leaves from, and whether it stops at the airport?",
    ),
}


def synth(url: str, text: str, voice: str) -> tuple[np.ndarray, int]:
    import io
    import wave

    body = json.dumps({"text": text, "voice": voice, "speed": 1.0, "format": "wav"}).encode()
    req = urllib.request.Request(
        f"{url}/v1/synthesize", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    args = ap.parse_args()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, (voice, text) in CLIPS.items():
        x, sr = synth(args.url, text, voice)
        if sr != SR:
            from math import gcd

            g = gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        out = FIXTURE_DIR / f"{name}.wav"
        out.write_bytes(to_wav_bytes(x, SR))
        print(f"{out.name}: {len(x) / SR:.2f}s, {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
