"""The shared ear: ``read(samples, sr) -> EarReading``.

One cheap pass over a chunk of audio that says what kind of sound it is
(speech, music, noise, silence), with a confidence for each, plus the
measurements the decision rests on (level, voicing, flatness, band energy,
chroma).

Speech evidence comes from the Silero VAD that ``vad.py`` already wraps. Music
vs noise comes from spectral flatness and pitch-class concentration, computed
with plain numpy FFTs (no librosa dependency, no per-frame Python loops). On a
laptop-class CPU this runs far faster than real time; see the PR for numbers.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np

from .reading import KINDS, MUSIC, NOISE, SILENCE, SPEECH, EarReading

logger = logging.getLogger("mod3.ear")

ANALYSIS_SR = 16000
FRAME = 1024
HOP = 512

# Frames quieter than this are ignored by the per-frame statistics.
ACTIVE_FRAME_DB = -50.0

# Level at which a whole chunk counts as silence (logistic centre / width, dB).
SILENCE_CENTER_DB = -40.0
SILENCE_WIDTH_DB = 2.0

# Silero VAD must mark at least this much of the chunk before it is speech-ish.
SPEECH_FRACTION_FULL = 0.25
# ... or at least this many seconds of speech (grows slowly with clip length so
# a stray VAD blip in a long instrumental does not read as speech).
SPEECH_MIN_SEC = 0.4
SPEECH_MIN_SEC_PER_SEC = 0.03

# log10(flatness) centre between tonal (music) and noisy. Measured: generated
# music -3.4 .. -1.7, speech -2.6 .. -2.0, white noise -0.25, pink noise -1.2.
FLATNESS_LOG_CENTER = -1.5
FLATNESS_LOG_WIDTH = 0.25
# Pitch-class entropy centre: music 0.48 .. 0.80, noise 0.94 .. 0.97.
CHROMA_ENT_CENTER = 0.88
CHROMA_ENT_WIDTH = 0.03

BAND_EDGES_HZ = {"sub": (20.0, 80.0), "low": (80.0, 500.0), "mid": (500.0, 4000.0), "high": (4000.0, 8000.0)}

_vad_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _tables() -> dict[str, Any]:
    """Window, chroma map and band masks, built once."""
    t = _cache.get("tables")
    if t is None:
        freqs = np.fft.rfftfreq(FRAME, 1.0 / ANALYSIS_SR)
        sel = (freqs >= 55.0) & (freqs <= 2000.0)
        pitch_class = np.round(12.0 * np.log2(np.maximum(freqs, 1.0) / 440.0) + 69.0).astype(int) % 12
        chroma_map = np.zeros((12, len(freqs)), dtype=np.float32)
        for pc in range(12):
            chroma_map[pc, sel & (pitch_class == pc)] = 1.0
        t = {
            "window": np.hanning(FRAME).astype(np.float32),
            "chroma_map": chroma_map,
            "bands": {
                name: ((freqs >= lo) & (freqs < hi)).astype(np.float32) for name, (lo, hi) in BAND_EDGES_HZ.items()
            },
            "lag_lo": ANALYSIS_SR // 1000,  # 1000 Hz
            "lag_hi": ANALYSIS_SR // 60,  # 60 Hz
        }
        _cache["tables"] = t
    return t


def _to_mono_float(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim > 1:
        # (n, channels) or (channels, n): average over the short axis
        x = x.mean(axis=1 if x.shape[0] >= x.shape[1] else 0)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == ANALYSIS_SR:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), ANALYSIS_SR)
    return resample_poly(x, ANALYSIS_SR // g, int(sr) // g).astype(np.float32)


def _db(v: float) -> float:
    return float(max(20.0 * math.log10(max(v, 1e-6)), -120.0))


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    k = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(k)[:, None]
    return x[idx]


def _speech_fraction(x16: np.ndarray, threshold: float) -> tuple[float, int, float, bool]:
    """Silero's verdict: (speech share, segment count, speech seconds, Silero usable)."""
    try:
        from vad import detect_speech

        with _vad_lock:
            res = detect_speech(x16, sample_rate=ANALYSIS_SR, threshold=threshold, min_speech_duration_ms=100)
        return float(res.speech_ratio), int(res.num_segments), float(res.total_speech_sec), True
    except Exception as exc:  # torch missing, hub model unavailable, ...
        if not _cache.get("vad_warned"):
            _cache["vad_warned"] = True
            logger.warning("ear: Silero VAD unavailable (%s); speech cannot be detected", exc)
        return 0.0, 0, 0.0, False


def read(samples: np.ndarray, sr: int, *, vad_threshold: float = 0.5) -> EarReading:
    """Listen to one chunk and report what it is.

    Args:
        samples: mono float audio in [-1, 1] (int16/int32 and multichannel are
            converted). Any length; 0.5 s or more gives stable results.
        sr: sample rate of ``samples`` in Hz.
        vad_threshold: Silero speech-probability threshold (higher = stricter).
    """
    x = _to_mono_float(samples)
    x16 = _resample(x, sr) if len(x) else x
    return _read16(x16, vad_threshold=vad_threshold, duration=len(x) / sr if sr > 0 else 0.0)


# Clips longer than this are not analysed in one go by read_clip().
CLIP_DIRECT_SEC = 20.0
CLIP_WINDOW_SEC = 2.0
CLIP_MAX_WINDOWS = 10


def read_clip(samples: np.ndarray, sr: int, *, vad_threshold: float = 0.5) -> EarReading:
    """Like :func:`read`, for a whole file or utterance of any length.

    Silero runs over the entire clip (it is cheap). The spectral features come
    from up to ``CLIP_MAX_WINDOWS`` evenly spaced 2 s windows, so a two-minute
    file costs about the same as a 20 s one.
    """
    x = _to_mono_float(samples)
    x16 = _resample(x, sr) if len(x) else x
    duration = len(x) / sr if sr > 0 else 0.0
    if duration <= CLIP_DIRECT_SEC:
        return _read16(x16, vad_threshold=vad_threshold, duration=duration)

    t0 = time.perf_counter()
    frac, segs, sec, vad_ok = _speech_fraction(x16, vad_threshold)
    w = int(CLIP_WINDOW_SEC * ANALYSIS_SR)
    starts = np.linspace(0, len(x16) - w, CLIP_MAX_WINDOWS).astype(int)
    sampled = np.concatenate([x16[s : s + w] for s in starts])
    rd = _read16(
        sampled,
        vad_threshold=vad_threshold,
        duration=duration,
        speech=(frac, segs, sec, vad_ok),
        speech_duration=duration,
    )
    rd.notes.append("sampled_windows")
    rd.rms_db = round(_db(float(np.sqrt(np.mean(x16**2)))), 2)
    rd.peak_db = round(_db(float(np.max(np.abs(x16)))), 2)
    rd.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return rd


def _read16(
    x16: np.ndarray,
    *,
    vad_threshold: float,
    duration: float,
    speech: tuple[float, int, float, bool] | None = None,
    speech_duration: float | None = None,
) -> EarReading:
    t0 = time.perf_counter()
    notes: list[str] = []
    tb = _tables()
    n_in = len(x16)
    speech_duration = duration if speech_duration is None else speech_duration

    # --- level -----------------------------------------------------------
    if n_in == 0:
        rms_db = peak_db = -120.0
    else:
        rms_db = _db(float(np.sqrt(np.mean(x16**2))))
        peak_db = _db(float(np.max(np.abs(x16))))

    silence_score = _sigmoid((SILENCE_CENTER_DB - rms_db) / SILENCE_WIDTH_DB)

    # --- per-frame spectral features -------------------------------------
    voiced_fraction = 0.0
    f0_hz: float | None = None
    flatness = 1.0
    chroma_vec = np.full(12, 1.0 / 12.0)
    chroma_ent = 1.0
    band_energy = {k: 0.0 for k in BAND_EDGES_HZ}

    if n_in and silence_score < 0.999:
        fr = _frames(x16)
        rms_env = np.sqrt(np.mean(fr**2, axis=1))
        active = rms_env > 10.0 ** (ACTIVE_FRAME_DB / 20.0)
        if active.any():
            fa = fr[active]
            spec = np.fft.rfft(fa * tb["window"], axis=1)
            power = (spec.real**2 + spec.imag**2).astype(np.float64) + 1e-12

            flat_f = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
            flatness = float(np.median(flat_f))

            total = power.sum(axis=0)
            band_sums = {name: float((total * mask).sum()) for name, mask in tb["bands"].items()}
            in_range = sum(band_sums.values()) or 1.0  # shares of the 20 Hz - 8 kHz range; DC/rumble excluded
            band_energy = {name: v / in_range for name, v in band_sums.items()}

            cp = power @ tb["chroma_map"].T  # (frames, 12)
            cpn = cp / (cp.sum(axis=1, keepdims=True) + 1e-12)
            ent_f = -(cpn * np.log(cpn + 1e-12)).sum(axis=1) / math.log(12)
            chroma_ent = float(np.median(ent_f))
            pooled = cp.sum(axis=0)
            chroma_vec = pooled / (pooled.sum() + 1e-12)

            # autocorrelation pitch (cheap stand-in for pYIN)
            ac = np.fft.irfft(np.abs(np.fft.rfft(fa, n=2 * FRAME, axis=1)) ** 2, axis=1)[:, :FRAME]
            ac = ac / (ac[:, :1] + 1e-12)
            seg = ac[:, tb["lag_lo"] : tb["lag_hi"]]
            peak = seg.max(axis=1)
            lag = seg.argmax(axis=1) + tb["lag_lo"]
            voiced = (peak > 0.6) & (flat_f < 0.3)  # flatness gate removes low-frequency noise false positives
            voiced_fraction = float(voiced.mean())
            if voiced.any():
                f0_hz = float(np.median(ANALYSIS_SR / lag[voiced]))

    # --- speech evidence --------------------------------------------------
    if silence_score >= 0.999:
        speech_fraction, speech_segments, speech_sec, vad_ok = 0.0, 0, 0.0, True
        notes.append("below_silence_floor")
    else:
        speech_fraction, speech_segments, speech_sec, vad_ok = (
            speech if speech is not None else _speech_fraction(x16, vad_threshold)
        )
        if not vad_ok:
            # No speech evidence without Silero. Say so; never guess "speech" from
            # envelope statistics (a beat-driven instrumental modulates like syllables).
            notes.append("vad_unavailable")

    # --- scores -----------------------------------------------------------
    # Speech evidence: a large enough share of the chunk, or enough absolute
    # speech seconds (a short "yes" inside a longer utterance still counts).
    need_sec = max(SPEECH_MIN_SEC, SPEECH_MIN_SEC_PER_SEC * speech_duration)
    s = float(np.clip(max(speech_fraction / SPEECH_FRACTION_FULL, speech_sec / need_sec), 0.0, 1.0))
    tonal = _sigmoid((FLATNESS_LOG_CENTER - math.log10(max(flatness, 1e-9))) / FLATNESS_LOG_WIDTH)
    concentrated = _sigmoid((CHROMA_ENT_CENTER - chroma_ent) / CHROMA_ENT_WIDTH)
    musicness = 0.6 * tonal + 0.4 * concentrated

    live = 1.0 - silence_score
    scores = {
        SPEECH: live * s,
        MUSIC: live * (1.0 - s) * musicness,
        NOISE: live * (1.0 - s) * (1.0 - musicness),
        SILENCE: silence_score,
    }
    total_p = sum(scores.values()) or 1.0
    confidences = {k: round(scores[k] / total_p, 4) for k in KINDS}
    kind = max(KINDS, key=lambda k: confidences[k])

    return EarReading(
        kind=kind,
        confidence=confidences[kind],
        confidences=confidences,
        rms_db=round(rms_db, 2),
        peak_db=round(peak_db, 2),
        voiced_fraction=round(voiced_fraction, 4),
        f0_hz=None if f0_hz is None else round(f0_hz, 1),
        flatness=round(flatness, 5),
        band_energy={k: round(v, 4) for k, v in band_energy.items()},
        chroma=[round(float(v), 4) for v in chroma_vec],
        chroma_entropy=round(chroma_ent, 4),
        speech_fraction=round(speech_fraction, 4),
        speech_segments=speech_segments,
        speech_sec=round(speech_sec, 3),
        vad_available=vad_ok,
        duration_sec=round(duration, 4),
        sample_rate=ANALYSIS_SR,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        notes=notes,
    )
