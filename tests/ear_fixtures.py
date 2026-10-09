"""Generated audio fixtures for the shared-ear tests.

Everything here is synthesized from scratch with numpy, so the fixtures carry no
third-party material. Speech clips are the exception: they are rendered by the
mod3 TTS engine (Kokoro) and committed as small 16 kHz WAVs under
``tests/fixtures/ear/`` (see ``scripts/make_ear_speech_fixtures.py``).

Music: square/triangle-wave chord progressions with a noise-hat / kick drum
pattern. No vocals, no samples.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ear"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32), sr


def to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import io

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def speech_clips() -> dict[str, tuple[np.ndarray, int]]:
    """The committed TTS clips, keyed by file stem."""
    return {p.stem: load_wav(p) for p in sorted(FIXTURE_DIR.glob("speech_*.wav"))}


# --------------------------------------------------------------------------
# Oscillators and drums
# --------------------------------------------------------------------------


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def _phase(freq: float, n: int, sr: int) -> np.ndarray:
    return 2.0 * np.pi * freq * np.arange(n) / sr


def square(freq: float, n: int, sr: int = SR, duty: float = 0.5) -> np.ndarray:
    return np.where((_phase(freq, n, sr) / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0).astype(np.float32)


def triangle(freq: float, n: int, sr: int = SR) -> np.ndarray:
    return (2.0 * np.abs(2.0 * ((_phase(freq, n, sr) / (2 * np.pi)) % 1.0) - 1.0) - 1.0).astype(np.float32)


def _env(n: int, attack: int, release: int) -> np.ndarray:
    e = np.ones(n, dtype=np.float32)
    a = min(attack, n)
    r = min(release, n)
    if a:
        e[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r:
        e[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return e


def noise_hat(n: int, rng: np.random.Generator, decay: float = 40.0, sr: int = SR) -> np.ndarray:
    """High-passed white-noise burst with an exponential decay."""
    x = rng.standard_normal(n).astype(np.float32)
    x = np.diff(x, prepend=0.0)  # crude high-pass
    t = np.arange(n) / sr
    return (x * np.exp(-decay * t)).astype(np.float32) * 0.5


def kick(n: int, sr: int = SR) -> np.ndarray:
    """Sine with a falling pitch."""
    t = np.arange(n) / sr
    f = 50.0 + 90.0 * np.exp(-30.0 * t)
    ph = 2.0 * np.pi * np.cumsum(f) / sr
    return (np.sin(ph) * np.exp(-9.0 * t)).astype(np.float32)


def snare(n: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    body = np.sin(2 * np.pi * 190.0 * t) * np.exp(-25.0 * t)
    return (0.5 * body + 0.6 * rng.standard_normal(n) * np.exp(-22.0 * t)).astype(np.float32)


# --------------------------------------------------------------------------
# Music
# --------------------------------------------------------------------------

# Chord roots (MIDI) and qualities for a four-chord loop in A minor: Am F C G
_PROGRESSION = [(57, (0, 3, 7)), (53, (0, 4, 7)), (60, (0, 4, 7)), (55, (0, 4, 7))]


def chiptune_instrumental(bpm: float = 174.0, bars: int = 8, sr: int = SR, seed: int = 7) -> np.ndarray:
    """Fast instrumental: square-wave arpeggio lead, triangle bass, noise hats.

    Four-on-the-floor kick, backbeat snare, eighth-note hats. One chord per bar,
    the lead arpeggiates the chord in sixteenth notes with an octave jump every
    other beat. No vocals.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    step = int(round(beat / 4 * sr))  # sixteenth note
    n_steps = bars * 16
    out = np.zeros(n_steps * step + sr // 2, dtype=np.float32)

    for s in range(n_steps):
        bar = s // 16
        pos = s % 16
        root, chord = _PROGRESSION[bar % len(_PROGRESSION)]
        i0 = s * step

        # lead: arpeggio over the chord, up an octave on odd beats
        tone = chord[pos % len(chord)] + (12 if (pos // 4) % 2 else 0)
        note = root + 12 + tone
        n = step
        lead = square(midi_hz(note), n, sr, duty=0.25) * _env(n, 8, step // 3) * 0.16
        out[i0 : i0 + n] += lead

        # bass: triangle on the root, eighth notes
        if pos % 2 == 0:
            nb = step * 2
            bass = triangle(midi_hz(root - 12), nb, sr) * _env(nb, 16, step // 2) * 0.30
            out[i0 : i0 + nb] += bass

        # drums
        if pos % 4 == 0:
            nk = int(0.18 * sr)
            out[i0 : i0 + nk] += kick(nk, sr) * 0.55
        if pos in (4, 12):
            ns = int(0.16 * sr)
            out[i0 : i0 + ns] += snare(ns, rng, sr) * 0.30
        if pos % 2 == 0:
            nh = int(0.05 * sr)
            out[i0 : i0 + nh] += noise_hat(nh, rng, sr=sr) * 0.35

    return np.clip(out, -1.0, 1.0)


def triangle_pad_progression(bpm: float = 96.0, bars: int = 4, sr: int = SR, seed: int = 11) -> np.ndarray:
    """Slow instrumental: sustained triangle-wave chords, soft hats, no lead."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar_n = int(round(4 * beat * sr))
    out = np.zeros(bars * bar_n + sr // 2, dtype=np.float32)
    for b in range(bars):
        root, chord = _PROGRESSION[b % len(_PROGRESSION)]
        i0 = b * bar_n
        for iv in chord:
            wave_ = triangle(midi_hz(root + iv), bar_n, sr) * _env(bar_n, 200, 600) * 0.16
            out[i0 : i0 + bar_n] += wave_
        out[i0 : i0 + bar_n] += triangle(midi_hz(root - 12), bar_n, sr) * _env(bar_n, 100, 600) * 0.2
        for k in range(8):
            nh = int(0.04 * sr)
            j = i0 + int(k * beat / 2 * sr)
            out[j : j + nh] += noise_hat(nh, rng, sr=sr) * 0.2
    return np.clip(out, -1.0, 1.0)


def square_melody(bpm: float = 128.0, bars: int = 4, sr: int = SR) -> np.ndarray:
    """Monophonic square-wave melody over a drone, no drums."""
    beat = 60.0 / bpm
    scale = [0, 2, 3, 5, 7, 8, 10, 12]  # natural minor
    seq = [0, 2, 4, 2, 5, 4, 2, 0, 7, 5, 4, 2, 3, 2, 1, 0]
    nlen = int(round(beat / 2 * sr))
    total = bars * 8 * nlen
    out = np.zeros(total, dtype=np.float32)
    for i in range(bars * 8):
        deg = scale[seq[i % len(seq)] % len(scale)]
        out[i * nlen : (i + 1) * nlen] += square(midi_hz(69 + deg), nlen, sr) * _env(nlen, 8, nlen // 4) * 0.2
    out += triangle(midi_hz(45), total, sr) * 0.2
    return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------------
# Noise and silence
# --------------------------------------------------------------------------


def white_noise(seconds: float = 4.0, sr: int = SR, level: float = 0.1, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * sr)) * level).astype(np.float32)


def pink_noise(seconds: float = 4.0, sr: int = SR, level: float = 0.1, seed: int = 2) -> np.ndarray:
    """1/f noise by spectral shaping."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f[0] = f[1]
    spec /= np.sqrt(f)
    x = np.fft.irfft(spec, n)
    x /= max(np.sqrt(np.mean(x**2)), 1e-9)
    return (x * level).astype(np.float32)


def silence(seconds: float = 4.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def room_tone(seconds: float = 4.0, sr: int = SR, seed: int = 3) -> np.ndarray:
    """Very quiet noise floor (about -70 dBFS)."""
    return white_noise(seconds, sr, level=0.0003, seed=seed)
