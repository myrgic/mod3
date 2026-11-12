"""Tests for the shared ear (``ear/``) and the inbound gate built on it.

Fixtures are all generated: speech comes from the committed TTS clips in
``tests/fixtures/ear/`` (see scripts/make_ear_speech_fixtures.py); music, noise
and silence are synthesized with numpy in ``tests/ear_fixtures.py``.

The speech-vs-music tests need the Silero VAD model (torch hub cache); they are
skipped, not failed, if it cannot be loaded.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ear_fixtures as fx  # noqa: E402

import ear  # noqa: E402
from bus import ModalityBus  # noqa: E402
from ear.gate import EarGate  # noqa: E402
from modality import (  # noqa: E402
    CognitiveEvent,
    Decoder,
    Encoder,
    ModalityModule,
    ModalityType,
)


def _silero_ok() -> bool:
    try:
        import vad

        vad._get_model()
        return True
    except Exception:
        return False


needs_silero = pytest.mark.skipif(not _silero_ok(), reason="Silero VAD model unavailable")


def _windows(x: np.ndarray, sr: int, seconds: float):
    n = int(seconds * sr)
    return [x[s : s + n] for s in range(0, len(x) - n + 1, n)]


# --------------------------------------------------------------------------
# fixture set: name -> (samples, sr, expected kind)
# --------------------------------------------------------------------------


def _fixture_set():
    items = {}
    for name, (x, sr) in fx.speech_clips().items():
        items[name] = (x, sr, "speech")
    items["chiptune_174bpm"] = (fx.chiptune_instrumental(bpm=174.0, bars=8), fx.SR, "music")
    items["chiptune_180bpm"] = (fx.chiptune_instrumental(bpm=180.0, bars=8, seed=21), fx.SR, "music")
    items["triangle_pad"] = (fx.triangle_pad_progression(), fx.SR, "music")
    items["square_melody"] = (fx.square_melody(), fx.SR, "music")
    items["white_noise"] = (fx.white_noise(), fx.SR, "noise")
    items["pink_noise"] = (fx.pink_noise(), fx.SR, "noise")
    items["digital_silence"] = (fx.silence(), fx.SR, "silence")
    items["room_tone"] = (fx.room_tone(), fx.SR, "silence")
    return items


FIXTURES = _fixture_set()


def test_speech_fixtures_are_committed_and_sane():
    clips = fx.speech_clips()
    assert len(clips) >= 3
    for name, (x, sr) in clips.items():
        assert sr == fx.SR
        assert 3.0 < len(x) / sr < 10.0, name
        assert np.abs(x).max() > 0.05, name


def test_instrumental_is_fast_and_has_no_vocals():
    x = FIXTURES["chiptune_174bpm"][0]
    # 8 bars of 4/4 at 174 BPM ~= 11 s
    assert 10.0 < len(x) / fx.SR < 13.0


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@needs_silero
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_whole_clip_classified_correctly(name):
    x, sr, expected = FIXTURES[name]
    r = ear.read_clip(x, sr)
    assert r.kind == expected, (name, r.kind, r.confidences)
    assert r.confidence >= 0.6, (name, r.confidence)
    assert abs(sum(r.confidences.values()) - 1.0) < 0.01


@needs_silero
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_two_second_chunks_classified_correctly(name):
    x, sr, expected = FIXTURES[name]
    for i, w in enumerate(_windows(x, sr, 2.0)):
        r = ear.read(w, sr)
        assert r.kind == expected, (name, i, r.kind, r.confidences)


@needs_silero
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_half_second_chunks(name):
    """0.5 s is short: a pause inside speech is honestly 'silence'. Hold the line on the rest."""
    x, sr, expected = FIXTURES[name]
    kinds = [ear.read(w, sr).kind for w in _windows(x, sr, 0.5)]
    if expected == "speech":
        # never mistaken for noise; the vast majority speech; the rest quiet gaps
        assert "noise" not in kinds
        assert kinds.count("speech") / len(kinds) >= 0.7, kinds
    else:
        assert "speech" not in kinds, (name, kinds)
        assert kinds.count(expected) / len(kinds) >= 0.9, (name, kinds)


@needs_silero
def test_non_speech_is_never_speech_at_any_level():
    """Quiet or loud, the instrumental and the noise beds never read as speech."""
    for name in ("chiptune_174bpm", "triangle_pad", "square_melody", "white_noise", "pink_noise"):
        x, sr, _ = FIXTURES[name]
        for gain in (1.0, 0.3, 0.1, 0.03):
            for w in _windows(x * gain, sr, 1.0):
                assert ear.read(w, sr).kind != "speech", (name, gain)


@needs_silero
def test_speech_is_found_inside_a_longer_noisy_clip():
    """A short utterance with a noise bed under it is still speech."""
    speech, sr = fx.speech_clips()["speech_a"]
    bed = fx.pink_noise(seconds=len(speech) / sr, level=0.01)
    r = ear.read(speech + bed, sr)
    assert r.kind == "speech", r.confidences


@needs_silero
def test_reading_has_the_measurements():
    x, sr, _ = FIXTURES["chiptune_174bpm"]
    r = ear.read(x[: 2 * sr], sr)
    d = r.to_dict()
    for key in (
        "kind",
        "confidence",
        "confidences",
        "rms_db",
        "peak_db",
        "voiced_fraction",
        "flatness",
        "band_energy",
        "chroma",
        "duration_sec",
    ):
        assert key in d, key
    assert set(r.confidences) == {"speech", "music", "noise", "silence"}
    assert -60 < r.rms_db < 0 and r.peak_db >= r.rms_db
    assert set(r.band_energy) == {"sub", "low", "mid", "high"}
    assert abs(sum(r.band_energy.values()) - 1.0) < 0.02
    assert len(r.chroma) == 12 and abs(sum(r.chroma) - 1.0) < 0.01
    assert 0.0 <= r.voiced_fraction <= 1.0 and 0.0 <= r.flatness <= 1.0
    assert abs(r.duration_sec - 2.0) < 0.01


def test_level_and_chroma_key_of_a_pure_tone():
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    r = ear.read(x, sr)
    assert abs(r.rms_db - (-9.03)) < 0.2
    assert abs(r.peak_db - (-6.02)) < 0.2
    assert int(np.argmax(r.chroma)) == 9  # A
    assert r.f0_hz is not None and abs(r.f0_hz - 440.0) < 8.0
    assert r.band_energy["low"] > 0.9  # 440 Hz sits in the 80-500 Hz band


def test_input_forms_agree():
    """int16, stereo and a different sample rate give the same kind as float32 mono."""
    x, sr, _ = FIXTURES["white_noise"]
    base = ear.read(x[: 2 * sr], sr)
    assert ear.read((x[: 2 * sr] * 32767).astype(np.int16), sr).kind == base.kind
    assert ear.read(np.stack([x[: 2 * sr]] * 2, axis=1), sr).kind == base.kind
    up = np.repeat(x[: 2 * sr], 3)  # crude 48 kHz
    assert ear.read(up, 48000).kind == base.kind
    assert ear.read(np.zeros(0, dtype=np.float32), sr).kind == "silence"
    assert ear.read(np.full(1600, np.nan, dtype=np.float32), sr).kind == "silence"


@needs_silero
def test_long_clip_is_sampled_not_scanned():
    long = np.tile(FIXTURES["chiptune_174bpm"][0], 6)  # ~70 s
    t0 = time.perf_counter()
    r = ear.read_clip(long, fx.SR)
    elapsed = time.perf_counter() - t0
    assert r.kind == "music"
    assert "sampled_windows" in r.notes
    assert abs(r.duration_sec - len(long) / fx.SR) < 0.01
    assert elapsed < 0.25 * (len(long) / fx.SR)  # well inside real time


def test_without_silero_the_ear_never_claims_speech(monkeypatch):
    """If the VAD cannot load, the ear says so and calls nothing speech; it does not crash."""
    import ear.core as core

    def boom(*a, **k):
        raise RuntimeError("no vad")

    monkeypatch.setattr("vad.detect_speech", boom)
    monkeypatch.setitem(core._cache, "vad_warned", True)
    for name in ("chiptune_174bpm", "triangle_pad", "white_noise", "pink_noise", "speech_a"):
        x, sr, expected = FIXTURES[name]
        r = ear.read_clip(x, sr)
        assert r.vad_available is False and "vad_unavailable" in r.notes
        assert r.kind != "speech", name
        if expected != "speech":
            assert r.kind == expected, (name, r.kind)


# --------------------------------------------------------------------------
# speed
# --------------------------------------------------------------------------


@needs_silero
@pytest.mark.parametrize("seconds,floor", [(0.5, 5.0), (2.0, 5.0)])
def test_faster_than_real_time(seconds, floor):
    """RFC target: at least 5x real time. Measured margin is far larger; the floor is loose so CI noise passes."""
    x, sr, _ = FIXTURES["chiptune_174bpm"]
    chunks = _windows(x, sr, seconds)
    ear.read(chunks[0], sr)  # warm: tables, model
    t0 = time.perf_counter()
    for c in chunks:
        ear.read(c, sr)
    per_chunk = (time.perf_counter() - t0) / len(chunks)
    assert seconds / per_chunk >= floor, f"{seconds / per_chunk:.1f}x real time"


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


class _SpyDecoder(Decoder):
    """Stands in for Whisper. Counts calls; returns a fixed transcript."""

    def __init__(self):
        self.calls = 0

    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent:
        self.calls += 1
        return CognitiveEvent(modality=ModalityType.VOICE, content="a transcript", confidence=0.9)


class _NoEncoder(Encoder):
    def encode(self, intent):  # pragma: no cover - never used
        raise NotImplementedError


class _SpyVoice(ModalityModule):
    def __init__(self):
        from modules.voice import VoiceGate

        self._gate = VoiceGate()
        self.spy = _SpyDecoder()
        self._decoder = self.spy
        self._encoder = _NoEncoder()

    modality_type = property(lambda self: ModalityType.VOICE)
    gate = property(lambda self: self._gate)
    decoder = property(lambda self: self._decoder)
    encoder = property(lambda self: self._encoder)


def _as_bus_bytes(x: np.ndarray) -> bytes:
    """What inbound.py sends: float32 PCM bytes at 16 kHz."""
    return x.astype(np.float32).tobytes()


def _spy_bus():
    bus = ModalityBus()
    voice = _SpyVoice()
    bus.register(voice)
    return bus, voice


@needs_silero
def test_gate_passes_speech_and_stt_runs():
    bus, voice = _spy_bus()
    x, sr, _ = FIXTURES["speech_a"]
    event, non_speech = bus.perceive_outcome(_as_bus_bytes(x), "voice", channel="t", sample_rate=sr)
    assert non_speech is None
    assert event is not None and event.content == "a transcript"
    assert voice.spy.calls == 1


@needs_silero
@pytest.mark.parametrize(
    "name,kind",
    [
        ("chiptune_174bpm", "music"),
        ("chiptune_180bpm", "music"),
        ("triangle_pad", "music"),
        ("square_melody", "music"),
        ("white_noise", "noise"),
        ("pink_noise", "noise"),
        ("digital_silence", "silence"),
        ("room_tone", "silence"),
    ],
)
def test_gate_never_calls_stt_on_non_speech(name, kind):
    bus, voice = _spy_bus()
    seen = []
    bus.on_event(seen.append)
    x, sr, _ = FIXTURES[name]
    event, non_speech = bus.perceive_outcome(_as_bus_bytes(x), "voice", channel="t", sample_rate=sr)

    assert voice.spy.calls == 0, "STT was called on non-speech"
    assert event is None
    assert non_speech is not None
    assert non_speech.content == ""  # no transcript, ever
    assert non_speech.metadata["non_speech"] is True
    assert non_speech.metadata["kind"] == kind
    assert non_speech.metadata["decoder"] is None
    assert "modality.non_speech" in [e.type for e in seen]
    # perceive() keeps its old contract: None
    assert bus.perceive(_as_bus_bytes(x), "voice", channel="t", sample_rate=sr) is None
    assert voice.spy.calls == 0


@needs_silero
def test_instrumental_in_a_wav_container_is_music_not_text():
    """The reported failure: an instrumental attachment sent as a WAV file."""
    bus, voice = _spy_bus()
    x, sr, _ = FIXTURES["chiptune_174bpm"]
    event, non_speech = bus.perceive_outcome(fx.to_wav_bytes(x, sr), "voice", channel="attachment")
    assert event is None and voice.spy.calls == 0
    assert non_speech is not None and non_speech.metadata["kind"] == "music"
    assert non_speech.metadata["reading"]["kind"] == "music"


@needs_silero
def test_gate_reads_int16_when_told():
    x, sr, _ = FIXTURES["speech_b"]
    pcm = (x * 32767).astype(np.int16).tobytes()
    res = EarGate().check(pcm, sample_rate=sr, sample_width=2)
    assert res.passed and res.metadata["kind"] == "speech"


def test_gate_fails_closed_when_the_ear_errors(monkeypatch):
    import ear.gate as gate_mod

    def boom(*a, **k):
        raise RuntimeError("ear down")

    monkeypatch.setattr(gate_mod, "read_clip", boom)
    res = EarGate().check(b"\x00" * 64, sample_rate=16000)
    assert res.passed is False and "ear_error" in res.reason


def test_voice_gate_falls_back_to_plain_vad_when_the_ear_errors(monkeypatch):
    import ear.gate as gate_mod
    import vad as vad_mod
    from modules.voice import VoiceGate

    monkeypatch.setattr(gate_mod, "read_clip", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ear down")))
    monkeypatch.setattr(
        vad_mod,
        "detect_speech",
        lambda *a, **k: vad_mod.VADResult(True, 0.9, 0.9, 1, 1.0, 1.0),
    )
    res = VoiceGate().check(np.zeros(16000, dtype=np.float32).tobytes(), sample_rate=16000)
    assert res.passed is True and "ear unavailable" in res.reason


# --------------------------------------------------------------------------
# HTTP: /v1/transcribe and /v1/bus/perceive
# --------------------------------------------------------------------------


@pytest.fixture()
def http_client(monkeypatch):
    from fastapi.testclient import TestClient

    import http_api

    voice = _SpyVoice()
    monkeypatch.setattr(http_api, "_stt_decoder", voice.spy)
    monkeypatch.setitem(http_api._bus._modules, ModalityType.VOICE, voice)
    monkeypatch.setattr(http_api, "_get_voice_module", lambda: voice)
    # the real module isinstance-checks the type; the spy is a stand-in
    return TestClient(http_api.app, base_url="http://localhost:7860"), voice


@needs_silero
def test_transcribe_endpoint_does_not_transcribe_an_instrumental(http_client):
    client, voice = http_client
    x, sr, _ = FIXTURES["chiptune_174bpm"]
    r = client.post("/v1/transcribe", files={"file": ("track.wav", fx.to_wav_bytes(x, sr), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == ""
    assert body["non_speech"] is True and body["kind"] == "music"
    assert voice.spy.calls == 0


@needs_silero
def test_transcribe_endpoint_still_transcribes_speech(http_client):
    client, voice = http_client
    x, sr, _ = FIXTURES["speech_a"]
    r = client.post("/v1/transcribe", files={"file": ("voice.wav", fx.to_wav_bytes(x, sr), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "a transcript"
    assert "non_speech" not in body
    assert voice.spy.calls == 1


@needs_silero
def test_bus_perceive_endpoint_reports_non_speech(http_client):
    client, voice = http_client
    x, sr, _ = FIXTURES["pink_noise"]
    r = client.post("/v1/bus/perceive", files={"file": ("n.wav", fx.to_wav_bytes(x, sr), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "non_speech" and body["kind"] == "noise"
    assert voice.spy.calls == 0
