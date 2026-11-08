"""The cpu (ONNX / faster-whisper) compute backend.

Covers backend selection, the MLX-only engines refusing with a 4xx (never a
hang, never an mlx import), the /v1/voices + /v1/health reporting, the STT
dispatch, and that the mlx backend is untouched.

Fast and hermetic: no model is loaded. A real-model end-to-end check is
``test_onnx_kokoro_real_synthesis``, which skips unless the ONNX model and
voices are present locally (MOD3_KOKORO_ONNX_MODEL / MOD3_KOKORO_ONNX_VOICES).

Run: python3 -m pytest tests/test_cpu_backend.py -v
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend  # noqa: E402

MLX_ONLY = ("voxtral", "chatterbox", "chatterbox-turbo", "spark")


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setenv("MOD3_BACKEND", "cpu")


@pytest.fixture
def mlx(monkeypatch):
    monkeypatch.setenv("MOD3_BACKEND", "mlx")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_auto_is_mlx_on_darwin(self, monkeypatch):
        monkeypatch.setenv("MOD3_BACKEND", "auto")
        monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")
        assert backend.active_backend() == "mlx"

    @pytest.mark.parametrize("system", ["Linux", "Windows"])
    def test_auto_is_cpu_elsewhere(self, monkeypatch, system):
        monkeypatch.delenv("MOD3_BACKEND", raising=False)
        monkeypatch.setattr(backend.platform, "system", lambda: system)
        assert backend.active_backend() == "cpu"

    def test_explicit_override_wins_over_platform(self, monkeypatch):
        monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")
        monkeypatch.setenv("MOD3_BACKEND", "cpu")
        assert backend.active_backend() == "cpu"
        monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
        monkeypatch.setenv("MOD3_BACKEND", "mlx")
        assert backend.active_backend() == "mlx"

    def test_onnx_is_an_alias_of_cpu(self, monkeypatch):
        monkeypatch.setenv("MOD3_BACKEND", "onnx")
        assert backend.active_backend() == "cpu"

    def test_mlx_backend_has_every_engine(self, mlx):
        import engine

        assert all(backend.engine_available(n) for n in engine.MODELS)

    def test_cpu_backend_serves_only_kokoro(self, cpu):
        import engine

        assert [n for n in engine.MODELS if backend.engine_available(n)] == ["kokoro"]


# ---------------------------------------------------------------------------
# engine layer
# ---------------------------------------------------------------------------


class TestEngineOnCpu:
    @pytest.mark.parametrize("name", MLX_ONLY)
    def test_get_model_refuses_mlx_only_engine_without_importing_mlx(self, cpu, monkeypatch, name):
        import engine

        # If anything tried to import the mlx stack, this would blow up with a different error.
        monkeypatch.setitem(sys.modules, "mlx_audio", None)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts", None)
        with pytest.raises(backend.EngineUnavailable) as ei:
            engine.get_model(name)
        assert ei.value.engine == name and ei.value.backend == "cpu"
        assert name not in engine.get_loaded_engines()

    def test_get_model_loads_kokoro_through_onnx(self, cpu, monkeypatch):
        import engine

        sentinel = object()
        fake = types.ModuleType("tts_onnx")
        fake.load_kokoro = lambda: sentinel
        monkeypatch.setitem(sys.modules, "tts_onnx", fake)
        monkeypatch.setitem(sys.modules, "mlx_audio", None)  # must not be touched
        monkeypatch.setattr(engine, "_models", {})
        assert engine.get_model("kokoro") is sentinel

    def test_get_model_on_mlx_still_uses_mlx_audio(self, mlx, monkeypatch):
        import engine

        sentinel = object()
        mlx_audio = types.ModuleType("mlx_audio")
        tts = types.ModuleType("mlx_audio.tts")
        seen = []
        tts.load = lambda model_id: seen.append(model_id) or sentinel
        mlx_audio.tts = tts
        monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts", tts)
        monkeypatch.setattr(engine, "_models", {})
        assert engine.get_model("chatterbox-turbo") is sentinel
        assert seen == [engine.MODELS["chatterbox-turbo"]["id"]]

    def test_voice_catalog_marks_availability(self, cpu):
        import engine

        cat = engine.voice_catalog(registry=_EmptyRegistry())
        assert cat["kokoro"]["available"] is True
        assert "unavailable_reason" not in cat["kokoro"]
        for name in MLX_ONLY:
            assert cat[name]["available"] is False
            assert "mlx" in cat[name]["unavailable_reason"]
            assert cat[name]["voices"]  # still listed, so a client can show them greyed out

    def test_voice_catalog_all_available_on_mlx(self, mlx):
        import engine

        cat = engine.voice_catalog(registry=_EmptyRegistry())
        assert all(e["available"] is True for e in cat.values())
        assert not any("unavailable_reason" in e for e in cat.values())

    def test_generate_audio_with_engine_override_refuses_early(self, cpu):
        import engine

        with pytest.raises(backend.EngineUnavailable):
            list(engine.generate_audio("hi", engine="spark"))


class _EmptyRegistry:
    def list(self):
        return []


# ---------------------------------------------------------------------------
# the onnx kokoro model object, with a fake kokoro-onnx
# ---------------------------------------------------------------------------


class _FakeKokoro:
    def __init__(self, model, voices):
        self.calls = []

    def get_voices(self):
        return ["af_heart", "bm_george"]

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return np.ones(2400, dtype=np.float32) * 0.1, 24000


@pytest.fixture
def onnx_model(monkeypatch):
    fake = types.ModuleType("kokoro_onnx")
    fake.Kokoro = _FakeKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake)
    import tts_onnx

    return tts_onnx.OnnxKokoroModel(model="m.onnx", voices="v.bin")


class TestOnnxKokoroModel:
    def test_result_has_the_fields_generate_audio_reads(self, onnx_model):
        (r,) = list(onnx_model.generate(text="Hello there.", voice="bm_george", speed=1.0, verbose=False))
        assert r.audio.dtype == np.float32 and r.audio.ndim == 1 and r.samples == 2400
        assert r.is_final_chunk is True
        assert r.processing_time_seconds >= 0 and r.real_time_factor >= 0
        assert r.token_count == 0 and r.peak_memory_usage == 0.0
        assert onnx_model.sample_rate == 24000

    def test_british_voice_gets_british_phonemes(self, onnx_model):
        list(onnx_model.generate(text="Tea.", voice="bm_george"))
        list(onnx_model.generate(text="Coffee.", voice="af_heart"))
        assert [c[3] for c in onnx_model._kokoro.calls] == ["en-gb", "en-us"]

    def test_speed_is_clamped_to_the_range_kokoro_onnx_accepts(self, onnx_model):
        list(onnx_model.generate(text="Fast.", voice="af_heart", speed=1.25))
        list(onnx_model.generate(text="Faster.", voice="af_heart", speed=9))
        list(onnx_model.generate(text="Slow.", voice="af_heart", speed=0.0))
        assert [c[2] for c in onnx_model._kokoro.calls] == [1.25, 2.0, 0.5]

    def test_unknown_voice_is_a_clear_error_not_a_crash(self, onnx_model):
        with pytest.raises(backend.UnknownVoice) as ei:
            list(onnx_model.generate(text="Hi.", voice="eng_uk_m_davids"))
        assert "af_heart" in str(ei.value)  # names what IS available

    def test_text_with_nothing_to_say_yields_nothing(self, onnx_model):
        assert list(onnx_model.generate(text=" ... ", voice="af_heart")) == []
        assert onnx_model._kokoro.calls == []

    def test_through_generate_audio_end_to_end(self, cpu, monkeypatch, onnx_model):
        import engine

        monkeypatch.setattr(engine, "_models", {"kokoro": onnx_model})
        chunks = list(engine.generate_audio("One. Two.", voice="bm_george", speed=1.0, stream=False))
        audio = [c for c in chunks if c.metadata]
        assert len(audio) == 2 and all(c.metadata["engine"] == "kokoro" for c in audio)
        assert all(c.sample_rate == 24000 for c in chunks)


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture
def http(cpu, monkeypatch):
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860", raise_server_exceptions=False)


class TestHttpOnCpu:
    def test_voices_lists_unavailable_engines_as_such(self, http):
        r = http.get("/v1/voices")
        assert r.status_code == 200
        engines = r.json()["engines"]
        assert engines["kokoro"]["available"] is True
        for name in MLX_ONLY:
            assert engines[name]["available"] is False

    def test_health_reports_backend_and_unavailable_engines(self, http):
        r = http.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["backend"] == "cpu"
        for name in MLX_ONLY:
            assert body["engines"][name] == "unavailable"
        assert body["engines"]["kokoro"] in ("loaded", "unloaded")

    def test_capabilities_names_the_cpu_stt_engine(self, http):
        body = http.get("/capabilities").json()
        assert body["backend"] == "cpu"
        assert body["capabilities"]["stt"]["engine"] == "faster_whisper"
        assert body["capabilities"]["tts"]["available_engines"] == ["kokoro"]

    @pytest.mark.parametrize("name", MLX_ONLY)
    def test_audio_speech_refuses_mlx_only_model_with_400(self, http, name):
        r = http.post("/v1/audio/speech", json={"input": "hi", "voice": "af_heart", "model": name})
        assert r.status_code == 400
        body = r.json()
        assert body["reason"] == "engine_unavailable_on_backend" and body["engine"] == name

    def test_synthesize_refuses_a_chatterbox_voice_with_400(self, http, monkeypatch):
        import http_api

        class _Reg:
            def get(self, name):
                return None

            def update_last_used_at(self, name):
                pass

        monkeypatch.setattr(http_api, "_registry", _Reg())
        r = http.post("/v1/synthesize", json={"text": "hi", "voice": "chatterbox"})
        assert r.status_code == 400, r.text
        assert "chatterbox" in r.json()["error"]

    def test_speak_refuses_a_voxtral_voice_with_400_and_queues_nothing(self, http):
        r = http.post("/v1/speak", json={"text": "hi", "voice": "casual_male"})
        assert r.status_code == 400, r.text
        assert r.json()["engine"] == "voxtral"

    def test_audio_speech_unknown_voice_falls_back_not_500(self, http, monkeypatch):
        """Pre-existing contract: /v1/audio/speech maps an unknown voice to af_heart. Unchanged on cpu."""
        import http_api

        seen = {}

        def fake_generate(text, voice, speed, stream):
            seen["voice"] = voice
            return iter([])

        monkeypatch.setattr(http_api, "generate_audio", fake_generate)
        http.post("/v1/audio/speech", json={"input": "hi", "voice": "eng_uk_m_davids", "model": "kokoro"})
        assert seen["voice"] == "af_heart"


class TestHttpOnMlxUnchanged:
    def test_voices_all_available(self, mlx):
        from fastapi.testclient import TestClient

        import http_api

        c = TestClient(http_api.app, base_url="http://localhost:7860", raise_server_exceptions=False)
        engines = c.get("/v1/voices").json()["engines"]
        assert all(e["available"] for e in engines.values())

    def test_audio_speech_does_not_refuse_chatterbox_on_mlx(self, mlx, monkeypatch):
        from fastapi.testclient import TestClient

        import http_api

        monkeypatch.setattr(http_api, "generate_audio", lambda *a, **k: iter([]))
        c = TestClient(http_api.app, base_url="http://localhost:7860", raise_server_exceptions=False)
        r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "af_heart", "model": "chatterbox-turbo"})
        assert r.json().get("reason") != "engine_unavailable_on_backend"  # (500 "No audio generated" from the stub)

    def test_health_has_no_unavailable_engine_on_mlx(self, mlx):
        from fastapi.testclient import TestClient

        import http_api

        c = TestClient(http_api.app, base_url="http://localhost:7860", raise_server_exceptions=False)
        body = c.get("/v1/health").json()
        assert body["backend"] == "mlx"
        assert "unavailable" not in body["engines"].values()


# ---------------------------------------------------------------------------
# STT dispatch
# ---------------------------------------------------------------------------


class TestSttDispatch:
    def test_mlx_backend_calls_mlx_whisper_with_the_original_arguments(self, mlx, monkeypatch):
        import stt_backend

        calls = []
        fake = types.ModuleType("mlx_whisper")
        fake.transcribe = lambda audio, **kw: calls.append(kw) or {"text": "ok"}
        monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
        audio = np.zeros(16000, dtype=np.float32)
        out = stt_backend.transcribe(
            audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en", temperature=0.0
        )
        assert out == {"text": "ok"}
        assert calls == [
            {"path_or_hf_repo": "mlx-community/whisper-large-v3-turbo", "language": "en", "temperature": 0.0}
        ]

    def test_mlx_backend_omits_language_when_not_given(self, mlx, monkeypatch):
        import stt_backend

        calls = []
        fake = types.ModuleType("mlx_whisper")
        fake.transcribe = lambda audio, **kw: calls.append(kw) or {"text": ""}
        monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
        stt_backend.transcribe(np.zeros(10, dtype=np.float32), path_or_hf_repo="mlx-community/whisper-base-mlx")
        assert calls == [{"path_or_hf_repo": "mlx-community/whisper-base-mlx"}]  # decode() never passed language

    def test_cpu_backend_returns_the_mlx_result_shape(self, cpu, monkeypatch):
        import stt_backend

        class Seg:
            def __init__(self, text):
                self.text, self.start, self.end, self.no_speech_prob = text, 0.0, 1.0, 0.1

        class FakeWhisper:
            def transcribe(self, audio, language=None, **kw):
                self.kw = (language, kw)
                return iter([Seg(" hello"), Seg(" world")]), types.SimpleNamespace(language="en")

        fw = FakeWhisper()
        monkeypatch.setattr(stt_backend, "_get_cpu_model", lambda name: fw)
        out = stt_backend.transcribe(
            np.zeros(16000, dtype=np.float64),
            path_or_hf_repo="mlx-community/whisper-base-mlx",
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
        )
        assert out["text"] == "hello world" and out["language"] == "en"
        assert out["segments"][0]["no_speech_prob"] == 0.1
        assert fw.kw[0] == "en"
        assert fw.kw[1]["condition_on_previous_text"] is False and fw.kw[1]["beam_size"] == 1

    def test_cpu_model_mapping(self, monkeypatch):
        import stt_backend

        monkeypatch.delenv("MOD3_STT_CPU_MODEL", raising=False)
        monkeypatch.delenv("MOD3_STT_CPU_SMALL_MODEL", raising=False)
        assert stt_backend.cpu_model_for("mlx-community/whisper-large-v3-turbo") == stt_backend.DEFAULT_CPU_MODEL
        assert stt_backend.cpu_model_for("mlx-community/whisper-base-mlx") == "base"
        assert stt_backend.DEFAULT_CPU_MODEL == "small"
        monkeypatch.setenv("MOD3_STT_CPU_MODEL", "/models/ct2")
        assert stt_backend.cpu_model_for("mlx-community/whisper-large-v3-turbo") == "/models/ct2"

    def test_no_stt_call_site_imports_mlx_whisper_directly(self):
        """The one-file rule: every STT call goes through stt_backend, so the cpu backend can't be bypassed."""
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        offenders = []
        for p in list(root.glob("*.py")) + list((root / "modules").glob("*.py")) + list((root / "mod3").rglob("*.py")):
            if p.name == "stt_backend.py":
                continue
            if re.search(r"^\s*import mlx_whisper|^\s*from mlx_whisper", p.read_text(), re.M):
                offenders.append(str(p.relative_to(root)))
        assert offenders == []


# ---------------------------------------------------------------------------
# the real thing, when the weights are here
# ---------------------------------------------------------------------------


def _have_onnx_weights() -> bool:
    try:
        import kokoro_onnx  # noqa: F401

        import tts_onnx

        return os.path.exists(tts_onnx.model_path()) and bool(tts_onnx.list_voices())
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("SKIP_TTS_TESTS") == "1" or not _have_onnx_weights(),
    reason="needs kokoro-onnx and the Kokoro ONNX model + voices locally",
)
def test_onnx_kokoro_real_synthesis(cpu, monkeypatch):
    import engine

    monkeypatch.setattr(engine, "_models", {})
    samples, sr = engine.synthesize("The stage speaks from Eclipse now.", voice="bm_george", speed=1.0)
    assert sr == 24000 and 1.0 < len(samples) / sr < 6.0
    assert float(np.abs(samples).max()) > 0.05
