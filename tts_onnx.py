"""Kokoro-82M on the cpu backend — kokoro-onnx (onnxruntime, CPU), no MLX.

``OnnxKokoroModel`` exposes the slice of the mlx_audio model interface that
``engine.generate_audio`` consumes (``sample_rate`` and ``generate(...)``
yielding result objects), so the whole generation pipeline — sentence
splitting, the adaptive sentence gap, the end-of-utterance feather — runs
unchanged on top of it.

Model and voice files (all overridable):

  MOD3_KOKORO_ONNX_MODEL   path to model.onnx. Default: the HF cache copy of
                           ``MOD3_KOKORO_ONNX_REPO`` (default
                           ``onnx-community/Kokoro-82M-v1.0-ONNX``: the same v1.0
                           weights and voice set as the MLX engine's
                           ``mlx-community/Kokoro-82M-bf16``), found offline
                           first (honours HF_HOME), downloaded if absent.
  MOD3_KOKORO_ONNX_VOICES  path to the packed voices file. Default:
                           ``~/.mod3/kokoro/voices.bin``, built on first use
                           from the repo's per-voice ``voices/*.bin`` files.

kokoro-onnx wants ONE combined voices file (an ``np.savez`` archive of
``name -> float32[N, 1, 256]``), but the ONNX repo ships one ``.bin`` per
voice; :func:`build_voices` packs them.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from backend import UnknownVoice

_log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
DEFAULT_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
DEFAULT_VOICE = "af_heart"  # same default as the MLX kokoro engine

_NO_SPEECH_RE = re.compile(r"\w")


def _repo() -> str:
    return os.environ.get("MOD3_KOKORO_ONNX_REPO", DEFAULT_REPO)


def _hub_call(fn, **kw):
    """Run a huggingface_hub fetch offline first (cache hit, no network), then online."""
    try:
        return fn(local_files_only=True, **kw)
    except Exception:
        return fn(**kw)


def model_path() -> str:
    explicit = os.environ.get("MOD3_KOKORO_ONNX_MODEL")
    if explicit:
        return explicit
    from huggingface_hub import hf_hub_download

    return _hub_call(hf_hub_download, repo_id=_repo(), filename="onnx/model.onnx")


def voices_path() -> str:
    explicit = os.environ.get("MOD3_KOKORO_ONNX_VOICES")
    out = Path(explicit) if explicit else Path.home() / ".mod3" / "kokoro" / "voices.bin"
    if not out.exists():
        build_voices(str(out))
    return str(out)


def build_voices(out: str) -> str:
    """Pack the repo's per-voice ``voices/*.bin`` files into one archive at ``out``."""
    from huggingface_hub import snapshot_download

    snap = _hub_call(snapshot_download, repo_id=_repo(), allow_patterns=["voices/*.bin"])
    voices: dict[str, np.ndarray] = {}
    for p in sorted(glob.glob(os.path.join(snap, "voices", "*.bin"))):
        voices[os.path.basename(p)[:-4]] = np.fromfile(p, dtype=np.float32).reshape(-1, 1, 256)
    if not voices:
        raise FileNotFoundError(f"no voices/*.bin under {snap} (repo {_repo()})")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, **voices)
    os.replace(tmp, out)
    return out


def list_voices() -> list[str]:
    """Voice names in the packed file. Cheap (reads the archive index only); [] if unavailable."""
    try:
        with zipfile.ZipFile(voices_path()) as z:
            return sorted(n[:-4] for n in z.namelist() if n.endswith(".npy"))
    except Exception as exc:  # noqa: BLE001 — a catalogue query must never raise
        _log.warning("kokoro-onnx voices unavailable: %s", exc)
        return []


_LANG = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


def lang_for(voice: str) -> str:
    """Kokoro voice ids start with a language letter (af_heart: American English female, bm_*: British male...)."""
    return _LANG.get(voice[:1], "en-us")


@dataclass
class _Result:
    """The fields engine.generate_audio reads off an mlx_audio generation result."""

    audio: np.ndarray
    processing_time_seconds: float
    real_time_factor: float  # audio seconds / wall seconds — same convention as mlx_audio
    samples: int
    token_count: int
    is_final_chunk: bool
    peak_memory_usage: float


class OnnxKokoroModel:
    sample_rate = SAMPLE_RATE

    def __init__(self, model: str | None = None, voices: str | None = None):
        from kokoro_onnx import Kokoro

        t0 = time.time()
        self._kokoro = Kokoro(model or model_path(), voices or voices_path())
        self.voices: list[str] = sorted(self._kokoro.get_voices())
        # espeak (the phonemizer) keeps global state; onnxruntime itself is thread-safe.
        self._lock = threading.Lock()
        _log.info("kokoro-onnx loaded in %.1fs (%d voices)", time.time() - t0, len(self.voices))

    def generate(
        self, text: str, voice: str, speed: float = 1.0, verbose: bool = False, **_ignored
    ) -> Iterator[_Result]:
        """Yield one result for ``text`` (engine.generate_audio already splits into sentences)."""
        if voice not in self.voices:
            raise UnknownVoice(
                f"Unknown voice {voice!r} for the cpu kokoro engine. Available: {', '.join(self.voices)}"
            )
        if not _NO_SPEECH_RE.search(text):
            return  # punctuation / whitespace only: nothing to say (kokoro-onnx would raise)
        speed = min(2.0, max(0.5, float(speed)))  # kokoro-onnx's accepted range
        t0 = time.perf_counter()
        with self._lock:
            samples, sr = self._kokoro.create(text, voice=voice, speed=speed, lang=lang_for(voice))
        wall = time.perf_counter() - t0
        samples = np.asarray(samples, dtype=np.float32).flatten()
        dur = len(samples) / sr
        yield _Result(
            audio=samples,
            processing_time_seconds=wall,
            real_time_factor=(dur / wall) if wall > 0 else 0.0,
            samples=len(samples),
            token_count=0,
            is_final_chunk=True,
            peak_memory_usage=0.0,
        )


def load_kokoro() -> OnnxKokoroModel:
    return OnnxKokoroModel()
