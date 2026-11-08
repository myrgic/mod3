"""Speech-to-text behind one call — ``transcribe()`` — for both compute backends.

Every STT call site in mod3 used to call ``mlx_whisper.transcribe`` directly,
which cannot run off Apple Silicon. They now call this instead:

  mlx backend  ->  ``mlx_whisper.transcribe(audio, path_or_hf_repo=model, **kw)``,
                   argument-for-argument what the call sites did before
                   (Darwin behaviour is unchanged).
  cpu backend  ->  faster-whisper (CTranslate2, ``MOD3_STT_COMPUTE_TYPE`` default int8,
                   ``MOD3_STT_DEVICE`` default cpu), loaded once per model and cached.

The signature mirrors ``mlx_whisper.transcribe`` (``path_or_hf_repo=``, ``language=``, extra kwargs) so a
call site changes by one word. Both return the mlx_whisper result shape the callers already parse:
``{"text": str, "language": str, "segments": [{"text", "start", "end", "no_speech_prob"}]}``.

cpu model mapping (mlx repo ids are what the callers pass; they are translated):
  ``*large-v3-turbo*``  ->  ``MOD3_STT_CPU_MODEL`` (default ``small``; for accuracy over latency set e.g. ``deepdml/faster-whisper-large-v3-turbo-ct2``)
  anything else (the small T1/validation model)  ->  ``MOD3_STT_CPU_SMALL_MODEL`` (default ``base``)
A value that is an existing directory is used as a local model path.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

import backend

_log = logging.getLogger(__name__)

# Measured on a 12-core WSL2 box, int8, a 2.2 s utterance, warm: large-v3-turbo 3.2-4.5 s (8 s inside the server
# next to kokoro), small 1.0 s, base 0.5 s. Conversation latency decides the default; turbo stays one env var away.
DEFAULT_CPU_MODEL = "small"
DEFAULT_CPU_SMALL_MODEL = "base"

_cpu_models: dict[str, object] = {}
_cpu_lock = threading.Lock()


def cpu_model_for(mlx_repo: str | None) -> str:
    """Translate an mlx-whisper repo id (what the callers pass) to a faster-whisper model name or path."""
    if mlx_repo and "large-v3-turbo" in mlx_repo:
        return os.environ.get("MOD3_STT_CPU_MODEL", DEFAULT_CPU_MODEL)
    if mlx_repo and os.path.isdir(mlx_repo):
        return mlx_repo
    return os.environ.get("MOD3_STT_CPU_SMALL_MODEL", DEFAULT_CPU_SMALL_MODEL)


def _get_cpu_model(name: str):
    with _cpu_lock:
        m = _cpu_models.get(name)
        if m is None:
            from faster_whisper import WhisperModel

            device = os.environ.get("MOD3_STT_DEVICE", "cpu")
            compute = os.environ.get("MOD3_STT_COMPUTE_TYPE", "int8")
            _log.info("faster-whisper: loading %s (device=%s, compute=%s)", name, device, compute)
            kw = {}
            threads = os.environ.get("MOD3_STT_CPU_THREADS")
            if threads:
                kw["cpu_threads"] = int(threads)
            try:
                m = WhisperModel(name, device=device, compute_type=compute, local_files_only=True, **kw)
            except Exception:  # not in the local HF cache yet: allow the download
                m = WhisperModel(name, device=device, compute_type=compute, **kw)
            _cpu_models[name] = m
        return m


def _transcribe_cpu(audio, model: str | None, language: str | None, **kw) -> dict:
    wm = _get_cpu_model(cpu_model_for(model))
    if isinstance(audio, np.ndarray):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
    fw_kw = {}
    for k in ("condition_on_previous_text", "initial_prompt", "beam_size"):
        if k in kw:
            fw_kw[k] = kw[k]
    if "temperature" in kw:
        fw_kw["temperature"] = kw["temperature"]
    fw_kw.setdefault("beam_size", 1)  # greedy: matches mlx_whisper's default, ~3x faster on cpu
    segs, info = wm.transcribe(audio, language=language, **fw_kw)
    segments = [{"text": s.text, "start": s.start, "end": s.end, "no_speech_prob": s.no_speech_prob} for s in segs]
    return {
        "text": "".join(s["text"] for s in segments).strip(),
        "language": info.language,
        "segments": segments,
    }


def transcribe(audio, path_or_hf_repo: str | None = None, language: str | None = None, **kw) -> dict:
    """Transcribe ``audio`` (float32 16 kHz ndarray, or a file path); see the module docstring."""
    if backend.active_backend() == backend.CPU:
        return _transcribe_cpu(audio, path_or_hf_repo, language, **kw)
    import mlx_whisper

    if language is not None:
        kw["language"] = language
    return mlx_whisper.transcribe(audio, path_or_hf_repo=path_or_hf_repo, **kw)


def engine_info() -> dict:
    """What /capabilities reports for stt on the active backend."""
    if backend.active_backend() == backend.CPU:
        return {"engine": "faster_whisper", "model": cpu_model_for("large-v3-turbo")}
    return {"engine": "mlx_whisper", "model": "mlx-community/whisper-large-v3-turbo"}
