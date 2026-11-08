"""Compute-backend selection — which inference stack serves TTS and STT.

Two backends:

  ``mlx``  Apple Silicon. mlx-audio for TTS (every engine), mlx-whisper for STT.
           This is what mod3 has always run; on Darwin it is the only choice
           ``auto`` ever makes, so macOS behaviour is unchanged.
  ``cpu``  Everything else (Linux, WSL, Windows). Kokoro through kokoro-onnx
           (onnxruntime, CPU) and STT through faster-whisper (CTranslate2,
           int8 on CPU). Silero VAD is unchanged (it never needed MLX).
           The MLX-only engines (voxtral, chatterbox, chatterbox-turbo, spark)
           are reported ``available: false`` and refuse with a 4xx.

Selection is at runtime, per process: ``MOD3_BACKEND=auto|mlx|cpu`` (``onnx``
is accepted as an alias of ``cpu``). ``auto`` (the default) is ``mlx`` on
Darwin and ``cpu`` elsewhere.

This module imports nothing heavy so every other module can ask it.
"""

from __future__ import annotations

import os
import platform

MLX = "mlx"
CPU = "cpu"

# Engines the cpu backend can serve. Everything else in engine.MODELS is MLX-only.
CPU_ENGINES = frozenset({"kokoro"})


def active_backend() -> str:
    """Return ``"mlx"`` or ``"cpu"`` for this process (re-read from the env on every call)."""
    forced = os.environ.get("MOD3_BACKEND", "auto").strip().lower()
    if forced == MLX:
        return MLX
    if forced in (CPU, "onnx"):
        return CPU
    # auto: Darwin keeps MLX (that is the platform it was built for); everything else is cpu.
    return MLX if platform.system() == "Darwin" else CPU


class EngineUnavailable(RuntimeError):
    """The requested TTS engine cannot run on the active backend.

    The HTTP layer maps this to a 4xx (see http_api); it is raised *before*
    any model load so a request for an MLX-only engine on the cpu backend
    fails immediately instead of hanging or importing mlx_audio.
    """

    def __init__(self, engine: str, backend: str):
        self.engine = engine
        self.backend = backend
        super().__init__(
            f"engine {engine!r} is not available on the {backend!r} backend "
            f"(it needs MLX / Apple Silicon); the cpu backend serves only: {sorted(CPU_ENGINES)}"
        )


class UnknownVoice(ValueError):
    """A voice name the serving engine does not have (a 4xx, never a 500)."""


def engine_available(engine: str) -> bool:
    """Whether ``engine`` can run on the active backend."""
    return active_backend() == MLX or engine in CPU_ENGINES
