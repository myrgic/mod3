"""mod3.worker.stt — STT subcommand handler.

Handles two operations:
- ``stt/transcribe``: full-utterance transcription via Whisper.
  Request: ``STTTranscribeRequest`` -> Response: ``STTTranscribeResponse``.
- ``stt/transcribe_streaming``: rolling LocalAgreement-2 transcription.
  Request: ``STTStreamingRequest`` -> Response: ``STTStreamingResponse``.

Wires to ``modules.voice.WhisperDecoder`` for both paths. The decoder is
lazy-loaded on first use; the subprocess stays resident after first call.
"""

from __future__ import annotations

import base64

import numpy as np

from schemas.operations import (
    STTStreamingRequest,
    STTStreamingResponse,
    STTTranscribeRequest,
    STTTranscribeResponse,
)
from schemas.wire import WireMessage

from .dispatcher import run_loop

# Module-level lazy decoder — created on first use, reused across requests
_decoder = None


def _get_decoder():
    global _decoder
    if _decoder is None:
        from modules.voice import WhisperDecoder

        _decoder = WhisperDecoder()
    return _decoder


def handle(msg: WireMessage) -> WireMessage:
    """Dispatch an STT request to the appropriate operation handler."""
    op = msg.op or ""
    data = msg.data or {}

    if op == "transcribe":
        return _handle_transcribe(msg.id, data)
    elif op == "transcribe_streaming":
        return _handle_transcribe_streaming(msg.id, data)
    else:
        return WireMessage(
            id=msg.id,
            type="error",
            error=f"stt: unknown op {op!r}",
            error_type="UnknownOp",
            recoverable=True,
        )


def _decode_audio_b64(audio_b64: str, sample_rate: int) -> np.ndarray:
    """Decode base64 PCM16 audio to float32 numpy array at the given rate."""
    raw_bytes = base64.b64decode(audio_b64)
    audio_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sample_rate != 16000 and len(audio_f32) > 0:
        import os

        if os.environ.get("MOD3_WORKER_MOCK") == "1":
            # Simple ratio resample for tests — avoids torch dependency
            import math

            new_len = int(len(audio_f32) * 16000 / sample_rate)
            indices = np.linspace(0, len(audio_f32) - 1, new_len)
            audio_f32 = np.interp(indices, np.arange(len(audio_f32)), audio_f32).astype(np.float32)
        else:
            import torch
            import torchaudio.functional as F

            tensor = torch.from_numpy(audio_f32)
            tensor = F.resample(tensor, orig_freq=sample_rate, new_freq=16000)
            audio_f32 = tensor.numpy()

    return audio_f32


def _handle_transcribe(request_id: str, data: dict) -> WireMessage:
    """Full-utterance transcription via mlx_whisper."""
    import os
    import time

    req = STTTranscribeRequest.model_validate(data)
    audio = _decode_audio_b64(req.audio_b64, req.sample_rate)

    # Duration from sample count at 16kHz (after resampling)
    duration_sec = len(audio) / 16000 if len(audio) > 0 else 0.0

    if os.environ.get("MOD3_WORKER_MOCK") == "1":
        transcript = ""
        stt_ms = 1.0
    else:
        import mlx_whisper
        from vad import is_hallucination

        decoder = _get_decoder()
        decoder._ensure_model()

        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=decoder._model,
            language=req.language,
        )
        stt_ms = (time.perf_counter() - t0) * 1000

        transcript = result.get("text", "").strip()
        if is_hallucination(transcript):
            transcript = ""

    resp = STTTranscribeResponse(
        transcript=transcript,
        confidence=1.0,
        language=req.language,
        duration_sec=round(duration_sec, 4),
        stt_ms=round(stt_ms, 1),
    )
    return WireMessage(id=request_id, type="response", result=resp.model_dump())


def _handle_transcribe_streaming(request_id: str, data: dict) -> WireMessage:
    """Rolling LocalAgreement-2 streaming transcription."""
    import os

    req = STTStreamingRequest.model_validate(data)
    audio = _decode_audio_b64(req.audio_b64, req.sample_rate)

    if os.environ.get("MOD3_WORKER_MOCK") == "1":
        result = {
            "confirmed": "",
            "tentative": "",
            "full_text": "",
            "tier": req.tier,
            "changed": False,
            "elapsed_ms": 1.0,
            "filtered": False,
        }
    else:
        decoder = _get_decoder()
        result = decoder.decode_streaming(audio, tier=req.tier)

    resp = STTStreamingResponse(
        confirmed=result.get("confirmed", ""),
        tentative=result.get("tentative", ""),
        tier=result.get("tier", req.tier),
        elapsed_ms=result.get("elapsed_ms", 0.0),
        changed=result.get("changed", False),
        filtered=result.get("filtered", False),
    )
    return WireMessage(id=request_id, type="response", result=resp.model_dump())


def main() -> None:
    """Entry point for ``python -m mod3.worker stt``."""
    run_loop(handle)


if __name__ == "__main__":
    main()
