"""mod3.worker.vad — VAD subcommand handler.

Handles one operation:
- ``vad/detect``: run Silero VAD on base64-encoded PCM audio and return a
  ``VADDetectResponse`` indicating whether speech was detected.

Wires to ``vad.detect_speech()`` from the top-level vad module.
"""

from __future__ import annotations

import base64

import numpy as np

from schemas.operations import VADDetectRequest, VADDetectResponse
from schemas.wire import WireMessage

from .dispatcher import run_loop


def handle(msg: WireMessage) -> WireMessage:
    """Dispatch a VAD request to the appropriate operation handler."""
    op = msg.op or ""
    data = msg.data or {}

    if op == "detect":
        return _handle_detect(msg.id, data)
    else:
        return WireMessage(
            id=msg.id,
            type="error",
            error=f"vad: unknown op {op!r}",
            error_type="UnknownOp",
            recoverable=True,
        )


def _handle_detect(request_id: str, data: dict) -> WireMessage:
    """Run VAD on the provided audio."""
    import os

    req = VADDetectRequest.model_validate(data)

    # Decode base64 PCM -> float32 numpy
    raw_bytes = base64.b64decode(req.audio_b64)
    # PCM16 input: normalize to float32
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if os.environ.get("MOD3_WORKER_MOCK") == "1":
        # Return a deterministic mock result (silence = no speech)

        class _FakeResult:
            has_speech = False
            confidence = 0.05
            speech_ratio = 0.0
            num_segments = 0
            total_speech_sec = 0.0
            total_audio_sec = len(audio) / req.sample_rate

        result = _FakeResult()
    else:
        from vad import detect_speech

        result = detect_speech(audio, sample_rate=req.sample_rate)

    resp = VADDetectResponse(
        has_speech=result.has_speech,
        confidence=result.confidence,
        speech_ratio=result.speech_ratio,
        total_speech_sec=result.total_speech_sec,
        total_audio_sec=result.total_audio_sec,
    )
    return WireMessage(id=request_id, type="response", result=resp.model_dump())


def main() -> None:
    """Entry point for ``python -m mod3.worker vad``."""
    run_loop(handle)


if __name__ == "__main__":
    main()
