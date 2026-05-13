"""mod3.worker.tts — TTS subcommand handler.

Handles two operations:
- ``tts/synthesize``: one-shot synthesis, returns a ``TTSSynthesizeResponse``
  packed into a ``WireMessage(type="response", result=...)``.
- ``tts/stream``: sub-sentence streaming, emits a sequence of
  ``WireMessage(type="event", event="tts.chunk", data=..., chunk=N, done=bool)``.

Wires to ``engine.generate_audio()`` for both paths. The streaming path yields
one ``WireMessage`` per ``AudioChunk`` the engine produces without buffering.
"""

from __future__ import annotations

import base64
import sys
from collections.abc import Iterator

import numpy as np

from schemas.operations import (
    TTSChunkEvent,
    TTSStreamRequest,
    TTSSynthesizeRequest,
    TTSSynthesizeResponse,
)
from schemas.wire import WireMessage

from .dispatcher import run_loop


def _float32_to_int16_b64(samples: np.ndarray) -> str:
    """Convert float32 samples to int16 PCM and base64-encode."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def _wav_b64(samples: np.ndarray, sample_rate: int) -> str:
    """Encode float32 samples as a WAV blob and return base64 string."""
    import io
    import wave

    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handle(msg: WireMessage) -> WireMessage | Iterator[WireMessage]:
    """Dispatch a TTS request to the appropriate operation handler."""
    op = msg.op or ""
    data = msg.data or {}

    if op == "synthesize":
        return _handle_synthesize(msg.id, data)
    elif op == "stream":
        return _handle_stream(msg.id, data)
    else:
        return WireMessage(
            id=msg.id,
            type="error",
            error=f"tts: unknown op {op!r}",
            error_type="UnknownOp",
            recoverable=True,
        )


def _mock_generate_audio(text: str, **kwargs):
    """Stub engine for tests (MOD3_WORKER_MOCK=1). Yields one silent chunk."""
    import os

    if os.environ.get("MOD3_WORKER_MOCK") != "1":
        raise RuntimeError("_mock_generate_audio called outside mock mode")

    # Yield a single silent chunk representing ~0.1s of audio at 24kHz
    silence = np.zeros(2400, dtype=np.float32)

    class _FakeChunk:
        samples = silence
        sample_rate = 24000
        metadata = {
            "gen_time_sec": 0.01,
            "rtf": 0.1,
            "samples": 2400,
            "tokens": 5,
            "is_final": True,
            "sentence": 0,
            "peak_memory_gb": 0.0,
            "engine": "mock",
        }

    yield _FakeChunk()


def _handle_synthesize(request_id: str, data: dict) -> WireMessage:
    """One-shot TTS: synthesize complete audio, return WAV blob."""
    import os
    import time

    if os.environ.get("MOD3_WORKER_MOCK") == "1":
        _gen = _mock_generate_audio
    else:
        from engine import generate_audio as _gen  # type: ignore[assignment]

    req = TTSSynthesizeRequest.model_validate(data)

    t0 = time.perf_counter()
    chunks = list(
        _gen(
            req.text,
            voice=req.voice,
            speed=req.speed,
            emotion=req.emotion if req.emotion is not None else 0.5,
            stream=False,
        )
    )
    gen_time = time.perf_counter() - t0

    if not chunks:
        # Empty synthesis: return silent WAV
        silence = np.zeros(0, dtype=np.float32)
        resp = TTSSynthesizeResponse(
            audio_b64=_wav_b64(silence, 24000),
            duration_sec=0.0,
            sample_rate=24000,
            engine="",
            voice=req.voice,
            gen_time_sec=round(gen_time, 4),
            rtf=0.0,
        )
        return WireMessage(id=request_id, type="response", result=resp.model_dump())

    sample_rate = chunks[0].sample_rate
    all_samples = np.concatenate([c.samples for c in chunks])
    duration_sec = len(all_samples) / sample_rate if sample_rate > 0 else 0.0
    rtf = gen_time / duration_sec if duration_sec > 0 else 0.0

    # Determine engine name from last chunk metadata
    last_meta = chunks[-1].metadata
    engine_name = last_meta.get("engine", "")

    resp = TTSSynthesizeResponse(
        audio_b64=_wav_b64(all_samples, sample_rate),
        duration_sec=round(duration_sec, 4),
        sample_rate=sample_rate,
        engine=engine_name,
        voice=req.voice,
        gen_time_sec=round(gen_time, 4),
        rtf=round(rtf, 4),
    )
    return WireMessage(id=request_id, type="response", result=resp.model_dump())


def _handle_stream(request_id: str, data: dict) -> Iterator[WireMessage]:
    """Streaming TTS: yield one WireMessage per engine chunk."""
    import os

    if os.environ.get("MOD3_WORKER_MOCK") == "1":
        _gen = _mock_generate_audio
    else:
        from engine import generate_audio as _gen  # type: ignore[assignment]

    req = TTSStreamRequest.model_validate(data)

    chunk_index = 0
    sentence_index = 0

    for audio_chunk in _gen(
        req.text,
        voice=req.voice,
        speed=req.speed,
        emotion=req.emotion if req.emotion is not None else 0.5,
        stream=True,
        streaming_interval=req.streaming_interval if req.streaming_interval > 0 else 1.0,
    ):
        meta = audio_chunk.metadata
        is_final = bool(meta.get("is_final", False))
        current_sentence = int(meta.get("sentence", sentence_index))

        samples = audio_chunk.samples
        if samples is None or len(samples) == 0:
            # Gap chunk — skip empty frames
            if is_final and sentence_index != current_sentence:
                sentence_index = current_sentence
            continue

        audio_b64 = _float32_to_int16_b64(samples)
        sample_count = len(samples)

        event_data = TTSChunkEvent(
            audio_b64=audio_b64,
            sample_rate=audio_chunk.sample_rate,
            num_channels=1,
            dtype="int16",
            chunk_index=chunk_index,
            sentence_index=current_sentence,
            is_final=is_final,
            gen_time_sec=float(meta.get("gen_time_sec", 0.0)),
            rtf=float(meta.get("rtf", 0.0)),
            peak_memory_gb=float(meta.get("peak_memory_gb", 0.0)),
            tokens=int(meta.get("tokens", 0)),
            samples=sample_count,
            engine=str(meta.get("engine", "")),
            voice=req.voice,
        )

        yield WireMessage(
            id=request_id,
            type="event",
            event="tts.chunk",
            data=event_data.model_dump(),
            chunk=chunk_index,
            done=is_final,
        )
        chunk_index += 1
        sentence_index = current_sentence

    # Emit a final done sentinel if the last chunk wasn't marked is_final
    # (handles engines that don't set is_final on the last chunk)
    if chunk_index > 0:
        # Check if last emitted was already done — emit a terminal marker
        yield WireMessage(
            id=request_id,
            type="event",
            event="tts.chunk",
            data=TTSChunkEvent(
                audio_b64="",
                sample_rate=24000,
                num_channels=1,
                dtype="int16",
                chunk_index=chunk_index,
                sentence_index=sentence_index,
                is_final=True,
                samples=0,
            ).model_dump(),
            chunk=chunk_index,
            done=True,
        )


def main() -> None:
    """Entry point for ``python -m mod3.worker tts``."""
    run_loop(handle)


if __name__ == "__main__":
    main()
