"""schemas.ws_audio — WebSocket frame schemas for /ws/audio/{session_id}.

This channel provides per-session audio fan-out to dashboard clients and
any native listener that subscribes via WebSocket (replacing the local
``afplay`` / ``sounddevice`` player when a subscriber is present).

Wire contract
-------------
Each audio delivery is a two-frame burst:

  1. **JSON text frame** — an :class:`~.audio_header.AudioHeaderFrame`
     describing the incoming audio (session, job, duration, sample rate,
     byte length, format, sequence number).
  2. **Binary frame** — the raw WAV bytes, exactly ``header.bytes`` long.

The binary frame carries no additional envelope; the preceding header is
the only metadata. The client MUST consume both frames before rendering.

Client-side decode example::

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        this._pendingHeader = JSON.parse(event.data);  // AudioHeaderFrame
      } else {
        const wav = event.data;  // ArrayBuffer
        audioCtx.decodeAudioData(wav).then(buf => play(buf));
        this._pendingHeader = null;
      }
    };

Server-side outbound (Python)::

    header = AudioHeaderFrame(
        session_id=session_id,
        job_id=job_id,
        duration_sec=duration,
        sample_rate=sample_rate,
        bytes=len(wav_bytes),
        seq=seq,
    )
    await ws.send_text(header.model_dump_json(exclude_none=True))
    await ws.send_bytes(wav_bytes)
"""

from .audio_header import AudioHeaderFrame

__all__ = ["AudioHeaderFrame"]
