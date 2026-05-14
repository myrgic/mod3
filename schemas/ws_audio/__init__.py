"""schemas.ws_audio — WebSocket frame schemas for /ws/audio/{session_id}.

This channel provides per-session audio fan-out to dashboard clients and
any native listener that subscribes via WebSocket (replacing the local
``afplay`` / ``sounddevice`` player when a subscriber is present).

Wire contract — RTVI 1.3.0 (current)
--------------------------------------
Each audio delivery is a **single JSON text frame** in RTVI envelope shape:

  * ``bot-tts-started`` — emitted before the first audio chunk of an utterance.
  * One or more ``bot-tts-audio`` frames — each carries base64-encoded int16
    PCM in ``data.audio`` along with ``data.sample_rate`` and
    ``data.num_channels``.
  * ``bot-tts-stopped`` — emitted after the last chunk.

Wire contract — legacy (deprecated)
-------------------------------------
The old two-frame burst (JSON header + binary WAV) is retained for backward
compatibility via :class:`~.audio_header.AudioHeaderFrame`.  New clients
should use the RTVI envelope.

Client-side decode example (RTVI)::

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'bot-tts-started') { clearPlaybackQueue(); return; }
      if (msg.type === 'bot-tts-audio') {
        const pcm = base64ToInt16(msg.data.audio);
        playback.enqueue(pcm.buffer);
        return;
      }
      if (msg.type === 'bot-tts-stopped') { markPlaybackEnd(); return; }
    };

Server-side outbound (Python)::

    await ws.send_text(RtviBotTtsStarted(id=str(uuid4())).model_dump_json())
    await ws.send_text(RtviBotTtsAudio(
        id=str(uuid4()),
        data=BotTtsAudioData(audio=b64pcm, sample_rate=24000),
    ).model_dump_json())
    await ws.send_text(RtviBotTtsStopped(id=str(uuid4())).model_dump_json())
"""

from .audio_header import AudioHeaderFrame
from .rtvi_audio import (
    RTVI_LABEL,
    BotTtsAudioData,
    RtviAudioFrame,
    RtviBotTtsAudio,
    RtviBotTtsStarted,
    RtviBotTtsStopped,
)

__all__ = [
    "AudioHeaderFrame",
    "BotTtsAudioData",
    "RTVI_LABEL",
    "RtviAudioFrame",
    "RtviBotTtsAudio",
    "RtviBotTtsStarted",
    "RtviBotTtsStopped",
]
