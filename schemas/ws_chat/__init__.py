"""schemas.ws_chat — WebSocket frame schemas for the /ws/chat endpoint.

Binary inbound contract
-----------------------
The browser sends raw Int16 PCM audio at 16 kHz as binary WebSocket
frames during speech capture (from the Silero VAD onFrameProcessed
callback). These frames have NO JSON envelope and no type discriminator.
The server identifies them by the presence of ``bytes`` in the raw
WebSocket message dict before attempting JSON decode. They are not
modelled as Pydantic types here for that reason.

JSON frame types
----------------
Inbound (browser -> server):
  * ``text_message``  — typed text submission
  * ``end_of_speech`` — browser VAD end-of-utterance signal
  * ``interrupt``     — stop in-flight TTS
  * ``config``        — session config update (voice/speed/model)

Outbound (server -> browser):
  * ``audio``              — base64-encoded WAV for playback
  * ``response_text``      — LLM response text
  * ``response_complete``  — turn complete with metrics
  * ``transcript``         — final T3 / text-input transcript
  * ``partial_transcript`` — rolling T1/T2 streaming preview
  * ``interrupted``        — TTS barge-in acknowledgement
  * ``draft_queue``        — draft queue state update
  * ``trace_event``        — kernel cycle-trace events (ADR-083)
  * ``error``              — structured handler error (JSON-RPC shape)

Usage
-----
Construct outbound frames with the typed model and serialise via
``.model_dump(exclude_none=True)`` before passing to ``ws.send_json()``.
Validate inbound JSON frames via ``InboundFrame`` (discriminated union
on the ``type`` field).
"""

from .inbound import (
    ConfigFrame,
    EndOfSpeechFrame,
    InboundFrame,
    InterruptFrame,
    TextMessageFrame,
)
from .outbound import (
    AudioFrame,
    DraftQueueFrame,
    InterruptedFrame,
    OutboundFrame,
    PartialTranscriptFrame,
    ResponseCompleteFrame,
    ResponseTextFrame,
    TraceEventFrame,
    TranscriptFrame,
    WsErrorDetail,
    WsErrorFrame,
)

# Convenience alias — the complete Frame union (inbound + outbound share the
# same wire; in practice callers pick the appropriate sub-union).
Frame = OutboundFrame

__all__ = [
    # inbound
    "ConfigFrame",
    "EndOfSpeechFrame",
    "InboundFrame",
    "InterruptFrame",
    "TextMessageFrame",
    # outbound
    "AudioFrame",
    "DraftQueueFrame",
    "Frame",
    "InterruptedFrame",
    "OutboundFrame",
    "PartialTranscriptFrame",
    "ResponseCompleteFrame",
    "ResponseTextFrame",
    "TraceEventFrame",
    "TranscriptFrame",
    "WsErrorDetail",
    "WsErrorFrame",
]
