"""schemas.ws_chat.inbound — Inbound WebSocket frame models for /ws/chat.

Inbound frames arrive as JSON text from the browser, except for raw PCM
audio which arrives as binary frames (no JSON envelope — see note below).

Binary inbound contract
-----------------------
When the browser Silero VAD detects speech, it sends raw Int16 PCM audio
at 16 kHz as one or more binary WebSocket frames directly (no JSON
wrapper). The server routes these to ``BrowserChannel._handle_audio()``.
These frames are NOT represented as a Pydantic model because they carry
no type discriminator; the WebSocket message router distinguishes them by
checking ``"bytes" in message`` before attempting JSON parse.

JSON inbound frame types
------------------------
* ``text_message``  — user typed text to submit
* ``end_of_speech`` — browser VAD finished collecting speech
* ``interrupt``     — user wants to cut off in-flight TTS
* ``config``        — session configuration update (voice, speed, model)
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class TextMessageFrame(_Base):
    """User submitted text chat (typed input, not voice)."""

    type: Literal["text_message"]
    text: str = ""


class EndOfSpeechFrame(_Base):
    """Browser VAD signaled end of utterance — trigger T3 STT."""

    type: Literal["end_of_speech"]


class InterruptFrame(_Base):
    """User requests immediate stop of in-flight TTS output."""

    type: Literal["interrupt"]


class ConfigFrame(_Base):
    """Session configuration update.

    Any subset of ``model``, ``voice``, ``speed`` may be present. Fields
    absent from the JSON payload are left unchanged on the server.
    """

    type: Literal["config"]
    model: str | None = Field(default=None)
    voice: str | None = Field(default=None)
    speed: float | None = Field(default=None)
    # Forward-compat: extra fields pass through (extra="allow")


# Discriminated union of all JSON inbound frame types.
InboundFrame = Annotated[
    Union[TextMessageFrame, EndOfSpeechFrame, InterruptFrame, ConfigFrame],
    Field(discriminator="type"),
]

__all__ = [
    "ConfigFrame",
    "EndOfSpeechFrame",
    "InboundFrame",
    "InterruptFrame",
    "TextMessageFrame",
]
