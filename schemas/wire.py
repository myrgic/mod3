"""D2 wire protocol — JSON-lines envelope for Go-Python subprocess IPC.

Mirrors the canonical Go schema in ``cogos/pkg/modality/wire.go``. Every
field name and JSON tag must stay byte-identical to the Go side; this
module is the Python end of the same wire.

Transport: each ``WireMessage`` is serialised as a single line of JSON
(UTF-8, no embedded newlines) written to stdout (Python → Go) or stdin
(Go → Python). Lines must not exceed ``MAX_WIRE_LINE_SIZE`` (1 MB).

Message kinds (the ``type`` discriminator):

* ``request``   — Go asks Python to perform an operation (vad/detect,
                  stt/transcribe, tts/synthesize, ...)
* ``response``  — Python's reply to a request, carries ``result``
* ``error``     — Python's reply to a failed request
* ``command``   — Go sends a lifecycle command (``shutdown``, ``health``)
* ``event``     — Python emits an unsolicited event (``ready``, ``health``,
                  streaming progress, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_WIRE_LINE_SIZE = 1024 * 1024  # 1 MiB cap; matches Go scanner buffer.

WireType = Literal["request", "response", "error", "command", "event"]


class WireMessage(BaseModel):
    """A single message on the D2 wire.

    The shape is a flat union discriminated by ``type``. Only the fields
    relevant to the active variant should be populated. Empty fields are
    omitted from the JSON encoding via ``exclude_none``.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",  # forward-compat: unknown fields pass through
    )

    id: str = Field(..., description="request/response correlation ID")
    type: WireType
    ts: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        ),
        description="RFC3339Nano timestamp",
    )

    # ---- request fields ----
    module: str | None = Field(default=None, description="worker module: tts/vad/stt")
    op: str | None = Field(default=None, description="operation name within module")
    data: dict[str, Any] | None = None

    # ---- response fields ----
    result: dict[str, Any] | None = None

    # ---- streaming fields (reserved for chunked responses/events) ----
    chunk: int | None = Field(
        default=None, description="0-based chunk index within a streamed response"
    )
    done: bool | None = Field(
        default=None, description="terminal chunk marker for streamed responses"
    )

    # ---- command fields ----
    command: str | None = Field(default=None, description="lifecycle command verb")

    # ---- event fields ----
    event: str | None = Field(default=None, description="event name (e.g. 'ready')")
    status: str | None = Field(
        default=None, description="event status payload (e.g. 'ok')"
    )

    # ---- error fields ----
    error: str | None = None
    error_type: str | None = None
    recoverable: bool | None = None

    def to_jsonl(self) -> str:
        """Serialise to a single JSON line (no trailing newline)."""
        return self.model_dump_json(exclude_none=True, by_alias=True)
