"""Wire-loop dispatcher — shared stdin/stdout JSON-lines event loop.

Every worker subcommand (tts, vad, stt) calls ``run_loop(handler)`` with a
callable that accepts a ``WireMessage`` request and returns either a single
``WireMessage`` response or an iterable of ``WireMessage`` records (for
streaming ops). The dispatcher handles startup, health, shutdown, error
wrapping, and the read/write loop.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Callable

from schemas.wire import WireMessage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _emit(msg: WireMessage) -> None:
    """Write a single WireMessage as a JSON line to stdout."""
    line = msg.to_jsonl()
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _emit_ready() -> None:
    _emit(WireMessage(id="__startup__", type="event", event="ready", status="ok"))


def _emit_health(request_id: str) -> None:
    _emit(
        WireMessage(
            id=request_id,
            type="event",
            event="health",
            status="ok",
            ts=_now(),
        )
    )


def _emit_error(request_id: str, message: str, error_type: str = "InternalError", recoverable: bool = True) -> None:
    _emit(
        WireMessage(
            id=request_id,
            type="error",
            error=message,
            error_type=error_type,
            recoverable=recoverable,
        )
    )


def run_loop(handler: Callable[[WireMessage], WireMessage | Iterable[WireMessage]]) -> None:
    """Main event loop for a worker subprocess.

    Reads JSON-lines from stdin, dispatches to ``handler``, writes responses
    to stdout. Handles lifecycle commands (health, shutdown) before delegating
    to the handler. Exits cleanly on shutdown command or stdin EOF.

    Args:
        handler: Callable that accepts a request ``WireMessage`` and returns
            either a single ``WireMessage`` (one-shot ops) or an iterable of
            ``WireMessage`` records (streaming ops). May raise; exceptions are
            caught and written as wire errors.
    """
    _emit_ready()

    for raw_line in sys.stdin:
        raw_line = raw_line.rstrip("\n")
        if not raw_line:
            continue

        # Parse envelope
        try:
            data = json.loads(raw_line)
            msg = WireMessage.model_validate(data)
        except Exception as exc:
            # Malformed input: emit a structured error with id "__parse_error__"
            _emit_error(
                request_id="__parse_error__",
                message=f"malformed wire message: {exc}",
                error_type="ParseError",
                recoverable=True,
            )
            continue

        # Lifecycle commands
        if msg.type == "command":
            if msg.command == "shutdown":
                sys.exit(0)
            elif msg.command == "health":
                _emit_health(msg.id)
            else:
                _emit_error(
                    msg.id,
                    f"unknown command: {msg.command!r}",
                    error_type="UnknownCommand",
                    recoverable=True,
                )
            continue

        # Dispatch to handler
        if msg.type == "request":
            try:
                result = handler(msg)
                if isinstance(result, WireMessage):
                    _emit(result)
                else:
                    for chunk_msg in result:
                        _emit(chunk_msg)
            except Exception as exc:  # noqa: BLE001
                _emit_error(
                    msg.id,
                    f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                    recoverable=True,
                )
                # Write traceback to stderr for debugging
                traceback.print_exc(file=sys.stderr)
            continue

        # Ignore events from the Go side (forward-compat)
