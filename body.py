"""Body channel: lets an agent drive an animated body (e.g. a Live2D avatar).

A *body* is a client (usually a browser page) that renders something the agent
can move. It connects to ``/ws/body/{body_id}`` and declares what it can do in
a manifest: its parameters (with ranges) and its named states. The agent then
calls ``POST /v1/bodies/{body_id}/act`` and gets back a **receipt**, which is the
body's report of the state it actually reached, not an echo of the request.

Wire protocol (JSON text frames):

  body → server  {"type": "hello", "manifest": {"kind": "live2d",
                  "params": [{"id": "ParamAngleX", "min": -30, "max": 30,
                  "default": 0}, ...], "states": ["IDLE", ...], ...}}
  server → body  {"type": "welcome", "body_id": "..."}
  server → body  {"type": "act", "id": "<uuid>", "state": "...",
                  "params": {"ParamAngleX": 12.0}, "release": ["ParamAngleX"],
                  "hold_ms": 1500}
  body → server  {"type": "receipt", "id": "<uuid>", "state": "...",
                  "params": {"ParamAngleX": 11.7, ...}}
  server → body  {"type": "play", "id": "<uuid>", "clip": {<compiled clip>}}
  server → body  {"type": "stop", "id": "<uuid>", "clip": "<name>"?}

Clips (see clip.py) are agent-authored animations: data only, compiled and
checked here against the body's manifest, so the body never runs agent code.

The server validates commands against the manifest before forwarding: unknown
states are refused, unknown parameters are dropped and listed in
``rejected``, and values are clamped to the declared range. Lip sync is not
routed through here; a body that wants it subscribes to the speaking
session's ``/ws/audio/{session_id}`` channel and drives its own mouth.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mod3.body")
_CLOSING: set[asyncio.Task] = set()


class BodyError(Exception):
    """A command could not be delivered. ``status`` is the HTTP status to use."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class _Body:
    body_id: str
    manifest: dict[str, Any]
    send: Any  # async callable(dict) -> None, bound to ``loop``
    loop: asyncio.AbstractEventLoop | None = None
    # async callable() -> None that closes this body's connection; used when a
    # newer connection takes over the same body_id.
    close: Any = None
    connected_at: float = field(default_factory=time.time)
    # concurrent futures, so a receipt arriving on the socket's loop can wake
    # an HTTP request waiting on a different loop (or thread).
    pending: dict[str, concurrent.futures.Future] = field(default_factory=dict)
    last_receipt: dict[str, Any] | None = None

    @property
    def param_ranges(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for p in self.manifest.get("params", []) or []:
            try:
                out[str(p["id"])] = (float(p["min"]), float(p["max"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @property
    def channels(self) -> dict[str, Any]:
        """Semantic clip channels this body maps (see clip.CHANNELS)."""
        ch = self.manifest.get("semantic_channels") or {}
        return ch if isinstance(ch, dict) else {}

    @property
    def states(self) -> list[str]:
        return [str(s) for s in self.manifest.get("states", []) or []]

    def describe(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "kind": self.manifest.get("kind", "unknown"),
            "states": self.states,
            "params": self.manifest.get("params", []),
            "meta": {k: v for k, v in self.manifest.items() if k not in ("params", "states")},
            "connected_at": self.connected_at,
            "last_receipt": self.last_receipt,
        }


def validate_command(body: _Body, command: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Check a command against the body's manifest.

    Returns the cleaned command and the list of rejected parameter ids.
    Raises BodyError(400) for an unknown state.
    """
    out: dict[str, Any] = {}
    rejected: list[str] = []
    state = command.get("state")
    if state is not None:
        if state not in body.states:
            raise BodyError(400, f"body '{body.body_id}' has no state '{state}' (has: {', '.join(body.states)})")
        out["state"] = state
    ranges = body.param_ranges
    params = command.get("params") or {}
    clean: dict[str, float] = {}
    for pid, value in params.items():
        if pid not in ranges:
            rejected.append(pid)
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            rejected.append(pid)
            continue
        if not math.isfinite(v):
            # NaN slips through min/max (every comparison is False) and inf is
            # not a pose; reject both rather than forward them to the body.
            rejected.append(pid)
            continue
        lo, hi = ranges[pid]
        clean[pid] = min(max(v, lo), hi)
    if clean:
        out["params"] = clean
    release = []
    for p in command.get("release") or []:
        if p in ranges:
            release.append(p)
        elif p not in rejected:
            rejected.append(p)
    if release:
        out["release"] = release
    if command.get("hold_ms") is not None:
        out["hold_ms"] = max(0, int(command["hold_ms"]))
    return out, rejected


class BodyRegistry:
    """body_id → connected body.

    Touched from two kinds of thread: the event loop (ws_body registers and
    unregisters; async routes act) and Starlette's thread pool (the sync GET
    routes call list()/get()). Every access to ``_bodies`` and to a body's
    ``pending`` map goes through ``_lock``; iteration works on snapshots taken
    under it. Futures are completed outside the lock.
    """

    def __init__(self) -> None:
        self._bodies: dict[str, _Body] = {}
        self._lock = threading.Lock()

    def register(
        self,
        body_id: str,
        manifest: dict[str, Any],
        send: Any,
        loop: asyncio.AbstractEventLoop | None = None,
        close: Any = None,
    ) -> _Body:
        body = _Body(body_id=body_id, manifest=manifest, send=send, loop=loop, close=close)
        with self._lock:
            old = self._bodies.get(body_id)
            self._bodies[body_id] = body
            waiters = list(old.pending.values()) if old is not None else []
            if old is not None:
                old.pending.clear()
        if old is not None:
            # A newer connection takes over this body_id: fail the old one's
            # waiters and close its socket (code 4409), so its server task ends
            # instead of idling forever. Clients must not auto-reconnect on 4409,
            # or two pages with the same id would evict each other in a loop.
            for fut in waiters:
                if not fut.done():
                    fut.set_exception(BodyError(409, "body reconnected"))
            self._close(old)
        logger.info("body attached: %s kind=%s params=%d", body_id, manifest.get("kind"), len(body.param_ranges))
        return body

    @staticmethod
    def _close(body: _Body) -> None:
        if body.close is None:
            return
        try:
            here = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        try:
            if body.loop is not None and body.loop is not here:
                asyncio.run_coroutine_threadsafe(body.close(), body.loop)
            elif here is not None:
                task = here.create_task(body.close())
                _CLOSING.add(task)  # asyncio holds tasks weakly; keep it alive
                task.add_done_callback(_CLOSING.discard)
        except Exception:  # noqa: BLE001 — a dead socket is already closed
            logger.debug("closing replaced body %s failed", body.body_id, exc_info=True)

    def unregister(self, body: _Body) -> None:
        with self._lock:
            if self._bodies.get(body.body_id) is body:
                del self._bodies[body.body_id]
            waiters = list(body.pending.values())
            body.pending.clear()
        for fut in waiters:
            if not fut.done():
                fut.set_exception(BodyError(410, "body disconnected"))
        logger.info("body detached: %s", body.body_id)

    def get(self, body_id: str) -> _Body | None:
        with self._lock:
            return self._bodies.get(body_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            bodies = list(self._bodies.values())
        return [b.describe() for b in bodies]

    def on_receipt(self, body: _Body, msg: dict[str, Any]) -> None:
        receipt = {k: v for k, v in msg.items() if k != "type"}
        body.last_receipt = receipt
        with self._lock:
            fut = body.pending.pop(str(msg.get("id")), None)
        if fut is not None and not fut.done():
            fut.set_result(receipt)

    async def act(self, body_id: str, command: dict[str, Any], timeout: float = 2.0) -> dict[str, Any]:
        body = self.get(body_id)
        if body is None:
            raise BodyError(404, f"no body '{body_id}' is connected")
        clean, rejected = validate_command(body, command)
        return await self._send_and_wait(body, "act", clean, rejected, timeout)

    async def play(
        self, body_id: str, clip: dict[str, Any], args: dict[str, Any] | None = None, timeout: float = 2.0
    ) -> dict[str, Any]:
        """Compile a clip for this body and start it. Receipt = first frames reached."""
        from clip import ClipError, compile_clip

        body = self.get(body_id)
        if body is None:
            raise BodyError(404, f"no body '{body_id}' is connected")
        try:
            compiled, rejected = compile_clip(
                clip,
                body_channels=body.channels,
                body_params=body.param_ranges,
                body_states=body.states,
                args=args,
            )
        except ClipError as exc:
            raise BodyError(400, str(exc)) from exc
        return await self._send_and_wait(body, "play", {"clip": compiled}, rejected, timeout)

    async def stop(self, body_id: str, clip: str | None = None, timeout: float = 2.0) -> dict[str, Any]:
        body = self.get(body_id)
        if body is None:
            raise BodyError(404, f"no body '{body_id}' is connected")
        return await self._send_and_wait(body, "stop", {"clip": clip} if clip else {}, [], timeout)

    async def _send_and_wait(
        self, body: _Body, kind: str, clean: dict[str, Any], rejected: list[str], timeout: float
    ) -> dict[str, Any]:
        body_id = body.body_id
        cid = str(uuid.uuid4())
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            body.pending[cid] = fut
        t0 = time.perf_counter()
        frame = {"type": kind, "id": cid, **clean}
        try:
            here = asyncio.get_running_loop()
            if body.loop is None or body.loop is here:
                await body.send(frame)
            else:
                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(body.send(frame), body.loop))
            receipt = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise BodyError(504, f"body '{body_id}' sent no receipt within {timeout}s") from exc
        finally:
            with self._lock:
                body.pending.pop(cid, None)
        return {
            "body_id": body_id,
            "command": clean,
            "rejected": rejected,
            "receipt": receipt,
            "round_trip_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


_default: BodyRegistry | None = None


def get_default_body_registry() -> BodyRegistry:
    global _default
    if _default is None:
        _default = BodyRegistry()
    return _default
