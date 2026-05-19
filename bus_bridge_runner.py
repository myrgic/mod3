"""Kernel-bus → dashboard bridge runner.

Consumes `KernelBusSubscriber.stream()` (see `bus_bridge.py`) and fans the
ADR-083 cycle-trace events out to every connected dashboard WebSocket via
`BrowserChannel.broadcast_trace_event()` (see `channels.py`).

Also handles identity-projection events (identity.projected,
identity.expression.updated) via ``handle_identity_event`` from
``identity_projection_handler``. These update the module-level
``IDENTITY_VOICE_CACHE`` so the TTS path can look up pre-resolved voice
conditionals by identity subject slug.

Wiring:

  kernel (bus_cycle_trace)
     └─► SSE /v1/events/stream?bus_id=bus_cycle_trace
            └─► KernelBusSubscriber.stream()       [C1]
                   └─► run_bridge() filter + forward
                          └─► BrowserChannel.broadcast_trace_event()  [C2]
                          └─► handle_identity_event() for IDENTITY_KINDS  [C3]

The subscriber does its own reconnect with exponential backoff, so a kernel
that is temporarily unreachable does not affect server startup. Disable the
bridge entirely at process boot by setting env `MOD3_BUS_BRIDGE_DISABLED=1`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from bus_bridge import KernelBusSubscriber, default_stream_url
from channels import BrowserChannel
from identity_projection_handler import (
    IDENTITY_KINDS,
    IdentityVoiceCache,
    handle_identity_event,
)

logger = logging.getLogger("mod3.bus_bridge")

# Module-level identity voice cache. Populated by the SSE bridge loop as
# identity.projected / identity.expression.updated events arrive. The TTS
# path (engine.resolve_model) can consult this for pre-resolved conditionals.
# Exported so tests and the TTS path can import it directly.
IDENTITY_VOICE_CACHE: IdentityVoiceCache = IdentityVoiceCache()

# ADR-083 kinds the dashboard trace panel cares about. Kept as a module-level
# constant so tests and the lifespan wiring share one definition.
ADR083_KINDS: frozenset[str] = frozenset({"state_transition", "tool_dispatch", "assessment"})

# Kernel-side bus name (see apps/cogos/trace_emit.go:const traceBusID).
TRACE_BUS_ID = "bus_cycle_trace"

# Env flag consulted at startup.
DISABLE_ENV = "MOD3_BUS_BRIDGE_DISABLED"


def is_disabled() -> bool:
    """True when MOD3_BUS_BRIDGE_DISABLED is set to a truthy value."""
    v = os.environ.get(DISABLE_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


async def run_bridge(
    subscriber: KernelBusSubscriber,
    *,
    filter_kinds: Optional[set[str]] = None,
    identity_cache: Optional[IdentityVoiceCache] = None,
) -> None:
    """Consume `subscriber` and broadcast cycle-trace events to dashboard clients.

    Also dispatches identity-projection events to ``handle_identity_event``
    regardless of `filter_kinds`, since identity state updates are side-effects
    on the voice cache, not dashboard broadcasts.

    `filter_kinds`:
      - `None`: forward everything to the dashboard (dev mode).
      - set of kind strings: only forward envelopes whose `BusEnvelope.kind`
        is in the set to the dashboard. Identity-projection events are always
        handled independently of this filter.

    `identity_cache`: the IdentityVoiceCache to update on identity events.
    Defaults to the module-level ``IDENTITY_VOICE_CACHE`` when None.

    `BrowserChannel.broadcast_trace_event()` is thread-safe and non-blocking:
    it dispatches each WS send via `run_coroutine_threadsafe`. We call it
    directly (no await).
    """
    cache = identity_cache if identity_cache is not None else IDENTITY_VOICE_CACHE
    first_event_logged = False
    forwarded = 0
    async for env in subscriber.stream():
        # Identity-projection events are handled unconditionally — they update
        # the voice cache as a side effect, independent of dashboard filtering.
        if env.kind in IDENTITY_KINDS:
            try:
                handle_identity_event(env.payload, cache)
            except Exception as exc:  # noqa: BLE001 — handler errors must not crash the loop
                logger.warning(
                    "bridge: identity event handler raised unexpectedly kind=%s: %s",
                    env.kind,
                    exc,
                )

        if filter_kinds is not None and env.kind not in filter_kinds:
            continue
        # The "connected" bootstrap frame has an empty payload; skip silently.
        if env.kind == "connected":
            continue
        if not first_event_logged:
            logger.info(
                "bridge: first event forwarded kind=%s event_id=%s",
                env.kind,
                env.event_id,
            )
            first_event_logged = True
        try:
            BrowserChannel.broadcast_trace_event(env.payload)
            forwarded += 1
            logger.debug(
                "bridge: forwarded kind=%s event_id=%s (total=%d)",
                env.kind,
                env.event_id,
                forwarded,
            )
        except Exception as exc:  # noqa: BLE001 — broadcaster is best-effort
            logger.debug("bridge: broadcast failed: %s", exc)


async def start_bridge(
    app_state: object,
    *,
    url: Optional[str] = None,
    bus_filter: str = TRACE_BUS_ID,
    filter_kinds: Optional[set[str]] = frozenset(ADR083_KINDS),
) -> None:
    """Construct the subscriber + bridge task and store them on `app_state`.

    Startup is non-blocking: we don't await the task or probe the kernel.
    The subscriber's own backoff loop handles reconnects. Logs a disabled
    notice and returns cleanly when `MOD3_BUS_BRIDGE_DISABLED` is set.

    ``url`` defaults to ``COGOS_ENDPOINT`` (resolved at call time) so the
    subscriber tracks whatever endpoint the rest of the cogos client code is
    using.
    """
    if is_disabled():
        logger.info("bridge: disabled via %s=1", DISABLE_ENV)
        setattr(app_state, "bus_bridge_subscriber", None)
        setattr(app_state, "bus_bridge_task", None)
        return

    resolved_url = url or default_stream_url()
    subscriber = KernelBusSubscriber(url=resolved_url, bus_filter=bus_filter, consumer_id="mod3-dashboard")
    task = asyncio.create_task(
        run_bridge(subscriber, filter_kinds=set(filter_kinds) if filter_kinds else None),
        name="mod3-bus-bridge",
    )
    setattr(app_state, "bus_bridge_subscriber", subscriber)
    setattr(app_state, "bus_bridge_task", task)
    logger.info(
        "bridge: started, target=%s bus_id=%s filter=%s",
        resolved_url,
        bus_filter,
        sorted(filter_kinds) if filter_kinds else "*",
    )


async def stop_bridge(app_state: object, *, timeout_s: float = 2.0) -> None:
    """Gracefully stop the bridge: close subscriber, await task, cancel on timeout."""
    subscriber: Optional[KernelBusSubscriber] = getattr(app_state, "bus_bridge_subscriber", None)
    task: Optional[asyncio.Task] = getattr(app_state, "bus_bridge_task", None)
    if subscriber is None and task is None:
        return
    if subscriber is not None:
        try:
            await subscriber.close()
        except Exception:  # pragma: no cover - best-effort
            pass
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pragma: no cover
                pass
    logger.info("bridge: stopped")
