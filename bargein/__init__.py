"""Barge-in subsystem.

This package owns the first-class barge-in primitive inside mod3. Sources
(SuperWhisper, browser VAD, MCP signals, etc.) register as
``BargeinProvider`` instances; each one emits ``BargeinEvent``s through a
callback. The registry below wires those callbacks into the shared consumer
helper ``handle_bargein_event``, which does the same work the legacy
``/tmp/mod3-barge-in.json`` file watcher in ``server.py`` does today:
interrupt in-progress playback via ``pipeline_state.interrupt()`` and log.

Env-driven config:
  MOD3_BARGEIN_PROVIDERS — comma-separated provider names (default: empty).
                           Example: ``MOD3_BARGEIN_PROVIDERS=superwhisper``

Default is empty so users without SuperWhisper installed see no behavior
change from the current setup — they can still run the standalone
``integrations/bargein-producer.py`` script and the legacy file watcher
in ``server.py`` keeps picking up its signals.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Callable

from pipeline_state import InterruptInfo, PipelineState
from schemas.bargein import BargeinSource

from .providers.base import BargeinCallback, BargeinEvent, BargeinEventType, BargeinProvider

log = logging.getLogger("bargein")

# ---------------------------------------------------------------------------
# Shared consumer helper
# ---------------------------------------------------------------------------
#
# Both the legacy file watcher in server.py and the new provider registry
# call this when a "user is speaking" signal arrives. It is the single
# authoritative "barge-in start" handler.
#
# Returning the InterruptInfo (or None) lets the file watcher continue its
# extra work of writing the interrupt detail back into the signal file —
# cross-process coordination that only matters for the file-based IPC.
# In-process providers ignore the return.


def handle_bargein_start(
    pipeline_state: PipelineState,
    source: str,
    metadata: dict | None = None,
    session_id: str = "",
    message_id: str = "",
) -> InterruptInfo | None:
    """Attempt to interrupt in-progress TTS playback because the user began speaking.

    Returns the ``InterruptInfo`` if playback was actually halted, or ``None``
    if nothing was speaking (or another process owns the speech — only the
    file watcher can handle that via the cross-process lock).

    When playback is halted, emits a ``bargein.event`` to the chat_flow_log
    capturing the structural split between solidified (actually played) and
    speculative (unsaid) content. This event feeds the trace panel's hatched
    bar and the speculative-output block in the kernel turn record.
    """
    if not pipeline_state.is_speaking:
        return None
    info = pipeline_state.interrupt(reason="barge_in")
    if info is not None:
        log.info(
            "Barge-in from %s: paused local playback (%.0f%% delivered)%s",
            source,
            info.spoken_pct * 100,
            f" meta={metadata}" if metadata else "",
        )
        # Emit bargein.event to chat_flow_log for observability and
        # speculative-output block construction.
        try:
            from chat_flow_log import get_chat_flow_log

            speculative = ""
            if info.full_text and info.delivered_text:
                # Speculative = everything after the delivered prefix
                delivered = info.delivered_text
                if info.full_text.startswith(delivered):
                    speculative = info.full_text[len(delivered) :].strip()
                else:
                    # Fallback: use pct estimate
                    cut = int(info.spoken_pct * len(info.full_text))
                    speculative = info.full_text[cut:].strip()

            get_chat_flow_log().emit_bargein(
                session_id=session_id or (metadata or {}).get("session_id", ""),
                message_id=message_id or (metadata or {}).get("message_id", ""),
                original_text=info.full_text or "",
                text_actually_played=info.delivered_text or "",
                text_speculative=speculative,
                bargein_position_ms=info.bargein_position_ms,
                bargein_position_text_offset=info.bargein_position_text_offset,
            )
        except Exception as _e:  # noqa: BLE001
            log.debug("handle_bargein_start: failed to emit bargein.event: %s", _e)
    return info


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


PROVIDER_NAMES = ["superwhisper"]


def _build_provider(name: str, on_event: BargeinCallback) -> BargeinProvider | None:
    """Instantiate a provider by name. Returns None if unknown or import fails."""
    name = name.strip().lower()
    if not name:
        return None
    if name == "superwhisper":
        from .providers.superwhisper import SuperWhisperProvider

        return SuperWhisperProvider(on_event=on_event)
    log.warning("Unknown barge-in provider: %r (known: %s)", name, PROVIDER_NAMES)
    return None


class BargeinRegistry:
    """Owns the set of active barge-in providers and routes their events.

    Use:
        registry = BargeinRegistry(pipeline_state)
        registry.start_from_env()       # or registry.register(SomeProvider(...))
        # ... later, on shutdown:
        registry.stop_all()

    Tests can install their own dispatch by passing ``on_event`` to
    ``register``; registry-level dispatch goes through ``_dispatch`` which
    calls both ``handle_bargein_start`` and any extra subscribers.
    """

    def __init__(self, pipeline_state: PipelineState):
        self._pipeline_state = pipeline_state
        self._providers: list[BargeinProvider] = []
        self._subscribers: list[Callable[[BargeinEvent], None]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, provider: BargeinProvider) -> None:
        """Register a pre-built provider. Does NOT start it (see ``start_all``)."""
        with self._lock:
            self._providers.append(provider)

    def subscribe(self, callback: Callable[[BargeinEvent], None]) -> None:
        """Register an additional event subscriber (fires after the consumer helper).

        Useful for tests and for future observers (metrics, bus emits, etc.).
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[BargeinEvent], None]) -> None:
        """Remove a previously-registered subscriber. Idempotent."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Synchronous wait primitive
    # ------------------------------------------------------------------

    def wait_for_event(
        self,
        event_type: BargeinEventType,
        source: BargeinSource | None = None,
        timeout: float | None = None,
    ) -> BargeinEvent | None:
        """Block until a matching event is dispatched, or until ``timeout``.

        Returns the matching ``BargeinEvent`` on success, or ``None`` on timeout.
        Thread-safe; multiple waiters may run concurrently — each receives the
        first matching event emitted after its wait began.

        Example::

            event = registry.wait_for_event("user_speaking_end", timeout=180)
            if event is None:
                ...  # timed out
        """
        signal = threading.Event()
        captured: list[BargeinEvent] = []

        def _waiter(event: BargeinEvent) -> None:
            if event.event_type != event_type:
                return
            if source is not None and event.source != source:
                return
            if signal.is_set():
                return
            captured.append(event)
            signal.set()

        self.subscribe(_waiter)
        try:
            if signal.wait(timeout):
                return captured[0]
            return None
        finally:
            self.unsubscribe(_waiter)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        """Start every registered provider."""
        with self._lock:
            providers = list(self._providers)
        for p in providers:
            p.start()

    def stop_all(self, timeout: float = 2.0) -> None:
        """Signal shutdown and (best-effort) join every provider thread."""
        with self._lock:
            providers = list(self._providers)
        for p in providers:
            p.stop(timeout=timeout)

    def start_from_env(self, env_var: str = "MOD3_BARGEIN_PROVIDERS") -> list[str]:
        """Instantiate and start providers listed in the env var. Returns started names.

        Providers already present on the registry are kept; we append whatever
        the env var asks for that isn't already there.
        """
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            log.info("No barge-in providers configured (set %s=superwhisper to enable)", env_var)
            return []

        requested = [n.strip().lower() for n in raw.split(",") if n.strip()]
        already = {type(p).__name__.lower() for p in self._providers}
        started: list[str] = []
        for name in requested:
            # Match by normalized class name (SuperWhisperProvider -> "superwhisperprovider")
            # or the logical name the factory accepts.
            if f"{name}provider" in already:
                continue
            provider = _build_provider(name, self._dispatch)
            if provider is None:
                continue
            self.register(provider)
            provider.start()
            started.append(name)
        log.info("Barge-in providers started: %s", started)
        return started

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(self, event: BargeinEvent) -> None:
        """Route a provider event through the shared consumer + any subscribers."""
        try:
            if event.event_type == "user_speaking_start":
                handle_bargein_start(
                    self._pipeline_state,
                    source=event.source,
                    metadata=event.metadata,
                )
            # user_speaking_end has no in-process consumer today (the legacy
            # file watcher also only reacts to "start"). Subscribers still
            # see it so future code can use it.
        except Exception:
            log.exception("consumer helper raised while handling %s", event)

        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                log.exception("barge-in subscriber raised")


def make_file_mirror_subscriber(signal_path: str) -> Callable[[BargeinEvent], None]:
    """Build a registry subscriber that mirrors events into the legacy signal file.

    The legacy ``/tmp/mod3-barge-in.json`` file is consumed by
    out-of-process clients (e.g. ``mcp_shim.py``'s ``await_voice_input``)
    that cannot subscribe to the in-process registry. Installing this
    subscriber lets in-process providers reach those pollers.

    Writes are atomic (tmp + rename). ``OSError`` is swallowed and logged
    at debug level — the file mirror is best-effort and must never break
    in-process delivery.
    """

    def _mirror(event: BargeinEvent) -> None:
        try:
            payload = {
                "event": event.event_type,
                "source": event.source,
                "timestamp": event.timestamp.isoformat(),
                "via": "bargein_registry",
                **event.metadata,
            }
            tmp = signal_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, signal_path)
        except OSError:
            log.debug("file mirror write failed", exc_info=True)

    return _mirror


__all__ = [
    "BargeinEvent",
    "BargeinProvider",
    "BargeinRegistry",
    "handle_bargein_start",
    "make_file_mirror_subscriber",
    "PROVIDER_NAMES",
]
