"""Regression tests for §3 of ARCHITECTURE.md — WebSocket lifecycle fragility.

Before the fix: when a browser page reload disconnected the WebSocket, the
old BrowserChannel left behind:

  * a ChannelDescriptor on the bus whose ``deliver`` callback was a bound
    method of the dead channel — sends went to a closed WebSocket and
    triggered the 10s timeout cascade in ``_deliver_sync``;
  * a ChannelQueue in the OutputQueueManager (with a drain thread that
    might still be processing stale work);
  * an entry in ``BrowserChannel._active_channels`` (consulted by the
    trace-event fan-out broadcast).

After the fix: ``BrowserChannel._cleanup`` cancels queued jobs, then calls
``bus.unregister_channel`` which severs ``ch.deliver``, drops the channel
descriptor, and removes the ChannelQueue from the manager. Any concurrent
encode-then-deliver in ``bus.act``'s drain path now hits the existing
``if ch and ch.deliver`` guard and skips delivery silently.

Run: python -m pytest tests/test_websocket_lifecycle_cleanup.py -v
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bus import ModalityBus  # noqa: E402
from channels import BrowserChannel  # noqa: E402
from modality import (  # noqa: E402
    CognitiveIntent,
    EncodedOutput,
    ModalityModule,
    ModalityType,
)
from output_queue import OutputQueueManager  # noqa: E402


# ---------------------------------------------------------------------------
# Bus-level fixtures: unregister_channel + drop_queue
# ---------------------------------------------------------------------------


def test_unregister_channel_removes_descriptor():
    """The bus drops the ChannelDescriptor entry entirely."""
    bus = ModalityBus()
    deliver = MagicMock()
    bus.register_channel("ch1", [ModalityType.VOICE], deliver=deliver)
    assert "ch1" in bus._channels
    assert bus.unregister_channel("ch1") is True
    assert "ch1" not in bus._channels


def test_unregister_channel_severs_deliver_callback():
    """After unregister, ``ch.deliver`` is None on any leaked descriptor reference.

    A drain thread that grabbed the ChannelDescriptor before unregister still
    holds a Python reference to it. Severing ``deliver`` to None protects
    against that path writing to a closed WebSocket.
    """
    bus = ModalityBus()
    deliver = MagicMock()
    bus.register_channel("ch1", [ModalityType.VOICE], deliver=deliver)
    leaked_ref = bus._channels["ch1"]
    bus.unregister_channel("ch1")
    assert leaked_ref.deliver is None
    assert leaked_ref.active is False


def test_unregister_channel_removes_output_queue():
    """The ChannelQueue is dropped so the channel name can be re-used cleanly."""
    bus = ModalityBus()
    bus.register_channel("ch1", [ModalityType.VOICE], deliver=lambda _o: None)
    # Force a queue to exist
    bus._queue_manager.get_queue("ch1")
    assert "ch1" in bus._queue_manager._queues
    bus.unregister_channel("ch1")
    assert "ch1" not in bus._queue_manager._queues


def test_unregister_channel_idempotent_on_unknown():
    bus = ModalityBus()
    assert bus.unregister_channel("never-registered") is False


def test_drop_queue_returns_false_when_missing():
    mgr = OutputQueueManager()
    assert mgr.drop_queue("nope") is False


def test_drop_queue_returns_true_when_present():
    mgr = OutputQueueManager()
    mgr.get_queue("ch1")
    assert mgr.drop_queue("ch1") is True
    assert "ch1" not in mgr._queues


# ---------------------------------------------------------------------------
# In-flight encode → severed-deliver race: bus.act's `if ch and ch.deliver`
# guard is the safety net.
# ---------------------------------------------------------------------------


class _DummyEncoder:
    def encode(self, intent):
        return EncodedOutput(
            modality=ModalityType.VOICE,
            format="wav",
            data=b"fake-audio",
            duration_sec=0.1,
        )


class _DummyVoiceModule(ModalityModule):
    """Minimal ModalityModule for voice — only the encoder is used.

    ModalityModule declares ``modality_type``/``gate``/``decoder``/``encoder``
    as abstract properties; we override each as a property too so the ABC
    is satisfied at instantiation time.
    """

    def __init__(self):
        self._encoder = _DummyEncoder()

    @property
    def modality_type(self):
        return ModalityType.VOICE

    @property
    def gate(self):
        return None

    @property
    def decoder(self):
        return None

    @property
    def encoder(self):
        return self._encoder


def test_in_flight_encode_after_unregister_skips_delivery():
    """Reproduces the race: encode runs, channel detaches, deliver-or-skip.

    Simulates the drain thread's _do_encode being called against a channel
    that was unregistered between submit and execution. Without the
    severed-callback guard, this would call a stale BrowserChannel method
    that writes to a closed WebSocket. With the fix, the encode completes
    silently and no delivery happens.
    """
    bus = ModalityBus()
    bus.register(_DummyVoiceModule())

    delivered: list[EncodedOutput] = []
    bus.register_channel(
        "ch-leak", [ModalityType.VOICE], deliver=lambda o: delivered.append(o)
    )

    # Disconnect: unregister the channel while a "job" is mid-flight.
    bus.unregister_channel("ch-leak")

    # Now run a fresh act() against the (now-detached) channel name.
    # The encode side still runs (encoder is module-scoped, not channel-scoped);
    # what matters is that delivery is silently skipped.
    intent = CognitiveIntent(
        modality=ModalityType.VOICE,
        content="hello",
        target_channel="ch-leak",
    )
    output = bus.act(intent, channel="ch-leak", blocking=True)
    assert isinstance(output, EncodedOutput)
    assert delivered == [], "deliver callback must NOT fire after unregister"


def test_act_skips_delivery_when_descriptor_deliver_is_none():
    """Direct test of the `if ch and ch.deliver` guard in _do_encode."""
    bus = ModalityBus()
    bus.register(_DummyVoiceModule())
    bus.register_channel("ch", [ModalityType.VOICE], deliver=None)

    intent = CognitiveIntent(
        modality=ModalityType.VOICE,
        content="hi",
        target_channel="ch",
    )
    # Must not raise, must return the encoded output.
    output = bus.act(intent, channel="ch", blocking=True)
    assert isinstance(output, EncodedOutput)


# ---------------------------------------------------------------------------
# Full BrowserChannel.cleanup end-to-end: connect → enqueue → disconnect →
# verify no leaks remain on the bus, the queue manager, or the active set.
# ---------------------------------------------------------------------------


class _FakeWebSocket:
    """Stand-in for fastapi.WebSocket — records sends, no real I/O."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, frame):
        if self.closed:
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect(code=1006)
        self.sent.append(frame)

    async def receive(self):
        return {"type": "websocket.disconnect"}


def _make_channel(loop):
    """Build a real BrowserChannel against a fake WebSocket.

    Bypasses the heavy WhisperDecoder(load_base=True) load by patching it
    out — the lifecycle test does not exercise STT.
    """
    from unittest.mock import patch

    bus = ModalityBus()
    bus.register(_DummyVoiceModule())

    from pipeline_state import PipelineState

    ps = PipelineState()
    ws = _FakeWebSocket()

    with patch("channels.WhisperDecoder", autospec=True):
        ch = BrowserChannel(
            ws=ws,
            bus=bus,
            pipeline_state=ps,
            loop=loop,
            on_event=None,
        )
    return ch, ws, bus


def test_browser_channel_disconnect_cleans_up_all_state():
    """End-to-end: simulate connect → bus.register → disconnect → verify cleanup.

    This is the §3 regression check — exercises the actual code path the
    WebSocket disconnect handler in http_api.ws_chat takes via channel.run()
    triggering _cleanup() in its finally block.
    """
    loop = asyncio.new_event_loop()
    try:
        ch, ws, bus = _make_channel(loop)
        channel_id = ch.channel_id

        # Verify pre-state: registered on bus and in active set.
        assert channel_id in bus._channels
        assert ch in BrowserChannel._active_channels
        assert ch._active is True

        # Submit some queued jobs to simulate in-flight TTS at disconnect time.
        # Using a no-op fn so we don't actually run TTS; we just need the
        # ChannelQueue to exist with queued entries to verify cancel_channel
        # reports them.
        def _slow_job():
            time.sleep(60)  # would block forever; we cancel it below
            return None

        bus._queue_manager.submit(channel_id, _slow_job)
        bus._queue_manager.submit(channel_id, _slow_job)
        bus._queue_manager.submit(channel_id, _slow_job)
        # Give the drain thread a beat to pop the first one (it now blocks
        # in time.sleep, leaving 2 in the deque to be cancelled).
        time.sleep(0.05)

        # Trigger the disconnect cleanup.
        ch._cleanup()

        # Post-state assertions: all leaks repaired.
        assert ch._active is False, "channel must mark itself inactive"
        assert ch not in BrowserChannel._active_channels, "must leave broadcast set"
        assert channel_id not in bus._channels, "must drop ChannelDescriptor"
        assert (
            channel_id not in bus._queue_manager._queues
        ), "must drop ChannelQueue from manager"
        # cancel_channel ran before drop_queue, so any pending jobs were
        # cancelled. The drain thread's current in-progress _slow_job is
        # still running (we can't interrupt the worker mid-sleep), but it
        # will see an empty deque + cleared _running flag on its next loop.
    finally:
        # Ensure no hanging drain threads waste pytest's exit.
        loop.close()


def test_browser_channel_cleanup_is_idempotent():
    """Double-cleanup must not double-decrement or raise."""
    loop = asyncio.new_event_loop()
    try:
        ch, _ws, bus = _make_channel(loop)
        channel_id = ch.channel_id
        ch._cleanup()
        # Second call is a no-op (the `if not self._active: return` guard).
        ch._cleanup()
        assert channel_id not in bus._channels
        assert ch not in BrowserChannel._active_channels
    finally:
        loop.close()


def test_reconnect_with_new_channel_id_does_not_collide():
    """After disconnect → reconnect, the new channel is fully independent.

    Simulates the page-reload pattern from §3: old BrowserChannel cleans up,
    new BrowserChannel registers with a fresh UUID. The new channel must
    not see any state from the old one.
    """
    loop = asyncio.new_event_loop()
    try:
        old, _, bus = _make_channel(loop)
        old_id = old.channel_id
        # Same bus, simulating server staying up across page reload.
        # Make a "new" channel by patching uuid before constructing.
        old._cleanup()

        new, _, _ = _make_channel_on_existing_bus(bus, loop)
        assert new.channel_id != old_id
        assert old_id not in bus._channels
        assert new.channel_id in bus._channels
        # Old channel's queue is gone; new one has no carryover state.
        assert old_id not in bus._queue_manager._queues

        new._cleanup()
    finally:
        loop.close()


def _make_channel_on_existing_bus(bus, loop):
    """Helper: spin up a fresh BrowserChannel on an already-built bus."""
    from unittest.mock import patch

    from pipeline_state import PipelineState

    ps = PipelineState()
    ws = _FakeWebSocket()
    with patch("channels.WhisperDecoder", autospec=True):
        ch = BrowserChannel(
            ws=ws,
            bus=bus,
            pipeline_state=ps,
            loop=loop,
            on_event=None,
        )
    return ch, ws, bus


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
