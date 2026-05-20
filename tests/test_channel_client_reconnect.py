"""Regression tests for channel_client's SSE reconnect-on-disconnect loop.

Before this fix, ``ChannelClient.run_sse_subscription`` was a one-shot — on
any disconnect (mod3 restart, network blip, server-side stream close) the
coroutine returned silently and the seat was lost until the MCP child was
respawned. Symptom: every mod3 restart made attached Claude Code sessions
invisible to the dashboard sidebar.

These tests exercise the reconnect loop without spinning up an HTTP server:
``_stream_seat_events_once`` is stubbed to simulate disconnect-then-success
and disconnect-then-cancel sequences. The loop's responsibilities under test:

- Stream end (clean ``return``) triggers re-register + resubscribe.
- ``httpx.RemoteProtocolError`` triggers re-register + resubscribe.
- ``httpx.HTTPStatusError`` (e.g. mod3 doesn't know our seat after restart)
  triggers re-register + resubscribe.
- ``asyncio.CancelledError`` exits the loop cleanly (does NOT re-register).
- ``_reregister_seat`` failure does NOT crash the loop; it backs off and
  tries again on the next iteration.

The reconnect path uses ``self.seat_id`` overwriting — successive iterations
must pick up the new seat_id returned from ``_reregister_seat``.

Run with:
    PYTHONPATH=. .venv/bin/python -m pytest tests/test_channel_client_reconnect.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    """Build a ChannelClient with the token / device-uuid paths in tmp_path."""
    # Patch the token + device-uuid storage to point at a writable tmpdir so
    # the test doesn't touch the user's ~/.mod3 directory.
    import clients.channel_client as cc

    monkeypatch.setattr(cc, "_TOKEN_PATH", tmp_path / "channel-client-token")

    def _build(seat_id: str = "seat-original"):
        client = cc.ChannelClient("http://localhost:7860", "session-test")
        client.seat_id = seat_id
        return client

    return _build


class TestSseReconnect:
    @pytest.mark.asyncio
    async def test_clean_stream_end_triggers_reregister(self, make_client, monkeypatch):
        client = make_client(seat_id="seat-1")

        # First pass returns cleanly (stream end). Second pass: cancel out.
        calls = {"stream": 0, "reregister": 0, "sleep": 0}

        async def fake_stream():
            calls["stream"] += 1
            if calls["stream"] == 1:
                return  # clean end → triggers reconnect
            raise asyncio.CancelledError()

        async def fake_reregister():
            calls["reregister"] += 1
            client.seat_id = f"seat-after-{calls['reregister']}"
            return True

        async def fake_sleep(_):
            calls["sleep"] += 1

        client._stream_seat_events_once = fake_stream
        client._reregister_seat = fake_reregister
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await client.run_sse_subscription()

        assert calls["stream"] == 2
        assert calls["reregister"] == 1
        assert client.seat_id == "seat-after-1"

    @pytest.mark.asyncio
    async def test_remote_protocol_error_triggers_reregister(self, make_client, monkeypatch):
        client = make_client(seat_id="seat-1")
        calls = {"stream": 0, "reregister": 0}

        async def fake_stream():
            calls["stream"] += 1
            if calls["stream"] == 1:
                raise httpx.RemoteProtocolError("connection dropped")
            raise asyncio.CancelledError()

        async def fake_reregister():
            calls["reregister"] += 1
            client.seat_id = "seat-fresh"
            return True

        async def fake_sleep(_):
            pass

        client._stream_seat_events_once = fake_stream
        client._reregister_seat = fake_reregister
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await client.run_sse_subscription()

        assert calls["reregister"] == 1
        assert client.seat_id == "seat-fresh"

    @pytest.mark.asyncio
    async def test_http_status_error_triggers_reregister(self, make_client, monkeypatch):
        """A 404 from mod3 (it doesn't know our seat anymore) must reconnect."""
        client = make_client(seat_id="seat-1")
        calls = {"stream": 0, "reregister": 0}

        async def fake_stream():
            calls["stream"] += 1
            if calls["stream"] == 1:
                response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
                raise httpx.HTTPStatusError("seat not found", request=response.request, response=response)
            raise asyncio.CancelledError()

        async def fake_reregister():
            calls["reregister"] += 1
            client.seat_id = "seat-new"
            return True

        async def fake_sleep(_):
            pass

        client._stream_seat_events_once = fake_stream
        client._reregister_seat = fake_reregister
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await client.run_sse_subscription()

        assert calls["reregister"] == 1

    @pytest.mark.asyncio
    async def test_cancel_exits_without_reregister(self, make_client, monkeypatch):
        """A cancel from outside the loop must NOT trigger another re-register."""
        client = make_client(seat_id="seat-1")
        calls = {"stream": 0, "reregister": 0}

        async def fake_stream():
            calls["stream"] += 1
            raise asyncio.CancelledError()

        async def fake_reregister():
            calls["reregister"] += 1
            return True

        async def fake_sleep(_):
            pass

        client._stream_seat_events_once = fake_stream
        client._reregister_seat = fake_reregister
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await client.run_sse_subscription()

        assert calls["stream"] == 1
        assert calls["reregister"] == 0  # cancellation is not a reconnect trigger

    @pytest.mark.asyncio
    async def test_reregister_failure_does_not_crash_loop(self, make_client, monkeypatch):
        """When mod3 is still down, _reregister_seat returns False — the loop
        must keep iterating with backoff until either mod3 returns or the
        task is cancelled."""
        client = make_client(seat_id="seat-1")
        calls = {"stream": 0, "reregister": 0, "sleep": 0}

        async def fake_stream():
            calls["stream"] += 1
            if calls["stream"] == 1:
                raise httpx.ConnectError("connection refused")
            raise asyncio.CancelledError()

        reregister_results = [False, False, True]

        async def fake_reregister():
            calls["reregister"] += 1
            return reregister_results[calls["reregister"] - 1]

        async def fake_sleep(seconds):
            calls["sleep"] += 1

        client._stream_seat_events_once = fake_stream
        client._reregister_seat = fake_reregister
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await client.run_sse_subscription()

        # 3 register attempts: two fails + one success
        assert calls["reregister"] == 3
        # 3 sleeps gating the retries
        assert calls["sleep"] == 3

    @pytest.mark.asyncio
    async def test_no_seat_id_returns_immediately(self, make_client):
        """Without a seat_id (initial bootstrap failure), the subscription
        must not start a reconnect loop — there's nothing to resume from."""
        client = make_client(seat_id=None)
        # No mock needed — the early-return path doesn't touch HTTP at all.
        await client.run_sse_subscription()
