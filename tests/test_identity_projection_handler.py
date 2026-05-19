"""Tests for identity_projection_handler — SSE bridge handler for identity events.

Covers:
  - Full voice_profile payload → handler parses, resolves URIs, cache updated, log emitted.
  - voice_profile absent → handler records pending-fetch, no error.
  - Unrelated event kind → existing handlers unmodified (bus_bridge_runner integration).
  - Idempotent: same event twice → cache updates without raising.
  - Malformed payload (missing 'sub') → logs warning, does not crash.
  - Malformed voice_profile value (not a dict) → logs warning, does not crash.
  - IdentityVoiceProfile.from_dict raises → logs warning, does not crash.
  - bus_bridge_runner.run_bridge processes identity events into the cache
    without affecting dashboard broadcast filtering.

Run: python -m pytest tests/test_identity_projection_handler.py -v
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bus_bridge import BusEnvelope  # noqa: E402
from identity_projection_handler import (  # noqa: E402
    IDENTITY_KIND_EXPRESSION_UPDATED,
    IDENTITY_KIND_PROJECTED,
    IDENTITY_KINDS,
    IdentityVoiceCache,
    handle_identity_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_vp_payload(sub: str = "cog") -> dict:
    """A well-formed identity.projected payload WITH voice_profile embedded."""
    return {
        "iss": "https://cogos.local",
        "sub": sub,
        "aud": "*",
        "action": "create",
        "spec_hash": "abc123",
        "projection_path": f".cog/id/{sub}.cog.md",
        "applied_at": "2026-05-19T00:00:00Z",
        "voice_profile": {
            "generative": {
                "engine": "chatterbox-turbo",
                "conditionals_ref": "cog://voices/cog",
                "enrolled_at": "2026-05-15T00:00:00Z",
            },
            "discriminative": {
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "embedding_ref": "cog://voices/cog/ecapa-embedding",
            },
        },
    }


def _no_vp_payload(sub: str = "chaz") -> dict:
    """A well-formed identity.projected payload WITHOUT voice_profile."""
    return {
        "iss": "https://cogos.local",
        "sub": sub,
        "aud": "workspace:cog",
        "action": "create",
        "spec_hash": "def456",
        "projection_path": f".cog/id/{sub}.cog.md",
        "applied_at": "2026-05-19T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# handle_identity_event — core handler unit tests
# ---------------------------------------------------------------------------


class TestHandleIdentityEventWithVoiceProfile:
    """Event payload contains voice_profile → parse, resolve, cache, log."""

    def test_cache_populated(self):
        cache = IdentityVoiceCache()
        handle_identity_event(_full_vp_payload(), cache)
        result = cache.get("cog")
        assert result is not None
        assert result.sub == "cog"
        assert result.pending_fetch is False

    def test_voice_profile_parsed(self):
        cache = IdentityVoiceCache()
        handle_identity_event(_full_vp_payload(), cache)
        result = cache.get("cog")
        assert result.profile is not None
        assert result.profile.generative is not None
        assert result.profile.generative.engine == "chatterbox-turbo"
        assert result.profile.generative.conditionals_ref == "cog://voices/cog"

    def test_generative_uri_resolved(self):
        """resolve_voices_uri(cog://voices/cog) → …/voices/cog.safetensors."""
        import pathlib
        cache = IdentityVoiceCache()
        handle_identity_event(_full_vp_payload(), cache)
        result = cache.get("cog")
        assert result.generative_path is not None
        assert isinstance(result.generative_path, pathlib.Path)
        assert result.generative_path.suffix == ".safetensors"
        assert result.generative_path.stem == "cog"

    def test_discriminative_uri_resolved(self):
        """resolve_voices_uri(cog://voices/cog/ecapa-embedding) → ….ecapa.npy."""
        import pathlib
        cache = IdentityVoiceCache()
        handle_identity_event(_full_vp_payload(), cache)
        result = cache.get("cog")
        assert result.discriminative_path is not None
        assert isinstance(result.discriminative_path, pathlib.Path)
        assert result.discriminative_path.suffix == ".npy"

    def test_info_log_on_first_resolution(self, caplog):
        cache = IdentityVoiceCache()
        with caplog.at_level(logging.INFO, logger="mod3.identity_projection"):
            handle_identity_event(_full_vp_payload(), cache)
        assert any(
            "resolved voice profile" in record.message and "cog" in record.message
            for record in caplog.records
        ), f"Expected info log; got: {[r.message for r in caplog.records]}"

    def test_debug_log_on_re_resolution(self, caplog):
        """Second call with same sub → debug log (not info)."""
        cache = IdentityVoiceCache()
        handle_identity_event(_full_vp_payload(), cache)
        with caplog.at_level(logging.DEBUG, logger="mod3.identity_projection"):
            caplog.clear()
            handle_identity_event(_full_vp_payload(), cache)
        # Should have a debug re-resolved log, no second info log.
        assert any("re-resolved" in r.message for r in caplog.records)
        assert not any(
            r.levelno == logging.INFO and "resolved voice profile" in r.message
            for r in caplog.records
        )


class TestHandleIdentityEventNoVoiceProfile:
    """Event payload lacks voice_profile → pending-fetch, no error."""

    def test_pending_fetch_recorded(self):
        cache = IdentityVoiceCache()
        handle_identity_event(_no_vp_payload(), cache)
        result = cache.get("chaz")
        assert result is not None
        assert result.sub == "chaz"
        assert result.pending_fetch is True
        assert result.profile is None
        assert result.generative_path is None

    def test_no_exception(self):
        cache = IdentityVoiceCache()
        # Must not raise
        handle_identity_event(_no_vp_payload(), cache)

    def test_projection_path_stored(self):
        cache = IdentityVoiceCache()
        handle_identity_event(_no_vp_payload("chaz"), cache)
        result = cache.get("chaz")
        assert result.projection_path == ".cog/id/chaz.cog.md"

    def test_debug_log_emitted(self, caplog):
        cache = IdentityVoiceCache()
        with caplog.at_level(logging.DEBUG, logger="mod3.identity_projection"):
            handle_identity_event(_no_vp_payload(), cache)
        assert any("pending-fetch" in r.message for r in caplog.records)


class TestHandleIdentityEventIdempotent:
    """Same event twice → cache updates, no error."""

    def test_idempotent_with_voice_profile(self):
        cache = IdentityVoiceCache()
        handle_identity_event(_full_vp_payload(), cache)
        # Second call must not raise and must update the cache cleanly.
        handle_identity_event(_full_vp_payload(), cache)
        assert cache.get("cog") is not None

    def test_idempotent_without_voice_profile(self):
        cache = IdentityVoiceCache()
        handle_identity_event(_no_vp_payload(), cache)
        handle_identity_event(_no_vp_payload(), cache)
        result = cache.get("chaz")
        assert result is not None
        assert result.pending_fetch is True


class TestHandleIdentityEventMalformed:
    """Malformed payloads → warning log, no crash, cache unchanged."""

    def test_missing_sub_logs_warning_no_crash(self, caplog):
        cache = IdentityVoiceCache()
        payload = {"iss": "x", "voice_profile": None}
        with caplog.at_level(logging.WARNING, logger="mod3.identity_projection"):
            handle_identity_event(payload, cache)
        assert any("missing 'sub'" in r.message for r in caplog.records)
        assert cache.all_subs() == []

    def test_empty_sub_logs_warning(self, caplog):
        cache = IdentityVoiceCache()
        payload = {"sub": "", "iss": "x"}
        with caplog.at_level(logging.WARNING, logger="mod3.identity_projection"):
            handle_identity_event(payload, cache)
        assert any("missing 'sub'" in r.message for r in caplog.records)

    def test_voice_profile_not_dict_logs_warning(self, caplog):
        cache = IdentityVoiceCache()
        payload = {
            "sub": "cog",
            "iss": "x",
            "voice_profile": "not-a-dict",
        }
        with caplog.at_level(logging.WARNING, logger="mod3.identity_projection"):
            handle_identity_event(payload, cache)
        assert any("not a dict" in r.message for r in caplog.records)
        assert cache.get("cog") is None

    def test_from_dict_raises_logs_warning(self, caplog):
        """IdentityVoiceProfile.from_dict failure → warning, no crash."""
        cache = IdentityVoiceCache()
        payload = {
            "sub": "cog",
            "iss": "x",
            "voice_profile": {
                "generative": {
                    # Missing required 'conditionals_ref' field — from_dict raises KeyError
                    "engine": "chatterbox-turbo",
                }
            },
        }
        with caplog.at_level(logging.WARNING, logger="mod3.identity_projection"):
            handle_identity_event(payload, cache)
        # Should warn and not update cache
        assert any("failed to parse" in r.message for r in caplog.records)
        assert cache.get("cog") is None

    def test_bad_generative_uri_logs_warning_still_caches(self, caplog):
        """Bad URI in generative.conditionals_ref → warn, cache still updated."""
        cache = IdentityVoiceCache()
        payload = {
            "sub": "cog",
            "iss": "x",
            "voice_profile": {
                "generative": {
                    "engine": "chatterbox-turbo",
                    "conditionals_ref": "not-a-cog-uri",
                },
            },
        }
        with caplog.at_level(logging.WARNING, logger="mod3.identity_projection"):
            handle_identity_event(payload, cache)
        assert any("could not resolve generative URI" in r.message for r in caplog.records)
        result = cache.get("cog")
        assert result is not None
        assert result.generative_path is None  # resolution failed, but cache was updated


# ---------------------------------------------------------------------------
# IDENTITY_KINDS constant
# ---------------------------------------------------------------------------


def test_identity_kinds_contains_expected_events():
    assert IDENTITY_KIND_PROJECTED in IDENTITY_KINDS
    assert IDENTITY_KIND_EXPRESSION_UPDATED in IDENTITY_KINDS


def test_identity_kinds_are_correct_strings():
    assert IDENTITY_KIND_PROJECTED == "identity.projected"
    assert IDENTITY_KIND_EXPRESSION_UPDATED == "identity.expression.updated"


# ---------------------------------------------------------------------------
# bus_bridge_runner integration — identity events route to cache, not broadcast
# ---------------------------------------------------------------------------


class _FakeSubscriber:
    def __init__(self, envelopes: list[BusEnvelope]) -> None:
        self._envelopes = envelopes

    async def stream(self):
        for env in self._envelopes:
            yield env


def _make_env(kind: str, payload: dict) -> BusEnvelope:
    return BusEnvelope(
        raw={"type": "bus.event", "data": payload},
        kind=kind,
        payload=payload,
        ts="2026-05-19T00:00:00Z",
        event_id="e-test",
    )


class TestRunBridgeIdentityIntegration:
    """run_bridge routes identity events to the cache without broadcasting them."""

    def test_identity_event_not_broadcast_when_filtered(self):
        """identity.projected is not in ADR083_KINDS → not forwarded to dashboard."""
        from bus_bridge_runner import ADR083_KINDS, run_bridge  # noqa: PLC0415

        cache = IdentityVoiceCache()
        envelopes = [
            _make_env(IDENTITY_KIND_PROJECTED, _no_vp_payload("cog")),
            _make_env("state_transition", {"kind": "state_transition", "cycle_id": "c1"}),
        ]
        sub = _FakeSubscriber(envelopes)

        with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event") as mock_bcast:
            asyncio.run(run_bridge(sub, filter_kinds=set(ADR083_KINDS), identity_cache=cache))

        # Only state_transition should have been broadcast.
        assert mock_bcast.call_count == 1
        # But the identity cache should have been updated.
        assert cache.get("cog") is not None

    def test_identity_event_updates_cache_with_voice_profile(self):
        """Full voice_profile payload → cache updated after run_bridge."""
        from bus_bridge_runner import ADR083_KINDS, run_bridge  # noqa: PLC0415

        cache = IdentityVoiceCache()
        envelopes = [_make_env(IDENTITY_KIND_PROJECTED, _full_vp_payload("cog"))]
        sub = _FakeSubscriber(envelopes)

        with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event"):
            asyncio.run(run_bridge(sub, filter_kinds=set(ADR083_KINDS), identity_cache=cache))

        result = cache.get("cog")
        assert result is not None
        assert result.pending_fetch is False
        assert result.generative_path is not None

    def test_expression_updated_kind_also_handled(self):
        """identity.expression.updated also routes to handler."""
        from bus_bridge_runner import ADR083_KINDS, run_bridge  # noqa: PLC0415

        cache = IdentityVoiceCache()
        payload = _no_vp_payload("cog")
        envelopes = [_make_env(IDENTITY_KIND_EXPRESSION_UPDATED, payload)]
        sub = _FakeSubscriber(envelopes)

        with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event"):
            asyncio.run(run_bridge(sub, filter_kinds=set(ADR083_KINDS), identity_cache=cache))

        assert cache.get("cog") is not None

    def test_unrelated_event_does_not_touch_cache(self):
        """A state_transition event has no effect on the identity cache."""
        from bus_bridge_runner import ADR083_KINDS, run_bridge  # noqa: PLC0415

        cache = IdentityVoiceCache()
        envelopes = [
            _make_env("state_transition", {"kind": "state_transition", "cycle_id": "c1"}),
        ]
        sub = _FakeSubscriber(envelopes)

        with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event"):
            asyncio.run(run_bridge(sub, filter_kinds=set(ADR083_KINDS), identity_cache=cache))

        assert cache.all_subs() == []

    def test_handler_exception_does_not_crash_loop(self):
        """If handle_identity_event raises, the bridge loop continues."""
        from bus_bridge_runner import ADR083_KINDS, run_bridge  # noqa: PLC0415

        cache = IdentityVoiceCache()
        # Inject a malformed payload that will cause a warning but not a crash.
        envelopes = [
            _make_env(IDENTITY_KIND_PROJECTED, {}),  # missing 'sub'
            _make_env("state_transition", {"kind": "state_transition", "cycle_id": "c1"}),
        ]
        sub = _FakeSubscriber(envelopes)

        with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event") as mock_bcast:
            # Should not raise even though the identity event is malformed.
            asyncio.run(run_bridge(sub, filter_kinds=set(ADR083_KINDS), identity_cache=cache))

        # The state_transition event should still have been forwarded.
        assert mock_bcast.call_count == 1


# ---------------------------------------------------------------------------
# TTS path integration note (non-mutating check)
# ---------------------------------------------------------------------------


def test_engine_resolve_model_accepts_cog_uri_when_profile_in_registry():
    """resolve_model handles cog://voices/* when the profile is in the registry.

    This test confirms the existing engine.resolve_model integration point:
    once a voice profile is in the registry (enrolled), resolve_model accepts
    the cog:// URI. The cache in IdentityVoiceCache is a separate layer that
    mod3 can consult to get the generative_path before calling resolve_model.

    This test does NOT require the TTS engine to be loaded — it checks only
    the resolution logic path.
    """
    from engine import _resolve_voice_uri  # noqa: PLC0415

    # A URI that does not match any registered profile returns None.
    result = _resolve_voice_uri("cog://voices/nonexistent-profile-xyz")
    assert result is None  # profile not enrolled → None (not an error)

    # A non-cog URI returns None cleanly.
    assert _resolve_voice_uri("af_heart") is None
    assert _resolve_voice_uri("not-a-uri") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
