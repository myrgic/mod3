"""Tests for the session voice update path and MOD3_DEFAULT_VOICE startup default.

Defect fixed here: POST /v1/sessions/register is idempotent on an already
registered session_id and SILENTLY DISCARDED preferred_voice. There was no
update path at all, and the startup default for 'main' was hardcoded to None.

Three halves:
  (a) SessionRegistry.set_voice() + PATCH /v1/sessions/{id}/voice — routed
      through _allocate_voice so voice_conflict semantics are preserved.
  (b) MOD3_DEFAULT_VOICE env var honored when auto-registering 'main'.
  (c) register() now reports voice_ignored instead of discarding silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_registry import SessionRegistry  # noqa: E402


@pytest.fixture
def registry():
    return SessionRegistry(voice_pool=["bm_lewis", "af_heart", "bf_emma"])


def _reg(registry, sid, voice=None):
    return registry.register(
        session_id=sid,
        participant_id=f"p-{sid}",
        preferred_voice=voice,
    )


# --- (c) silent discard is now reported -----------------------------------


def test_reregister_reports_voice_ignored(registry):
    _reg(registry, "main")
    result = _reg(registry, "main", "eng_uk_m_davids")
    assert result.created is False
    assert result.session.assigned_voice == "bm_lewis"
    assert result.voice_ignored == "eng_uk_m_davids"


def test_reregister_same_voice_is_not_reported_as_ignored(registry):
    _reg(registry, "main", "af_heart")
    result = _reg(registry, "main", "af_heart")
    assert result.voice_ignored is None


def test_first_registration_still_honors_preferred_voice(registry):
    result = _reg(registry, "main", "eng_uk_m_davids")
    assert result.created is True
    assert result.session.assigned_voice == "eng_uk_m_davids"
    assert result.voice_ignored is None


# --- (a) set_voice ---------------------------------------------------------


def test_set_voice_changes_assigned_voice(registry):
    _reg(registry, "main")
    out = registry.set_voice("main", "eng_uk_m_davids")
    assert out["status"] == "ok"
    assert out["changed"] is True
    assert out["previous_voice"] == "bm_lewis"
    assert out["assigned_voice"] == "eng_uk_m_davids"
    assert registry.get("main").assigned_voice == "eng_uk_m_davids"


def test_set_voice_releases_the_old_voice_to_the_pool(registry):
    _reg(registry, "main")
    registry.set_voice("main", "eng_uk_m_davids")
    holders = registry.voice_holder_snapshot()
    assert holders.get("eng_uk_m_davids") == "main"
    assert "bm_lewis" not in holders


def test_set_voice_flags_conflict_when_another_session_holds_it(registry):
    _reg(registry, "a")  # takes bm_lewis
    _reg(registry, "b")  # takes af_heart
    out = registry.set_voice("b", "bm_lewis")
    assert out["status"] == "ok"
    assert out["assigned_voice"] == "bm_lewis"
    assert out["voice_conflict"] is True
    assert registry.get("b").voice_conflict is True


def test_set_voice_noop_when_already_assigned(registry):
    _reg(registry, "main", "af_heart")
    out = registry.set_voice("main", "af_heart")
    assert out["status"] == "ok"
    assert out["changed"] is False


def test_set_voice_unknown_session(registry):
    assert registry.set_voice("nope", "af_heart")["status"] == "not_found"


def test_set_voice_rejects_empty(registry):
    _reg(registry, "main")
    assert registry.set_voice("main", "")["status"] == "error"


def test_set_voice_extends_pool_for_out_of_band_voice(registry):
    _reg(registry, "main")
    registry.set_voice("main", "eng_uk_m_davids")
    assert "eng_uk_m_davids" in registry.voice_pool()


# --- HTTP surface ----------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import http_api

    monkeypatch.delenv("MOD3_DEFAULT_VOICE", raising=False)
    with TestClient(http_api.app, base_url="http://localhost:7860") as c:
        yield c


def test_patch_voice_endpoint_changes_live_session(client):
    client.post(
        "/v1/sessions/register",
        json={"session_id": "voice-test", "participant_id": "tester"},
    )
    r = client.patch("/v1/sessions/voice-test/voice", json={"voice": "eng_uk_m_davids"})
    assert r.status_code == 200, r.text
    assert r.json()["assigned_voice"] == "eng_uk_m_davids"

    listed = client.get("/v1/sessions").json()["sessions"]
    row = next(s for s in listed if s["session_id"] == "voice-test")
    assert row["assigned_voice"] == "eng_uk_m_davids"
    client.post("/v1/sessions/voice-test/deregister")


def test_patch_voice_unknown_session_404(client):
    r = client.patch("/v1/sessions/does-not-exist/voice", json={"voice": "af_heart"})
    assert r.status_code == 404


def test_register_response_warns_when_voice_ignored(client):
    first = client.post(
        "/v1/sessions/register",
        json={
            "session_id": "warn-test",
            "participant_id": "tester",
            "preferred_voice": "bf_emma",
        },
    ).json()
    assert first["assigned_voice"] == "bf_emma"
    r = client.post(
        "/v1/sessions/register",
        json={
            "session_id": "warn-test",
            "participant_id": "tester",
            "preferred_voice": "eng_uk_m_davids",
        },
    )
    body = r.json()
    assert body["created"] is False
    assert body["voice_ignored"] == "eng_uk_m_davids"
    assert "warning" in body
    client.post("/v1/sessions/warn-test/deregister")


# --- (b) startup default ---------------------------------------------------


def test_main_session_uses_mod3_default_voice_on_startup(monkeypatch):
    from fastapi.testclient import TestClient

    import http_api
    from session_registry import get_default_registry

    monkeypatch.setenv("MOD3_DEFAULT_VOICE", "eng_uk_m_davids")
    # Simulate a warm registry that already has 'main' on a different voice —
    # this is the case a bare register() could never fix.
    get_default_registry().register(
        session_id="main",
        participant_id="channel-client-pool",
        preferred_voice="bm_lewis",
    )
    with TestClient(http_api.app, base_url="http://localhost:7860") as c:
        rows = c.get("/v1/sessions").json()["sessions"]
        main = next(s for s in rows if s["session_id"] == "main")
        assert main["assigned_voice"] == "eng_uk_m_davids"


def test_no_env_var_leaves_default_allocation_behavior(monkeypatch):
    from fastapi.testclient import TestClient

    import http_api

    monkeypatch.delenv("MOD3_DEFAULT_VOICE", raising=False)
    with TestClient(http_api.app, base_url="http://localhost:7860") as c:
        rows = c.get("/v1/sessions").json()["sessions"]
        assert any(s["session_id"] == "main" for s in rows)
