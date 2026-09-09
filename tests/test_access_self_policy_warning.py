"""Tests for the policy=self startup trust warning (access.warn_if_self_policy).

`policy=self` trusts every loopback client unconditionally, which is a silent
trust-boundary hole once anything else (a tunnel, a reverse proxy) is also
bound to loopback in front of mod3. warn_if_self_policy() logs that once at
startup when the effective policy resolves to 'self', and stays silent for
'allowlist' / 'deny'.

Run with::

    PYTHONPATH=. python -m pytest tests/test_access_self_policy_warning.py -v
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import access  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_access_config(tmp_path, monkeypatch):
    """Point access.py at a scratch access.json and reset the warn-once latch.

    Every test gets a fresh config path and a fresh _self_policy_warned=False,
    so tests cannot see each other's "already warned this process" state.
    """
    config_path = tmp_path / "access.json"
    monkeypatch.setenv("MOD3_ACCESS_CONFIG", str(config_path))
    monkeypatch.setattr(access, "_self_policy_warned", False)
    yield config_path


def _write_policy(config_path: Path, policy: str) -> None:
    config_path.write_text(json.dumps({"policy": policy, "allow": [], "pending": []}))


def test_warns_once_when_policy_is_self(_isolated_access_config, caplog):
    _write_policy(_isolated_access_config, "self")

    with caplog.at_level(logging.WARNING, logger="mod3.access"):
        access.warn_if_self_policy()

    messages = [r.message for r in caplog.records if r.name == "mod3.access"]
    assert any("policy is 'self'" in m for m in messages)
    assert any("allowlist" in m for m in messages)


def test_does_not_warn_twice_in_one_process(_isolated_access_config, caplog):
    _write_policy(_isolated_access_config, "self")

    with caplog.at_level(logging.WARNING, logger="mod3.access"):
        access.warn_if_self_policy()
        caplog.clear()
        access.warn_if_self_policy()

    messages = [r.message for r in caplog.records if r.name == "mod3.access"]
    assert not any("policy is 'self'" in m for m in messages)


def test_no_warning_for_allowlist_policy(_isolated_access_config, caplog):
    _write_policy(_isolated_access_config, "allowlist")

    with caplog.at_level(logging.WARNING, logger="mod3.access"):
        access.warn_if_self_policy()

    messages = [r.message for r in caplog.records if r.name == "mod3.access"]
    assert not any("policy is 'self'" in m for m in messages)


def test_no_warning_for_deny_policy(_isolated_access_config, caplog):
    _write_policy(_isolated_access_config, "deny")

    with caplog.at_level(logging.WARNING, logger="mod3.access"):
        access.warn_if_self_policy()

    messages = [r.message for r in caplog.records if r.name == "mod3.access"]
    assert not any("policy is 'self'" in m for m in messages)
