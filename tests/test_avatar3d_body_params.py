"""avatar3d (kind=three) act params must survive body.py's validate_command().

The manifest fixture is the JSON that dashboard/avatar3d-body.js manifest() emits; the node test
tests/avatar3d-body-manifest.test.mjs asserts the fixture still equals the live output, so this
test cannot drift from the real body. Before the fix, manifest() had no `params` list and every
act param was stripped as rejected.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "avatar3d_manifest.json"


def _manifest() -> dict:
    return json.loads(FIXTURE.read_text())


def _body():
    from body import _Body

    return _Body(body_id="three-1", manifest=_manifest(), send=None)


def test_params_act_is_not_stripped():
    from body import validate_command

    clean, rejected = validate_command(_body(), {"params": {"head.x": 0.5, "mouth.open": 0.25}})
    assert rejected == []
    assert clean["params"] == {"head.x": 0.5, "mouth.open": 0.25}


def test_params_release_is_not_stripped():
    from body import validate_command

    clean, rejected = validate_command(_body(), {"release": ["head.x"]})
    assert rejected == []
    assert clean["release"] == ["head.x"]


def test_out_of_range_is_clamped_to_polarity():
    from body import validate_command

    clean, _ = validate_command(_body(), {"params": {"head.x": 5, "mouth.open": -3, "mouth.smile": -9}})
    assert clean["params"] == {"head.x": 1.0, "mouth.open": 0.0, "mouth.smile": -1.0}


def test_unknown_channel_still_rejected():
    from body import validate_command

    clean, rejected = validate_command(_body(), {"params": {"not.a.channel": 1}})
    assert rejected == ["not.a.channel"]
    assert "params" not in clean


def test_non_finite_is_rejected():
    from body import validate_command

    clean, rejected = validate_command(_body(), {"params": {"head.x": math.nan, "head.y": math.inf}})
    assert sorted(rejected) == ["head.x", "head.y"]
    assert "params" not in clean
