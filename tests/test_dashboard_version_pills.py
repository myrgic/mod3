"""Regression tests for the dashboard's kernel + mod3 version pills.

Bug observed in the live dashboard:

- ``kernel vv0.10.0`` (double-v) — kernel ``/health`` returns version with a
  leading ``v`` ("v0.10.0") and the renderer concatenated ``' v' + d.version``,
  producing the double prefix.
- ``mod3`` pill rendered without any version at all — the mod3 ``/health``
  response carries ``"version": "0.7.0"`` but the renderer hard-coded the label
  to the bare string ``'mod3'``.

These tests are HTML grep regressions — JS unit-testing the dashboard would
require a JSDOM harness this project doesn't have yet. We assert that:

1. A normalizer helper (``formatVersion``) exists and strips a leading ``v``
   before re-prefixing, so both health-response shapes render the same way.
2. The kernel pill setter routes through that helper.
3. The mod3 pill setter routes through that helper (closes the missing-version
   regression).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = REPO_ROOT / "dashboard" / "index.html"


@pytest.fixture(scope="module")
def src() -> str:
    return INDEX_HTML.read_text()


class TestVersionPills:
    def test_format_version_helper_exists(self, src: str):
        """The shared normalizer prevents either pill from re-introducing the
        leading-v collision (kernel emits 'vX', mod3 emits 'X')."""
        assert re.search(r"function\s+formatVersion\s*\(", src), (
            "dashboard/index.html must define formatVersion() to normalize "
            "version strings across kernel + mod3 health responses"
        )
        assert "startsWith('v')" in src, "formatVersion must detect and strip a leading 'v' before re-prefixing"

    def test_kernel_pill_uses_formatter(self, src: str):
        """The kernel pill must funnel d.version through formatVersion so
        'v0.10.0' renders as 'kernel v0.10.0' rather than 'kernel vv0.10.0'."""
        assert re.search(
            r"setPill\(\s*'pill-kernel-dot'\s*,\s*'pill-kernel-label'\s*,\s*'ok'\s*,"
            r"\s*'kernel'\s*\+\s*formatVersion\(",
            src,
        ), "kernel pill setter must use formatVersion(d.version)"

    def test_mod3_pill_uses_formatter(self, src: str):
        """The mod3 pill must include the version — previously it hard-coded
        the bare label 'mod3' with no version concatenation at all."""
        assert re.search(
            r"setPill\(\s*'pill-mod3-dot'\s*,\s*'pill-mod3-label'\s*,"
            r"[^)]+,\s*'mod3'\s*\+\s*formatVersion\(",
            src,
        ), "mod3 pill setter must concatenate formatVersion(d.version)"

    def test_no_legacy_double_v_concat(self, src: str):
        """Guard against re-introducing the original `' v' + d.version` pattern
        on either pill — the formatter owns the leading v."""
        # The legacy bug shape: `' v' + d.version` (single-quoted leading v
        # prefix). Allowed elsewhere (e.g. log messages), but not in the pill
        # setters.
        for needle in ("'kernel' +", "'mod3' +"):
            # Each pill setter should ONLY concatenate through formatVersion,
            # not via a literal ' v' prefix.
            for line in src.splitlines():
                if needle in line and "formatVersion" not in line:
                    # Allow error/offline branches that emit just status codes
                    if "offline" in line or "r.status" in line:
                        continue
                    pytest.fail(f"version concatenation outside formatVersion(): {line.strip()!r}")
