"""Unit tests for _dedup_repeated_transcript — Whisper phrase-doubling backstop.

The function lives in channels.py and is applied to raw Whisper output before
the is_hallucination() filter.  These tests exercise the exact-match,
near-duplicate, and no-op paths.

Run: python -m pytest tests/test_stt_dedup.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channels import _dedup_repeated_transcript  # noqa: E402

# ---------------------------------------------------------------------------
# Exact-match cases
# ---------------------------------------------------------------------------


def test_exact_double_sentence():
    """Classic Whisper hallucination: same sentence appears twice verbatim."""
    result = _dedup_repeated_transcript("Hello world. Hello world.")
    assert result == "Hello world."


def test_exact_double_multisentence():
    """Two-sentence utterance repeated exactly."""
    inp = "How are you feeling today? Are you able to hear me okay? How are you feeling today? Are you able to hear me okay?"
    result = _dedup_repeated_transcript(inp)
    # Should return only the first half.
    assert "How are you feeling today? Are you able to hear me okay?" in result
    # And not contain the duplicate.
    assert result.count("How are you feeling today?") == 1


def test_no_change_single_occurrence():
    """Single utterance with no repetition is returned unchanged."""
    inp = "Hello world."
    assert _dedup_repeated_transcript(inp) == inp


def test_no_change_normal_sentence():
    """Normal multi-word sentence without repetition passes through."""
    inp = "The quick brown fox jumps over the lazy dog."
    assert _dedup_repeated_transcript(inp) == inp


# ---------------------------------------------------------------------------
# Near-duplicate cases
# ---------------------------------------------------------------------------


def test_near_duplicate_single_char_diff():
    """One-character difference in otherwise identical halves."""
    # "Hello world." vs "Hello world!" — 11/12 chars match = 91.7 % < 95 %
    # This should NOT be deduped (below threshold).
    inp = "Hello world. Hello world!"
    result = _dedup_repeated_transcript(inp)
    # Below threshold — original returned.
    assert result == inp


def test_near_duplicate_whitespace_variation():
    """Trailing whitespace differences are normalised by strip() in the function."""
    inp = "Hello world.  Hello world."
    result = _dedup_repeated_transcript(inp)
    assert result == "Hello world."


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------


def test_empty_string():
    assert _dedup_repeated_transcript("") == ""


def test_whitespace_only():
    assert _dedup_repeated_transcript("   ") == "   "


def test_very_short_text():
    """Strings shorter than 4 chars are returned as-is (too short to dedup)."""
    assert _dedup_repeated_transcript("Hi") == "Hi"


def test_single_word_doubled():
    """Single word repeated — exact match path."""
    result = _dedup_repeated_transcript("okay okay")
    assert result == "okay"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
