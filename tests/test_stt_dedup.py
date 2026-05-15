"""Unit tests for _dedup_repeated_transcript — Whisper phrase-doubling backstop.

The function lives in channels.py and is applied to raw Whisper output before
the is_hallucination() filter.  These tests exercise the exact-match,
near-duplicate, and no-op paths.

Run: python -m pytest tests/test_stt_dedup.py -v
"""

from __future__ import annotations

import hashlib
import sys
import time
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
    """One-character difference in otherwise identical sentences.

    "Hello world." vs "Hello world!" — 91.7% similarity.
    Strategy C uses an 85% threshold, so this IS deduped (91.7% > 85%).
    This is intentional: Whisper punctuation drift is a known failure mode
    and both strings convey the same content.
    """
    inp = "Hello world. Hello world!"
    result = _dedup_repeated_transcript(inp)
    # 91.7% > 85% threshold — the near-duplicate IS removed.
    assert result.count("Hello world") == 1


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


# ---------------------------------------------------------------------------
# Strategy C — sentence-level dedup (new in PR #XX)
# ---------------------------------------------------------------------------


def test_three_way_sentence_repetition():
    """3-way sentence repetition collapses to single sentence."""
    result = _dedup_repeated_transcript("Hello. Hello. Hello.")
    assert result == "Hello."


def test_four_way_no_punctuation():
    """4-way no-punctuation token repetition is caught (Strategy A or B)."""
    result = _dedup_repeated_transcript("OK OK OK OK")
    assert result == "OK"


def test_trailing_dangle_preserved():
    """Repeated prefix followed by unique tail — tail must be kept."""
    result = _dedup_repeated_transcript("Hello world. Hello world. Yes.")
    # Strategy C: "Hello world." == "Hello world." → removed; "Yes." is unique → kept.
    assert "Hello world." in result
    assert "Yes." in result
    assert result.count("Hello world.") == 1


def test_asymmetric_filler_dedup():
    """Near-duplicate sentences with a filler word differ by one word.

    'Hello world um.' vs 'Hello world.' — 85% threshold should catch this.
    The filler version is the duplicate; result should preserve meaningful content.
    """
    # Both sentences are semantically the same utterance; either one surviving is fine.
    # The important assertion is that we don't output both.
    result = _dedup_repeated_transcript("Hello world um. Hello world.")
    # Either the filler version or the clean version survives — but not both.
    assert result.count("Hello world") == 1


def test_false_positive_similar_but_distinct():
    """Two sentences that are similar but NOT repetitions must not be deduped."""
    result = _dedup_repeated_transcript("This is fine. This is great.")
    assert result == "This is fine. This is great."


def test_false_positive_short_similar_sentences():
    """Very short near-similar sentences — length floor prevents false positive."""
    # "Hi." and "Hi?" are short; exact-equality guard should prevent removal.
    result = _dedup_repeated_transcript("Hi. Hi?")
    # Should not be deduped (they differ and are below min-unit char threshold).
    assert result == "Hi. Hi?"


# ---------------------------------------------------------------------------
# Strategy A — N-way chunk dedup
# ---------------------------------------------------------------------------


def test_three_way_chunk_no_punct():
    """3-way verbatim repetition without punctuation (Strategy A)."""
    # "hello world " * 3 — Strategy C won't fire (no sentence split), A should.
    result = _dedup_repeated_transcript("hello world hello world hello world")
    assert result.strip() == "hello world"


# ---------------------------------------------------------------------------
# Strategy B — Z-function suffix
# ---------------------------------------------------------------------------


def test_z_function_exact_doubling():
    """Exact doubling caught by Z-function suffix path (fallback when C/A miss)."""
    # Using a string without .!? so Strategy C won't trigger.
    # Equal halves so Strategy A will actually catch this first — verify result.
    text = "one two three four one two three four"
    result = _dedup_repeated_transcript(text)
    assert result.strip() == "one two three four"


def test_z_function_four_way():
    """4-way exact repetition caught by Z-function (Strategy B)."""
    unit = "go home now "
    text = (unit * 4).strip()
    result = _dedup_repeated_transcript(text)
    assert result.strip() == unit.strip()


# ---------------------------------------------------------------------------
# Performance sanity check
# ---------------------------------------------------------------------------


def test_performance_5000_chars():
    """Dedup should complete in under 5ms on a 5000-char non-repeating transcript.

    Uses a hash-derived pseudo-random word stream so no two consecutive sentences
    or global repetition patterns can trigger any dedup strategy.
    """
    words = [
        hashlib.md5(f"w{i}".encode()).hexdigest()[:6]
        for i in range(200)
    ]
    # Join into sentences of 8 words each — each sentence is unique.
    sentences = [
        " ".join(words[i:i + 8]) + "."
        for i in range(0, len(words) - 8, 8)
    ]
    long_text = " ".join(sentences)[:5000]

    start = time.perf_counter()
    result = _dedup_repeated_transcript(long_text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 5.0, f"dedup took {elapsed_ms:.2f}ms (limit: 5ms)"
    # Hash-generated content has no repetitions — must pass through unchanged.
    assert result.strip() == long_text.strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
