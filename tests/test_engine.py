"""Tests for engine.py — generation pipeline helpers.

Run: python3 -m pytest tests/test_engine.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSparkSpeedSnap:
    """Spark's mlx_audio backend uses speed as a key into a fixed map.

    Any value not in {0.0, 0.5, 1.0, 1.5, 2.0} raises KeyError. Mod³'s default
    speak() speed is 1.25 (tuned for Kokoro), which previously hit this on
    every Spark call.
    """

    def test_default_speak_speed_snaps_to_moderate(self):
        from engine import _snap_to_spark_discrete

        # Mod³'s baseline default; equidistant from 1.0 and 1.5, prefer 1.0.
        assert _snap_to_spark_discrete(1.25) == 1.0

    def test_exact_discrete_values_pass_through(self):
        from engine import _snap_to_spark_discrete

        for v in (0.0, 0.5, 1.0, 1.5, 2.0):
            assert _snap_to_spark_discrete(v) == v

    def test_intermediate_values_snap_to_nearest(self):
        from engine import _snap_to_spark_discrete

        assert _snap_to_spark_discrete(0.3) == 0.5
        assert _snap_to_spark_discrete(0.74) == 0.5
        assert _snap_to_spark_discrete(1.49) == 1.5
        assert _snap_to_spark_discrete(1.75) == 1.5

    def test_out_of_range_clamps(self):
        from engine import _snap_to_spark_discrete

        assert _snap_to_spark_discrete(-1.0) == 0.0
        assert _snap_to_spark_discrete(3.0) == 2.0
        assert _snap_to_spark_discrete(100.0) == 2.0
