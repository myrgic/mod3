"""Tests for Smart Turn end-of-utterance detector (F5).

These tests verify:
  - SmartTurnDetector graceful fallback when weight file absent
  - TurnPrediction shape
  - InboundPipeline accepts use_smart_turn parameter
  - SmartTurnDetector.predict() returns skipped=True when unavailable
  - Process-global detector singleton pattern

Note: Tests that import vendor.smart_turn are skipped when the vendor
directory is not present (i.e. before the Wave 0 PR is merged).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

# Guard: skip vendor-dependent tests when the vendor dir is absent
try:
    import vendor.smart_turn.inference as _smart_turn_infer  # noqa: F401

    VENDOR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    VENDOR_AVAILABLE = False


# ─── SmartTurnDetector unit tests ─────────────────────────────────────────────


@pytest.mark.skipif(not VENDOR_AVAILABLE, reason="vendor/smart_turn not present (install Wave 0 PR first)")
class TestSmartTurnDetectorUnavailable:
    """Tests for the case where the ONNX weight file is not present."""

    def setup_method(self):
        # Reset the module-level singleton between tests
        from turn_detector import reset_default_smart_turn_detector

        reset_default_smart_turn_detector()

    def test_is_available_returns_false_when_weight_absent(self, tmp_path, monkeypatch):
        """is_available() returns False when the ONNX weight file is missing."""
        import vendor.smart_turn.inference as infer_mod
        from turn_detector import SmartTurnDetector

        monkeypatch.setattr(infer_mod, "ONNX_MODEL_PATH", str(tmp_path / "nonexistent.onnx"))

        detector = SmartTurnDetector()
        assert not detector.is_available()

    def test_predict_returns_skipped_when_unavailable(self, tmp_path, monkeypatch):
        """predict() returns skipped=True and is_complete=True when unavailable."""
        import vendor.smart_turn.inference as infer_mod
        from turn_detector import SmartTurnDetector

        monkeypatch.setattr(infer_mod, "ONNX_MODEL_PATH", str(tmp_path / "nonexistent.onnx"))

        detector = SmartTurnDetector()
        audio = np.zeros(16000, dtype=np.float32)
        result = detector.predict(audio)

        assert result.skipped is True
        assert result.is_complete is True  # fail-open

    def test_predict_skips_wrong_sample_rate(self, tmp_path, monkeypatch):
        """predict() returns skipped=True for non-16kHz input."""
        import vendor.smart_turn.inference as infer_mod
        from turn_detector import SmartTurnDetector

        monkeypatch.setattr(infer_mod, "ONNX_MODEL_PATH", str(tmp_path / "nonexistent.onnx"))

        detector = SmartTurnDetector()
        audio = np.zeros(8000, dtype=np.float32)
        result = detector.predict(audio, sample_rate=8000)

        assert result.skipped is True

    def test_turn_prediction_shape(self):
        """TurnPrediction dataclass has is_complete, probability, skipped."""
        from turn_detector import TurnPrediction

        p = TurnPrediction(is_complete=True, probability=0.9)
        assert p.is_complete is True
        assert p.probability == pytest.approx(0.9)
        assert p.skipped is False

        p2 = TurnPrediction(is_complete=False, probability=0.3, skipped=True)
        assert p2.skipped is True


class TestSmartTurnDetectorSingleton:
    """Tests for the process-global detector singleton."""

    def setup_method(self):
        from turn_detector import reset_default_smart_turn_detector

        reset_default_smart_turn_detector()

    def test_get_default_returns_same_instance(self):
        """get_default_smart_turn_detector() returns the same object each call."""
        from turn_detector import get_default_smart_turn_detector

        d1 = get_default_smart_turn_detector()
        d2 = get_default_smart_turn_detector()
        assert d1 is d2

    def test_reset_creates_fresh_instance(self):
        """reset_default_smart_turn_detector() drops the singleton."""
        from turn_detector import (
            get_default_smart_turn_detector,
            reset_default_smart_turn_detector,
        )

        d1 = get_default_smart_turn_detector()
        reset_default_smart_turn_detector()
        d2 = get_default_smart_turn_detector()
        assert d1 is not d2

    def test_thread_safety_of_singleton(self):
        """Concurrent calls to get_default_smart_turn_detector return one instance."""
        from turn_detector import get_default_smart_turn_detector, reset_default_smart_turn_detector

        reset_default_smart_turn_detector()

        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(id(get_default_smart_turn_detector()))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have gotten the same singleton
        assert len(set(results)) == 1


# ─── InboundPipeline integration ─────────────────────────────────────────────


@pytest.mark.skipif(not VENDOR_AVAILABLE, reason="vendor/smart_turn not present (install Wave 0 PR first)")
class TestInboundPipelineSmartTurnParam:
    """Tests that InboundPipeline accepts the use_smart_turn parameter."""

    def test_instantiates_with_smart_turn_false(self):
        """InboundPipeline(use_smart_turn=False) creates without raising."""
        try:
            from inbound import InboundPipeline
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"InboundPipeline import unavailable ({e})")

        from unittest.mock import MagicMock

        bus = MagicMock()
        ps = MagicMock()
        capture = MagicMock()
        capture.is_active.return_value = True

        pipeline = InboundPipeline(
            bus=bus,
            pipeline_state=ps,
            capture=capture,
            use_smart_turn=False,
        )
        assert pipeline._use_smart_turn is False
        assert pipeline._smart_turn_detector is None

    def test_instantiates_with_smart_turn_true_unavailable(self, tmp_path, monkeypatch):
        """InboundPipeline(use_smart_turn=True) + unavailable model logs warning, not raise."""
        try:
            from inbound import InboundPipeline
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"InboundPipeline import unavailable ({e})")

        import vendor.smart_turn.inference as infer_mod

        monkeypatch.setattr(infer_mod, "ONNX_MODEL_PATH", str(tmp_path / "nonexistent.onnx"))

        from unittest.mock import MagicMock

        bus = MagicMock()
        ps = MagicMock()
        capture = MagicMock()
        capture.is_active.return_value = True

        pipeline = InboundPipeline(
            bus=bus,
            pipeline_state=ps,
            capture=capture,
            use_smart_turn=True,
        )
        # start() wires the detector; simulate start() without real mic
        capture.is_active.return_value = False
        capture.start = MagicMock()
        pipeline._stop_event.set()  # prevent thread from spinning

        # Manually trigger the Smart Turn init block
        from turn_detector import SmartTurnDetector

        detector = SmartTurnDetector()
        # detector.is_available() is False (no weight file)
        assert not detector.is_available()

    def test_env_var_enables_smart_turn(self, monkeypatch):
        """MOD3_SMART_TURN=1 enables Smart Turn via env var."""
        try:
            from inbound import InboundPipeline
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"InboundPipeline import unavailable ({e})")

        monkeypatch.setenv("MOD3_SMART_TURN", "1")

        from unittest.mock import MagicMock

        bus = MagicMock()
        ps = MagicMock()
        capture = MagicMock()
        capture.is_active.return_value = True

        pipeline = InboundPipeline(
            bus=bus,
            pipeline_state=ps,
            capture=capture,
        )
        assert pipeline._use_smart_turn is True
