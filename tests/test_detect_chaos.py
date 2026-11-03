"""Tests for decay classification and report generation in OrbitChaosDetector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocd_gd._types import DecayClassification
from ocd_gd.orbit_detector import OrbitChaosDetector


@pytest.fixture
def chaos_detector():
    """Create a multi-orbit detector fixture with mocked integrations."""
    dummy_time = np.linspace(0, 100, 50)
    dummy_traj = np.zeros((2, 50, 6))
    dummy_dev = np.ones((2, 6, 50, 6))
    dummy_lyap = np.array([[0.01, 0.005], [0.001, 0.0001]])

    pot_mock = MagicMock()
    pot_mock.Tcirc.return_value = np.array([2.0, 2.5])

    with patch(
        "agama.orbit", return_value=(dummy_time, dummy_traj, dummy_dev, dummy_lyap)
    ):
        batch_ic = [
            [1.0, 0.0, 0.0, 0.0, 220.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 200.0, 0.0],
        ]
        detector = OrbitChaosDetector(ic=batch_ic, pot=pot_mock, n_periods=30.0)
        yield detector


class TestChaosDetection:
    """Unit tests for chaos classification, caching, DecayClassification, and indexing."""

    def test_decay_classification_output(self, chaos_detector):
        report = chaos_detector.detect_chaos()
        assert isinstance(report, DecayClassification)
        assert isinstance(report.sali, dict)
        assert isinstance(report.gali, dict)
        assert "label" in report.sali
        assert "label" in report.gali
        assert hasattr(report, "sali_check")
        assert hasattr(report, "gali_check")
        assert hasattr(report, "sali_time")
        assert hasattr(report, "gali_time")
        assert hasattr(report, "lyapunov_check")
        assert len(report.sali_check) == 2
        assert len(report.gali_check) == 2

    def test_caching_default_runs(self, chaos_detector):
        report1 = chaos_detector.detect_chaos()
        report2 = chaos_detector.detect_chaos()
        assert report1 is report2

    def test_overrides_bypass_cache(self, chaos_detector):
        report1 = chaos_detector.detect_chaos()
        report2 = chaos_detector.detect_chaos(sali_preset="conservative")
        assert report1 is not report2

    def test_separate_sali(self, chaos_detector):
        report = chaos_detector.detect_chaos(separate_sali=True)
        assert isinstance(report, DecayClassification)
        assert report.sali["label"].shape[0] == 15
        assert len(report.sali_check) == 2

    def test_sliced_orbit_index(self, chaos_detector):
        summary = chaos_detector.detect_chaos(orbit_idx=1)
        assert isinstance(summary, DecayClassification)
        assert summary.gali["label"].size == 1

    def test_invalid_index_raises(self, chaos_detector):
        with pytest.raises(IndexError):
            chaos_detector.detect_chaos(orbit_idx=99)
