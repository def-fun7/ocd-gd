"""Tests for chaos classification and report generation in OrbitChaosDetector."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ocd_gd.orbit_detector import ChaosFullReport, ChaosSummary, OrbitChaosDetector


class TestChaosDetection:
    """Unit tests for chaos classification, caching, namedtuples, and indexing."""

    def test_summary_output(self, chaos_detector):
        detector, _ = chaos_detector
        report = detector.detect_chaos(check_only=True)
        assert isinstance(report, ChaosSummary)

    def test_full_report_output(self, chaos_detector):
        detector, _ = chaos_detector
        report = detector.detect_chaos(check_only=False)
        assert isinstance(report, ChaosFullReport)

    def test_caching_default_runs(self, chaos_detector):
        detector, mock_eval = chaos_detector
        _ = detector.detect_chaos()
        assert mock_eval.call_count == 2
        _ = detector.detect_chaos()
        assert mock_eval.call_count == 2  # Pulled from cache

    def test_threshold_overrides_bypass_cache(self, chaos_detector):
        detector, mock_eval = chaos_detector
        _ = detector.detect_chaos()
        _ = detector.detect_chaos(sali_override=1e-5)
        assert mock_eval.call_count == 4  # Cache bypassed

    def test_sliced_orbit_index(self, chaos_detector):
        detector, _ = chaos_detector
        summary = detector.detect_chaos(orbit_idx=1, check_only=True)
        assert summary.gali_check == 0


@pytest.fixture
def chaos_detector():
    """Create a multi-orbit detector fixture with mocked integrations and evaluation functions."""
    dummy_time = np.linspace(0, 100, 10)
    dummy_traj = np.zeros((2, 10, 6))
    dummy_dev = np.ones((2, 6, 10, 6))
    dummy_lyap = np.array([[0.01, 0.005], [0.001, 0.0001]])

    # Standard return values
    gali_mock_res = (np.array([True, False]), np.array([50.0, 100.0]))
    sali_mock_res = (np.array([True, False]), np.array([30.0, 100.0]))

    # Dynamic side effect function so it can be called infinitely many times
    def mock_eval_side_effect(*args, **kwargs):
        # You can inspect threshold or array shapes if needed, or simply return alternating results
        # Assuming GALI uses a default threshold (e.g. 1e-16) and SALI uses another (e.g. 1e-8)
        threshold = kwargs.get("threshold", 1e-16)
        if threshold == 1e-16:
            return gali_mock_res
        return sali_mock_res

    with (
        patch(
            "agama.orbit", return_value=(dummy_time, dummy_traj, dummy_dev, dummy_lyap)
        ),
        patch(
            "ocd_gd.orbit_detector.evaluate_chaos", side_effect=mock_eval_side_effect
        ) as mock_eval,
    ):
        batch_ic = [
            [1.0, 0.0, 0.0, 0.0, 220.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 200.0, 0.0],
        ]
        detector = OrbitChaosDetector(ic=batch_ic, pot=MagicMock())
        yield detector, mock_eval


# =========================================================================
# SUMMARY VS FULL REPORT CONTAINER TESTS
# =========================================================================


def test_detect_chaos_summary_output(chaos_detector):
    """When check_only=True (default), return a ChaosSummary namedtuple."""
    detector, _ = chaos_detector

    report = detector.detect_chaos(check_only=True)

    assert isinstance(report, ChaosSummary)
    assert hasattr(report, "gali_check")
    assert hasattr(report, "gali_time")
    assert hasattr(report, "sali_check")
    assert hasattr(report, "sali_time")

    # Check batch output length matches 2 orbits
    assert len(report.gali_check) == 2


def test_detect_chaos_full_report_output(chaos_detector):
    """When check_only=False, return a ChaosFullReport container."""
    detector, _ = chaos_detector

    report = detector.detect_chaos(check_only=False)

    assert isinstance(report, ChaosFullReport)
    assert isinstance(report.summary, ChaosSummary)
    assert report.timestamps is not None
    assert report.gali_array is not None
    assert report.sali_array is not None


# =========================================================================
# CACHING & OVERRIDE BEHAVIOR
# =========================================================================


def test_detect_chaos_caching_default_runs(chaos_detector):
    """Subsequent calls with default parameters should hit cache and avoid calling evaluate_chaos again."""
    detector, mock_eval = chaos_detector

    # Initial call triggers evaluate_chaos twice (once for GALI, once for SALI)
    _ = detector.detect_chaos()
    assert mock_eval.call_count == 2

    # Second call should pull directly from cache without calling evaluate_chaos
    _ = detector.detect_chaos()
    assert mock_eval.call_count == 2


def test_detect_chaos_threshold_overrides_bypass_cache(chaos_detector):
    """Providing threshold overrides must bypass the cache and invoke evaluate_chaos with new parameters."""
    detector, mock_eval = chaos_detector

    # First call populates cache (call_count becomes 2)
    _ = detector.detect_chaos()

    # Second call provides a custom sali threshold -> bypasses cache (call_count becomes 4)
    _ = detector.detect_chaos(sali_override=1e-5)
    assert mock_eval.call_count == 4


# =========================================================================
# SINGLE ORBIT SLICING
# =========================================================================


def test_detect_chaos_sliced_orbit_index(chaos_detector):
    """Targeting a specific orbit index returns single-orbit scalar/vector slices rather than batch arrays."""
    detector, _ = chaos_detector

    # Slice for orbit index 1
    summary = detector.detect_chaos(orbit_idx=1, check_only=True)

    # Check elements correspond to orbit 1
    assert summary.gali_check == False  # Match mock return value for index 1
    assert summary.gali_time == 100.0


def test_detect_chaos_invalid_index_raises(chaos_detector):
    """Passing an out-of-bounds orbit_idx raises IndexError."""
    detector, _ = chaos_detector

    with pytest.raises(IndexError):
        detector.detect_chaos(orbit_idx=99)
