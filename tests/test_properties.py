"""Tests for public properties and lazy-loading cache layers in OrbitChaosDetector."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ocd_gd.orbit_detector import OrbitChaosDetector


@pytest.fixture
def mock_detector():
    """Create a detector with controlled raw integration mock data."""
    dummy_time = np.linspace(0, 10, 5)
    dummy_traj = np.ones((1, 5, 6)) * 42.0
    # Shape: (1 orbit, 6 deviation vectors, 5 timesteps, 6 dimensions)
    dummy_dev = np.ones((1, 6, 5, 6))
    dummy_lyap = np.array([[0.005, 0.001]])

    with patch(
        "agama.orbit", return_value=(dummy_time, dummy_traj, dummy_dev, dummy_lyap)
    ):
        detector = OrbitChaosDetector(ic=[1, 0, 0, 0, 220, 0], pot=MagicMock())
        yield detector


# =========================================================================
# DIRECT GETTER PROPERTIES
# =========================================================================


def test_raw_property_getters(mock_detector):
    """Verify raw simulation properties directly return underlying private attributes."""
    assert mock_detector.timestamps is mock_detector._time_arr
    assert mock_detector.trajectories is mock_detector._traj_arr
    assert mock_detector.lyapunov_exponents is mock_detector._lyap

    # Confirm values match mock integration output
    assert np.all(mock_detector.trajectories == 42.0)
    assert mock_detector.timestamps.shape == (5,)


# =========================================================================
# LAZY LOADING & CACHING PROPERTIES
# =========================================================================


def test_deviation_vectors_lazy_loading(mock_detector):
    """Verify deviation vectors remain uncalculated until explicitly requested."""
    # Internal cache must start as None right after initialization
    assert mock_detector._dev_arr_normalized is None

    # Access property for the first time -> triggers _normalize_deviation_vectors()
    dev_v = mock_detector.deviation_vectors
    assert dev_v is not None
    assert mock_detector._dev_arr_normalized is not None

    # Subsequent access should return the cached instance
    assert mock_detector.deviation_vectors is dev_v


def test_sali_array_lazy_loading(mock_detector):
    """Verify SALI matrix computation is deferred until property access."""
    # Cache layer starts empty
    assert mock_detector._sali_arr is None

    # First access computes and populates cache
    sali = mock_detector.sali_array
    assert sali is not None
    assert mock_detector._sali_arr is not None

    # Second access returns cached reference
    assert mock_detector.sali_array is sali


def test_gali_array_lazy_loading(mock_detector):
    """Verify GALI matrix computation is deferred until property access."""
    # Cache layer starts empty
    assert mock_detector._gali_arr is None

    # First access computes and populates cache
    gali = mock_detector.gali_array
    assert gali is not None
    assert mock_detector._gali_arr is not None

    # Second access returns cached reference
    assert mock_detector.gali_array is gali


def test_compute_sali_called_only_once(mock_detector):
    """Ensure internal _compute_sali is only called on first property access."""
    with patch.object(
        mock_detector, "_compute_sali", wraps=mock_detector._compute_sali
    ) as mock_sali:
        _ = mock_detector.sali_array
        _ = mock_detector.sali_array
        _ = mock_detector.sali_array

        # Compute should only be invoked once across multiple reads
        mock_sali.assert_called_once()
