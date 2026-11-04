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
# LAZY LOADING & CACHING PROPERTIES
# =========================================================================
@pytest.mark.parametrize(
    ("prop", "cache_attr", "shape"),
    [
        ("deviation_vectors", "_dev_arr_normalized", (1, 6, 5, 6)),
        ("sali_array", "_sali_arr", (15, 1, 5)),
        ("gali_array", "_gali_arr", (1, 5)),
    ],
)
def test_lazy_property_cached_with_expected_shape(
    mock_detector, prop, cache_attr, shape
):
    assert getattr(mock_detector, cache_attr) is None  # nothing computed at init
    first = getattr(mock_detector, prop)
    assert first.shape == shape
    assert getattr(mock_detector, cache_attr) is first  # stored in the cache
    assert getattr(mock_detector, prop) is first  # second access doesn't recompute
