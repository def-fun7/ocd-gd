"""Tests for index validation and sliced retrieval methods in OrbitChaosDetector."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocd_gd.orbit_detector import OrbitChaosDetector


@pytest.fixture
def multi_orbit_detector():
    """Create a detector containing 3 distinct integrated orbits."""
    dummy_time = np.linspace(0, 10, 5)

    # Create distinct trajectory data for 3 orbits to verify correct slicing
    # Orbit 0 filled with 1.0, Orbit 1 filled with 2.0, Orbit 2 filled with 3.0
    dummy_traj = np.zeros((3, 5, 6))
    dummy_traj[0] = 1.0
    dummy_traj[1] = 2.0
    dummy_traj[2] = 3.0

    dummy_dev = np.ones((3, 6, 5, 6))
    dummy_lyap = np.zeros((3, 2))

    with patch(
        "agama.orbit", return_value=(dummy_time, dummy_traj, dummy_dev, dummy_lyap)
    ):
        # Provide 3 initial conditions
        batch_ic = [
            [1.0, 0.0, 0.0, 0.0, 220.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 210.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 200.0, 0.0],
        ]
        detector = OrbitChaosDetector(ic=batch_ic, pot=MagicMock())
        yield detector


# =========================================================================
# INDEX VALIDATION TESTS
# =========================================================================


@pytest.mark.parametrize(
    ("idx", "expectation"),
    [
        (None, nullcontext()),
        (0, nullcontext()),
        (2, nullcontext()),
        (-1, pytest.raises(IndexError, match="out of bounds")),
        (3, pytest.raises(IndexError, match="out of bounds")),
        (10, pytest.raises(IndexError, match="out of bounds")),
    ],
)
def test_validate_index(multi_orbit_detector, idx, expectation):
    with expectation:
        multi_orbit_detector._validate_index(idx)


# =========================================================================
# GET_TRAJECTORY TESTS
# =========================================================================


def test_get_trajectory_single_orbit_slice(multi_orbit_detector):
    """get_trajectory(orbit_idx=1) returns data specifically for orbit 1."""
    traj = multi_orbit_detector.get_trajectory(orbit_idx=1)

    # Check shape is reduced by 1 dimension (orbit axis sliced)
    assert traj.shape == (5, 6)
    # Check we got Orbit 1's unique values (2.0)
    assert np.all(traj == 2.0)


# =========================================================================
# GET_SALI AND GET_GALI TESTS
# =========================================================================


def test_get_sali_slicing(multi_orbit_detector):
    """get_sali() slices batch matrix properly when an index is passed."""
    full_sali = multi_orbit_detector.get_sali()
    sliced_sali = multi_orbit_detector.get_sali(orbit_idx=0)

    assert full_sali.shape[0] == 15
    assert full_sali.shape[1] == 3
    assert np.array_equal(sliced_sali, full_sali[:, 0])


def test_get_gali_slicing(multi_orbit_detector):
    """get_gali() slices batch matrix properly when an index is passed."""
    full_gali = multi_orbit_detector.get_gali()
    sliced_gali = multi_orbit_detector.get_gali(orbit_idx=2)

    assert full_gali.shape[0] == 3
    assert np.array_equal(sliced_gali, full_gali[2])


@pytest.mark.parametrize("getter", ["get_trajectory", "get_sali", "get_gali"])
def test_getters_raise_on_invalid_index(multi_orbit_detector, getter):
    with pytest.raises(IndexError):
        getattr(multi_orbit_detector, getter)(orbit_idx=99)
