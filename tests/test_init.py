"""Tests for OrbitChaosDetector initialization and integration triggers."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ocd_gd.orbit_detector import (
    OrbitChaosDetector,
)


@pytest.fixture
def mock_agama_orbit():
    """Mock agama.orbit to return dummy integration data without doing actual math."""
    # Dummy outputs corresponding to (time_arr, traj_arr, dev_arr, lyap_arr)
    dummy_time = np.linspace(0, 10, 100)
    dummy_traj = np.zeros((1, 100, 6))
    dummy_dev = np.ones((1, 6, 100, 6))
    dummy_lyap = np.array([[0.01, 0.02]])

    with patch(
        "agama.orbit", return_value=(dummy_time, dummy_traj, dummy_dev, dummy_lyap)
    ) as mock_orbit:
        yield mock_orbit


def test_init_single_orbit(mock_agama_orbit):
    """Test initializing with 1D initial conditions converts them to 2D automatically."""
    single_ic = [1.0, 0.0, 0.0, 0.0, 220.0, 0.0]  # 1D array-like
    mock_pot = MagicMock()

    detector = OrbitChaosDetector(ic=single_ic, pot=mock_pot)

    # 1D input should be reshaped to shape (1, 6)
    assert detector.ic.ndim == 2
    assert detector.ic.shape == (1, 6)
    assert detector.num_orbits == 1


def test_init_batch_orbits(mock_agama_orbit):
    """Test initializing with multiple 2D initial conditions tracks num_orbits correctly."""
    batch_ic = [
        [1.0, 0.0, 0.0, 0.0, 220.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 200.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 180.0, 0.0],
    ]
    mock_pot = MagicMock()

    detector = OrbitChaosDetector(ic=batch_ic, pot=mock_pot)

    assert detector.ic.shape == (3, 6)
    assert detector.num_orbits == 3


def test_init_triggers_agama_integration(mock_agama_orbit):
    """Verify that initialization immediately calls agama.orbit with correct parameters."""
    ic = [[1.0, 0.0, 0.0, 0.0, 220.0, 0.0]]
    mock_pot = MagicMock()

    detector = OrbitChaosDetector(
        ic=ic,
        pot=mock_pot,
        omega=0.05,
        iter_time=50.0,
        accuracy=1e-6,
        max_num_steps=500000,
    )

    # Verify agama.orbit was called exactly once during __init__
    mock_agama_orbit.assert_called_once()

    # Check that agama.orbit received the custom arguments
    _, kwargs = mock_agama_orbit.call_args
    assert kwargs["potential"] == mock_pot
    assert kwargs["Omega"] == 0.05
    assert kwargs["time"] == 50.0
    assert kwargs["accuracy"] == 1e-6
    assert kwargs["maxNumSteps"] == 500000

    # Verify attributes received mock integration outputs
    assert detector.timestamps is not None
    assert detector.trajectories is not None
    assert detector.lyapunov_exponents is not None
