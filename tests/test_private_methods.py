"""Tests for internal mathematical helper functions of OrbitChaosDetector."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ocd_gd.orbit_detector import OrbitChaosDetector


@pytest.fixture
def mock_detector():
    """Create a detector with controlled, predictable raw mock inputs."""
    # Dummy orbit return matching agama signature: (time, traj, dev, lyap)
    # Shape of dev_arr: (1 orbit, 6 deviation vectors, 10 time steps, 6 phase dimensions)
    dummy_time = np.linspace(0, 10, 10)
    dummy_traj = np.zeros((1, 10, 6))
    dummy_dev = np.random.randn(1, 6, 10, 6)
    dummy_lyap = np.zeros((1, 2))

    with patch(
        "agama.orbit", return_value=(dummy_time, dummy_traj, dummy_dev, dummy_lyap)
    ):
        detector = OrbitChaosDetector(ic=[1, 0, 0, 0, 220, 0], pot=MagicMock())
        yield detector


# =========================================================================
# DEVIATION VECTOR NORMALIZATION TESTS
# =========================================================================


def test_normalize_deviation_vectors_unit_length(mock_detector):
    """Verify normalized vectors always have unit length (norm == 1.0)."""
    normalized = mock_detector._normalize_deviation_vectors()

    # Calculate L2 norm along the vector space dimension (axis -1)
    norms = np.linalg.norm(normalized, axis=-1)

    # All vector norms must equal 1.0 within floating point precision
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_normalize_deviation_vectors_handles_nan_and_inf(mock_detector):
    """Ensure NaNs and numeric overflow (Inf) in raw deviation arrays are safely cleaned."""
    # Inject dirty data into internal dev array
    dirty_dev = np.ones((1, 6, 5, 6))
    dirty_dev[0, 0, 0, 0] = np.nan
    dirty_dev[0, 0, 1, 0] = np.inf
    dirty_dev[0, 0, 2, 0] = -np.inf
    dirty_dev[0, 1, :, :] = 0.0  # Entire zero vector

    mock_detector._dev_arr = dirty_dev
    clean_normalized = mock_detector._normalize_deviation_vectors()

    # Check no NaNs or Infs remain
    assert not np.isnan(clean_normalized).any()
    assert not np.isinf(clean_normalized).any()


# =========================================================================
# SALI COMPUTATION TESTS
# =========================================================================


def test_compute_sali_parallel_vectors(mock_detector):
    """When two deviation vectors are identical or antiparallel, SALI must equal 0.0."""
    # Create 2 deviation vectors across 1 timestep in 6D
    # Vector 1 and Vector 2 are identical (parallel)
    v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Fill remaining 4 vectors with random values
    dev_arr = np.random.randn(1, 6, 1, 6)
    dev_arr[0, 0, 0, :] = v1
    dev_arr[0, 1, 0, :] = v2

    mock_detector._dev_arr_normalized = dev_arr
    sali = mock_detector._compute_sali()

    # SALI for pair (v1, v2) where v1 == v2 is min(||v1+v2||, ||v1-v2||) = ||v1-v2|| = 0.0
    # Checking that min alignment for pair (0, 1) evaluates to 0.0
    assert np.isclose(sali[0, 0, 0], 0.0)


def test_compute_sali_output_range(mock_detector):
    """SALI values should strictly fall within [0.0, 2.0] range."""
    sali = mock_detector._compute_sali()

    assert np.all(sali >= 0.0)
    assert np.all(sali <= 2.0 + 1e-12)


# =========================================================================
# GALI COMPUTATION TESTS
# =========================================================================


def test_compute_gali_linearly_dependent_vectors(mock_detector):
    """GALI must collapse to 0.0 if deviation vectors become linearly dependent."""
    # Create 6 linearly dependent deviation vectors (all identical copies)
    single_v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dev_arr = np.zeros((1, 6, 1, 6))
    for i in range(6):
        dev_arr[0, i, 0, :] = single_v

    mock_detector._dev_arr_normalized = dev_arr
    gali = mock_detector._compute_gali()

    # Since rank is 1 instead of 6, SVD singular values will have zeros, resulting in product 0.0
    assert np.isclose(gali[0, 0], 0.0, atol=1e-12)


def test_compute_gali_orthonormal_vectors(mock_detector):
    """GALI for 6 mutually orthogonal unit vectors must equal 1.0."""
    # 6D identity matrix forms 6 orthonormal vectors
    identity_6d = np.eye(6)
    dev_arr = np.zeros((1, 6, 1, 6))
    for i in range(6):
        dev_arr[0, i, 0, :] = identity_6d[i]

    mock_detector._dev_arr_normalized = dev_arr
    gali = mock_detector._compute_gali()

    # Product of all singular values for identity matrix is 1.0 * 1.0 * ... = 1.0
    assert np.isclose(gali[0, 0], 1.0, atol=1e-12)
