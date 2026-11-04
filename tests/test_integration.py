"""Real-AGAMA sanity checks for `OrbitChaosDetector` (a few seconds in total).

A static spherical Hernquist-like potential and three orbits with different
radii, so every orbit has its own circular period `Tcirc` and therefore its own
integration time and time axis.  The spherical potential is integrable: all
orbits are regular and energy is conserved.
"""

import agama
import numpy as np
import pytest

from ocd_gd._decay_classifier import is_chaotic
from ocd_gd.orbit_detector import OrbitChaosDetector

agama.setUnits(length=1, mass=1, velocity=1)

# 30 periods is the package default.  SALI gave a marginal false "chaotic" call
# for orbit 0 at <= 15 periods, so keep a comfortable margin above that.
# ~50 samples per period keeps the sampling fixed as the length changes.
N_PERIODS = 30.0
TRAJ_SIZE = 600
# (x, y, z, vx, vy, vz) in kpc and km/s: near-circular, eccentric 3D, outer
ICS = np.array(
    [
        [10.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        [6.0, 0.0, 2.0, 20.0, 90.0, 40.0],
        [15.0, 0.0, 0.0, 0.0, 60.0, 30.0],
    ]
)
N_ORBITS = len(ICS)


@pytest.fixture(scope="module")
def spherical_potential():
    """Spherical Hernquist-like potential (integrable)."""
    return agama.Potential(
        type="Spheroid", mass=1e11, scaleradius=10.0, gamma=1.0, beta=4.0, xi=1.0
    )


@pytest.fixture(scope="module")
def batch_detector(spherical_potential):
    """All three orbits integrated together once and shared by the tests."""
    return OrbitChaosDetector(
        ic=ICS, pot=spherical_potential, n_periods=N_PERIODS, traj_size=TRAJ_SIZE
    )


# =========================================================================
# SHAPE AND TIME-AXIS TESTS
# =========================================================================


def test_array_shapes(batch_detector):
    """sali (15, N, T), gali (N, T), timestamps (N, T), trajectories (N, T, 6)."""
    assert batch_detector.sali_array.shape == (15, N_ORBITS, TRAJ_SIZE)
    assert batch_detector.gali_array.shape == (N_ORBITS, TRAJ_SIZE)
    assert batch_detector.timestamps.shape == (N_ORBITS, TRAJ_SIZE)
    assert batch_detector.trajectories.shape == (N_ORBITS, TRAJ_SIZE, 6)
    assert batch_detector.deviation_vectors.shape == (N_ORBITS, 6, TRAJ_SIZE, 6)


def test_integration_time_is_n_periods_times_circular_period(
    batch_detector, spherical_potential
):
    """Each orbit runs for n_periods * Tcirc, so the time axes differ per orbit."""
    expected = N_PERIODS * spherical_potential.Tcirc(ICS)
    np.testing.assert_allclose(batch_detector.iter_time, expected)
    assert np.ptp(batch_detector.iter_time) > 0

    timestamps = batch_detector.timestamps
    np.testing.assert_allclose(timestamps[:, -1], expected, rtol=1e-2)
    assert not np.allclose(timestamps[0], timestamps[1])


# =========================================================================
# PHYSICS SANITY TESTS
# =========================================================================


def test_energy_is_conserved_in_a_static_potential(batch_detector, spherical_potential):
    """E = Phi + v^2/2 drifts by less than 1e-6 (relative) along each orbit."""
    traj = batch_detector.trajectories
    positions = traj[..., :3].reshape(-1, 3)
    phi = spherical_potential.potential(positions).reshape(N_ORBITS, TRAJ_SIZE)
    energy = phi + 0.5 * np.sum(traj[..., 3:] ** 2, axis=-1)

    drift = np.max(np.abs(energy - energy[:, :1]), axis=1) / np.abs(energy[:, 0])
    assert np.all(drift < 1e-6), drift


def test_spherical_potential_orbits_are_not_flagged_chaotic(batch_detector):
    """The spherical potential is integrable, so no orbit may be called chaotic."""
    report = batch_detector.detect_chaos()
    assert not np.any(is_chaotic(report.gali["label"]))
    assert not np.any(report.gali_check)
    assert not np.any(report.sali_check)


# =========================================================================
# REPRODUCIBILITY AND BATCH-CONSISTENCY TESTS
# =========================================================================


def test_identical_runs_give_identical_results(batch_detector, spherical_potential):
    """Two detectors built from the same inputs agree exactly."""
    again = OrbitChaosDetector(
        ic=ICS, pot=spherical_potential, n_periods=N_PERIODS, traj_size=TRAJ_SIZE
    )

    np.testing.assert_array_equal(batch_detector.timestamps, again.timestamps)
    np.testing.assert_allclose(batch_detector.gali_array, again.gali_array, rtol=1e-12)
    np.testing.assert_allclose(batch_detector.sali_array, again.sali_array, rtol=1e-12)
    np.testing.assert_array_equal(
        batch_detector.detect_chaos().gali["label"], again.detect_chaos().gali["label"]
    )


@pytest.mark.parametrize("i", range(N_ORBITS))
def test_batch_result_matches_single_orbit_result(
    i, batch_detector, spherical_potential
):
    """An orbit integrated alone gives the same indices and labels as in a batch."""
    single = OrbitChaosDetector(
        ic=ICS[i], pot=spherical_potential, n_periods=N_PERIODS, traj_size=TRAJ_SIZE
    )

    np.testing.assert_allclose(single.timestamps[0], batch_detector.timestamps[i])
    np.testing.assert_allclose(
        single.sali_array[:, 0], batch_detector.sali_array[:, i], rtol=1e-6, atol=1e-12
    )
    np.testing.assert_allclose(
        single.gali_array[0], batch_detector.gali_array[i], rtol=1e-6, atol=1e-12
    )
    assert (
        single.detect_chaos().gali["label"][0]
        == batch_detector.detect_chaos().gali["label"][i]
    )
