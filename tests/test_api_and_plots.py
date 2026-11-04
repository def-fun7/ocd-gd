"""Package-level smoke tests: public API names and per-orbit plot methods.

The plot tests use a small real batch of orbits with different integration
times, so `timestamps` is 2-D (one time axis per orbit) -- the case the recent
plotting fix addressed.  They only check that every method runs.
"""

import agama
import numpy as np
import pytest

import ocd_gd
from ocd_gd.orbit_detector import OrbitChaosDetector

agama.setUnits(length=1, mass=1, velocity=1)

PLOT_METHODS = [
    "plot_sali",
    "plot_gali",
    "plot_trajectory_2d",
    "plot_trajectory_3d",
    "plot_phase_space",
    "plot_colored_trajectory",
    "plot_dashboard",
]
BATCH_PLOT_METHODS = ["plot_sali_batch", "plot_gali_batch", "plot_sali_gali_batch"]


# =========================================================================
# PUBLIC API TESTS
# =========================================================================


@pytest.mark.parametrize("name", ocd_gd.__all__)
def test_every_name_in_all_resolves(name):
    """Each name listed in `__all__` must exist on the package."""
    assert hasattr(ocd_gd, name), f"{name!r} is in __all__ but not importable"


def test_all_has_no_duplicates():
    """`__all__` lists each public name once."""
    assert len(ocd_gd.__all__) == len(set(ocd_gd.__all__))


# =========================================================================
# PLOT SMOKE TESTS
# =========================================================================


@pytest.fixture(scope="module")
def plot_detector():
    """Three short orbits with different Tcirc, hence different time axes."""
    pot = agama.Potential(
        type="Spheroid", mass=1e11, scaleradius=10.0, gamma=1.0, beta=4.0, xi=1.0
    )
    ics = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 100.0, 0.0],
            [6.0, 0.0, 2.0, 20.0, 90.0, 40.0],
            [15.0, 0.0, 0.0, 0.0, 60.0, 30.0],
        ]
    )
    detector = OrbitChaosDetector(ic=ics, pot=pot, n_periods=5.0, traj_size=300)
    assert np.ndim(detector.timestamps) == 2
    return detector


@pytest.mark.parametrize("method", PLOT_METHODS)
def test_single_orbit_plots_run_on_per_orbit_time_axes(plot_detector, method):
    """Every single-orbit plot works for an orbit that is not the first one."""
    getattr(plot_detector, method)(orbit_idx=1, backend="matplotlib", show=False)


@pytest.mark.parametrize("method", BATCH_PLOT_METHODS)
def test_batch_plots_return_figures(plot_detector, method):
    """Batch plots return a non-empty list of figures."""
    figures = getattr(plot_detector, method)(orbit_indices=[0, 1, 2], show=False)
    assert isinstance(figures, list)
    assert len(figures) >= 1
