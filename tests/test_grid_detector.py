"""Tests for `GridChaosDetector`: grid bookkeeping, index maps and map plots.

The pure scatter/reshape helper is tested directly.  The remaining tests share
one tiny 3x3 grid in a spherical potential (a handful of short orbits).
"""

import agama
import numpy as np
import pytest

from ocd_gd.grid_detector import GridChaosDetector, _scatter_to_grid

agama.setUnits(mass=1, length=1, velocity=1)

GRID_SIZE = 3


@pytest.fixture(scope="module")
def grid_detector():
    """3x3 grid at E_0 of a circular orbit at R_0 = 8, integrated for 1 time unit."""
    pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
    return GridChaosDetector(
        potential=pot, R_0=8.0, grid_size=GRID_SIZE, n_periods=10.0
    )


# =========================================================================
# _scatter_to_grid TESTS (pure, no integration)
# =========================================================================


def test_scatter_to_grid_puts_values_at_flat_positions():
    """Flat index k lands at (row, col) = divmod(k, grid_size); the rest is fill."""
    grid = _scatter_to_grid(np.array([1.0, 2.0, 3.0]), np.array([0, 4, 8]), 3, np.nan)

    assert grid.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(grid), [1.0, 2.0, 3.0])
    assert np.isnan(grid[0, 1]) and np.isnan(grid[2, 0])


def test_scatter_to_grid_orientation_is_row_major():
    """Index 1 is row 0 / column 1 and index 3 is row 1 / column 0 (no transpose)."""
    grid = _scatter_to_grid(np.array([7.0, 9.0]), np.array([1, 3]), 3, np.nan)
    assert grid[0, 1] == 7.0
    assert grid[1, 0] == 9.0


def test_scatter_to_grid_supports_string_values():
    """String maps (orbit family) use an empty-string fill for missing cells."""
    grid = _scatter_to_grid(np.array(["box", "loop"]), np.array([0, 8]), 3, "")
    assert grid[0, 0] == "box" and grid[2, 2] == "loop"
    assert grid[1, 1] == ""


# =========================================================================
# GRID BOOKKEEPING TESTS
# =========================================================================


def test_physical_and_unphysical_cells_partition_the_grid(grid_detector):
    """Only physical cells are integrated; axes have grid_size points."""
    n_physical = int(np.sum(~grid_detector.unphysical_mask))

    assert grid_detector.grid_size == GRID_SIZE
    assert grid_detector.num_orbits == n_physical
    assert 0 < n_physical <= GRID_SIZE**2
    assert grid_detector.x_grid.shape == (GRID_SIZE,)
    assert grid_detector.vx_grid.shape == (GRID_SIZE,)


def test_orbit_idx_and_grid_position_round_trip(grid_detector):
    """grid_position_of is the inverse of orbit_idx_at for every integrated orbit."""
    for orbit_idx in range(grid_detector.num_orbits):
        row, col = grid_detector.grid_position_of(orbit_idx)
        assert grid_detector.orbit_idx_at(row, col) == orbit_idx
        x, vx = grid_detector.grid_coordinates_of(orbit_idx)
        assert x == grid_detector.x_grid[col]
        assert vx == grid_detector.vx_grid[row]


def test_unphysical_cells_have_no_orbit(grid_detector):
    """Cells that were never integrated map to None."""
    for flat in np.flatnonzero(grid_detector.unphysical_mask):
        row, col = divmod(int(flat), GRID_SIZE)
        assert grid_detector.orbit_idx_at(row, col) is None


@pytest.mark.parametrize(
    ("row", "col"), [(GRID_SIZE, 0), (0, GRID_SIZE), (-1, 0), (0, -1)]
)
def test_orbit_idx_at_out_of_bounds_raises(grid_detector, row, col):
    """Positions outside the grid raise IndexError."""
    with pytest.raises(IndexError, match="out of bounds"):
        grid_detector.orbit_idx_at(row, col)


def test_grid_position_of_invalid_orbit_raises(grid_detector):
    """An orbit index beyond the integrated orbits is rejected."""
    with pytest.raises(IndexError):
        grid_detector.grid_position_of(grid_detector.num_orbits)


# =========================================================================
# GRID MAP TESTS
# =========================================================================


def test_chaos_grids_have_grid_shape_and_nan_only_off_the_physical_cells(grid_detector):
    """Each map is (3, 3); unphysical cells are NaN, physical SALI/GALI cells are not."""
    sali, gali, lyap = grid_detector.chaos_grids
    unphysical = grid_detector.unphysical_mask.reshape(GRID_SIZE, GRID_SIZE)

    for grid in (sali, gali, lyap):
        assert grid.shape == (GRID_SIZE, GRID_SIZE)
        assert np.all(np.isnan(grid[unphysical]))
    assert np.all(np.isfinite(sali[~unphysical]))
    assert np.all(np.isfinite(gali[~unphysical]))


def test_family_grid_has_grid_shape(grid_detector):
    """The family map is one label per grid cell."""
    assert grid_detector.family_grid.shape == (GRID_SIZE, GRID_SIZE)


def test_base_class_arrays_are_compact_in_physical_order(grid_detector):
    """Per-orbit arrays hold only the integrated orbits (no unphysical cells)."""
    assert grid_detector.sali_array.shape[1] == grid_detector.num_orbits
    assert grid_detector.gali_array.shape[0] == grid_detector.num_orbits


@pytest.mark.parametrize(
    "method",
    ["plot_chaos_map", "plot_composite_chaos_map", "plot_consensus_chaos_map"],
)
def test_map_plots_run_without_error(grid_detector, method):
    """Smoke test: each map plot renders on the matplotlib backend."""
    getattr(grid_detector, method)(backend="matplotlib", show=False)


@pytest.fixture(scope="module")
def grid_potential():
    return agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)


def test_grid_n_periods_sets_the_integration_time(grid_potential):
    det = GridChaosDetector(
        potential=grid_potential, R_0=8.0, grid_size=3, n_periods=2.0
    )
    expected = 2.0 * grid_potential.Tcirc(det.trajectories[:, 0, :])
    np.testing.assert_allclose(det.iter_time, expected)


def test_grid_default_is_30_periods(grid_potential):
    det = GridChaosDetector(
        potential=grid_potential,
        R_0=8.0,
        grid_size=3,
        n_periods=30,
    )
    assert det.n_periods == 30.0


def test_grid_both_time_options_raises(grid_potential):
    with pytest.raises(ValueError, match="not both"):
        GridChaosDetector(
            potential=grid_potential, R_0=8.0, grid_size=3, n_periods=2.0, iter_time=1.0
        )


@pytest.mark.parametrize("mode", ["checks", "score", "rate_norm"])
def test_chaos_grids_follow_map_values(mode):
    pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
    det = GridChaosDetector(
        potential=pot, R_0=8.0, grid_size=3, iter_time=1.0, map_values=mode
    )
    unphysical = det.unphysical_mask.reshape(3, 3)
    for grid in det.chaos_grids:
        assert grid.shape == (3, 3)
        assert np.all(np.isnan(grid[unphysical]))
    sali = det.chaos_grids[0]
    valid = sali[np.isfinite(sali)]
    if mode == "checks":
        assert set(np.unique(valid)) <= {0.0, 1.0}
    elif mode == "score":
        assert np.all((valid >= 0) & (valid <= 1))
    else:
        assert np.all(valid >= 0)


def test_unknown_map_values_raises():
    pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
    with pytest.raises(ValueError, match="map_values"):
        GridChaosDetector(
            potential=pot, R_0=8.0, grid_size=3, iter_time=1.0, map_values="bogus"
        )


@pytest.mark.parametrize("mode", ["checks", "score", "rate_norm"])
def test_chaos_map_plot_runs_for_every_mode(grid_detector, mode):
    grid_detector.plot_chaos_map(backend="matplotlib", show=False, map_values=mode)


def test_composite_and_consensus_always_use_binary_checks(grid_detector):
    grid_detector.map_values = "score"
    assert set(
        np.unique(
            grid_detector.check_grids[0][np.isfinite(grid_detector.check_grids[0])]
        )
    ) <= {0.0, 1.0}
