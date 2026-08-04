"""
Grid-based orbit chaos detection.

Builds a 2D (x, v_x) grid of initial conditions at fixed energy and layers
grid-shaped chaos-map visualizations on top of OrbitChaosDetector.
"""

import time
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.table import QTable

from ._grid_ics import _circular_velocity, _generate_grid_ics, _reference_energy
from ._grid_plotting import _GridChaosPlottingMixin
from ._terminal_config import get_logger
from ._resonance import ResonanceRadii, compute_resonance_radii
from .orbit_detector import OrbitChaosDetector

logger = get_logger(__name__)


class GridChaosDetector(_GridChaosPlottingMixin, OrbitChaosDetector):
    """Build a 2D (x, v_x) grid of initial conditions at fixed energy and
    detect/visualize chaos across the grid as a spatial map.

    Subclasses `OrbitChaosDetector`: once the grid of initial conditions is
    generated, integration and chaos detection proceed exactly as in the
    parent class — `detect_chaos`, `get_sali`, `plot_sali`, etc. all work
    unchanged, indexed by the flattened grid order. This class adds the
    grid-generation step plus `plot_chaos_map` / `plot_composite_chaos_map` /
    `save_chaos_maps` (see `_GridChaosPlottingMixin` in `_grid_plotting.py`).
    """

    def __init__(
        self,
        potential: Any,
        R_0: float = 8.0,
        y_0: float = 1e-4,
        z_0: float = 0.1,
        v_y0_frac: float = 0.2,
        v_z0_frac: float = 0.02,
        E_0: float | None = None,
        grid_size: int = 10,
        x_search_range: tuple[float, float] = (-10.0, 10.0),
        search_resolution: int = 1000,
        omega: float = 0.0,
        iter_time: float = 10.0,
        gali_threshold: float = 1e-20,
        sali_threshold: float = 1e-3,
        gali_window_size: int = 50,
        sali_window_size: int = 25,
        accuracy: float = 1e-8,
        max_num_steps: int = 100000000,
        plotting_backend: str = "matplotlib",
        keep_raw_deviations: bool = False,
    ) -> None:
        """Generate a grid of initial conditions at fixed energy, then
        integrate and analyze them via `OrbitChaosDetector`.

        Parameters
        ----------
        potential : agama.Potential
            Agama gravitational potential object. Its units must already be
            set via `agama.setUnits(...)` before constructing this detector.
        R_0 : float, default 8.0
            Reference radius used to derive the local circular velocity —
            which converts `v_y0_frac`/`v_z0_frac` into actual transverse
            velocities — and, unless `E_0` is given explicitly, the grid's
            total energy.
        y_0, z_0 : float, default 1e-4, 0.1
            Fixed transverse position coordinates for every grid orbit.
            Cannot both be zero.
        v_y0_frac, v_z0_frac : float, default 0.2, 0.02
            Transverse velocities as a fraction of the local circular
            velocity at `R_0`. Individually capped and jointly capped in
            quadrature — see `_validate_grid_params` in `_grid_ics.py` for
            the exact limits and rationale.
        E_0 : float, optional
            Total energy defining the accessible (x, v_x) region. If None
            (default), it's derived from a circular orbit at `R_0`. Passing
            this explicitly decouples the grid's energy from `R_0`'s
            circular velocity — `R_0` is still used to set `v_y0`/`v_z0`
            from their fractions.
        grid_size : int, default 10
            Number of grid points per axis (grid_size x grid_size orbits
            total).
        x_search_range : tuple of float, default (-10.0, 10.0)
            Range to scan when locating the physical x turning points at E_0.
        search_resolution : int, default 1000
            Number of scan points used to locate the turning points.
        omega, iter_time, gali_threshold, sali_threshold, gali_window_size,
        sali_window_size, accuracy, max_num_steps, plotting_backend,
        keep_raw_deviations :
            Forwarded to `OrbitChaosDetector.__init__` — see its docstring
            for details.
        """
        logger.info(
            "Generating %dx%d (x, v_x) grid of initial conditions at R_0=%.4g ...",
            grid_size,
            grid_size,
            R_0,
        )
        grid_gen_start = time.perf_counter()
        grid_info = _generate_grid_ics(
            potential=potential,
            R_0=R_0,
            y_0=y_0,
            z_0=z_0,
            v_y0_frac=v_y0_frac,
            v_z0_frac=v_z0_frac,
            E_0=E_0,
            x_search_range=x_search_range,
            grid_size=grid_size,
            search_resolution=search_resolution,
        )
        logger.info(
            "Finished generating grid in %.3fs", time.perf_counter() - grid_gen_start
        )

        self.grid_size = grid_size
        self.R_0 = R_0
        self.E_0 = grid_info.E_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.v_y0_frac = v_y0_frac
        self.v_z0_frac = v_z0_frac
        self.unphysical_mask = grid_info.unphysical_mask
        self.x_grid = grid_info.x_vals
        self.vx_grid = grid_info.v_x_vals
        self.energy_remainder = grid_info.E_rem_vals
        self._chaos_grids_cache: tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64] | None,
        ] = None
        self._orbit_idx_lookup_cache: npt.NDArray[np.float64] | None = None
        self._resonance_radii_cache: ResonanceRadii | None = None

        # Only physically valid grid cells get integrated — skips wasted
        # agama.orbit() work on cells already known to violate the energy
        # constraint. `_physical_indices` records each integrated orbit's
        # original flat grid position, so chaos results can be scattered
        # back into (grid_size, grid_size) shape later.
        self._physical_indices = np.where(~grid_info.unphysical_mask)[0]

        n_total_cells = grid_size**2
        n_physical = len(self._physical_indices)
        logger.info(
            "%d/%d grid cells are physical at E_0=%.6g (%d unphysical, skipped)",
            n_physical,
            n_total_cells,
            self.E_0,
            n_total_cells - n_physical,
        )

        super().__init__(
            ic=grid_info.ics[self._physical_indices],
            pot=potential,
            omega=omega,
            iter_time=iter_time,
            gali_threshold=gali_threshold,
            sali_threshold=sali_threshold,
            gali_window_size=gali_window_size,
            sali_window_size=sali_window_size,
            accuracy=accuracy,
            max_num_steps=max_num_steps,
            plotting_backend=plotting_backend,
            keep_raw_deviations=keep_raw_deviations,
        )

    def _compute_chaos_grids(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Reshape SALI/GALI/Lyapunov chaos checks into (grid_size, grid_size)
        maps, masking unphysical grid cells as NaN.

        `detect_chaos()` only ever ran on the physical orbits (see
        `_physical_indices` in `__init__`), so its check arrays are compact —
        one entry per integrated orbit, not per grid cell. This scatters them
        back into full grid_size**2-length arrays at their original flat grid
        positions before reshaping, leaving every unphysical cell as NaN.

        No transpose is applied: `_generate_grid_ics` builds `ics` from
        `np.meshgrid(x_vals, v_x_vals)` (default 'xy' indexing), which already
        lays the flattened array out as row = v_x index, column = x index —
        exactly the (row=y, col=x) orientation `imshow`/`Heatmap`/`Image`
        expect. Transposing here would rotate the map 90 degrees relative to
        the x-axis-indexed zero-velocity curve overlay.
        """
        summary = self.detect_chaos()
        n_cells = self.grid_size**2

        sali = np.full(n_cells, np.nan)
        gali = np.full(n_cells, np.nan)
        lyap = np.full(n_cells, np.nan)

        sali[self._physical_indices] = summary.sali_check
        gali[self._physical_indices] = summary.gali_check
        lyap[self._physical_indices] = summary.lyapunov_check

        shape = (self.grid_size, self.grid_size)
        return (
            sali.reshape(shape),
            gali.reshape(shape),
            lyap.reshape(shape),
        )

    @property
    def chaos_grids(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Lazy-loaded (sali_grid, gali_grid, lyapunov_grid) maps, each shaped
        (grid_size, grid_size) with unphysical cells set to NaN."""
        if self._chaos_grids_cache is None:
            self._chaos_grids_cache = self._compute_chaos_grids()
        return self._chaos_grids_cache

    @property
    def _orbit_idx_lookup(self) -> npt.NDArray[np.float64]:
        """Lazy-loaded (grid_size**2,) int array mapping a flat grid index to
        its orbit_idx, or -1 for grid cells that were never integrated
        (unphysical)."""
        if self._orbit_idx_lookup_cache is None:
            lookup = np.full(self.grid_size**2, -1, dtype=int)
            lookup[self._physical_indices] = np.arange(len(self._physical_indices))
            self._orbit_idx_lookup_cache = lookup
        return self._orbit_idx_lookup_cache

    def orbit_idx_at(self, row: int, col: int) -> int | None:
        """Return the orbit_idx for grid cell (row, col), or None if that
        cell was unphysical and was never integrated.

        `row` indexes v_x and `col` indexes x — matching `chaos_grids`' axes
        (`chaos_grids[0][row, col]` is the SALI check for the orbit this
        returns). Use the returned orbit_idx with any inherited orbit_idx-
        based method: `get_sali`, `get_gali`, `get_trajectory`, `plot_sali`,
        `plot_gali`, `detect_chaos(orbit_idx=...)`, etc.
        """
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            raise IndexError(
                f"Grid position ({row}, {col}) is out of bounds for a "
                f"{self.grid_size}x{self.grid_size} grid."
            )
        flat = row * self.grid_size + col
        idx = self._orbit_idx_lookup[flat]
        return None if idx == -1 else int(idx)

    def grid_position_of(self, orbit_idx: int) -> tuple[int, int]:
        """Return the (row, col) grid position an orbit_idx corresponds to —
        the inverse of `orbit_idx_at`. `row` indexes v_x, `col` indexes x."""
        self._validate_index(orbit_idx)
        flat = self._physical_indices[orbit_idx]
        return divmod(int(flat), self.grid_size)

    def grid_coordinates_of(self, orbit_idx: int) -> tuple[float, float]:
        """Return the (x, v_x) physical coordinates an orbit_idx corresponds
        to, rather than its raw (row, col) grid position."""
        row, col = self.grid_position_of(orbit_idx)
        return float(self.x_grid[col]), float(self.vx_grid[row])

    @property
    def resonance_radii(self) -> ResonanceRadii:
        """Lazy-loaded corotation/inner/outer Lindblad radii for this
        detector's potential at its pattern speed `self.omega`. Used by
        `plot_chaos_map`/`plot_composite_chaos_map` to overlay reference
        lines; any radius is None if `omega == 0` or no root was found."""
        if self._resonance_radii_cache is None:
            self._resonance_radii_cache = compute_resonance_radii(self.pot, self.omega)
        return self._resonance_radii_cache

    def metadata_row(self, extra: dict[str, Any | None] | None = None) -> QTable:
        """Extends `OrbitChaosDetector.metadata_row` with grid-specific
        parameters (R_0, E_0, grid geometry, transverse velocity fractions).
        See the base method for the `extra` parameter and general behavior.
        """
        grid_columns: dict[str, Any] = {
            "R_0": self.R_0,
            "E_0": self.E_0,
            "y_0": self.y_0,
            "z_0": self.z_0,
            "v_y0_frac": self.v_y0_frac,
            "v_z0_frac": self.v_z0_frac,
            "grid_size": self.grid_size,
            "omega": self.omega,
        }
        merged_extra = {**grid_columns, **(extra or {})}
        return super().metadata_row(extra=merged_extra)

    @staticmethod
    def circular_velocity(potential: Any, R_0: float) -> float:
        """Local circular velocity magnitude at (R_0, 0, 0) from the
        potential's radial force — the same helper used internally to turn
        `v_y0_frac`/`v_z0_frac` into actual velocities. Exposed for
        inspection (e.g. checking what velocity a given R_0 implies) without
        needing to construct a detector first.
        """
        return _circular_velocity(potential, R_0)

    @staticmethod
    def reference_energy(potential: Any, R_0: float) -> float:
        """Total energy of a circular orbit at (R_0, 0, 0) — the same value
        `GridChaosDetector` derives internally when `E_0` is left as None.
        Exposed for inspection ahead of constructing a detector.
        """
        v_circ = _circular_velocity(potential, R_0)
        return _reference_energy(potential, R_0, v_circ)
