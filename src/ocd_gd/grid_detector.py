"""
Grid Chaos Detector Module.

Generates a dense (x, v_x) phase-space grid over a constant energy surface
E_0, integrates the orbital trajectories, and computes SALI, GALI, and Lyapunov
maps across the grid.
"""

from typing import Any, Tuple, Optional
import numpy as np

from ._grid_ics import _generate_grid_ics
from ._grid_plotting import _GridChaosPlottingMixin
from .orbit_detector import OrbitChaosDetector


class GridChaosDetector(_GridChaosPlottingMixin, OrbitChaosDetector):
    """Generate a phase-space initial-condition grid and compute chaos maps.

    Combines automatic grid generation on a constant energy surface with the
    orbital integration and chaos-detection infrastructure of `OrbitChaosDetector`.
    Unphysical grid cells (where local kinetic energy would be negative) are
    automatically masked out with `np.nan`.

    Inherits all visualization utilities (`plot_chaos_map`,
    `plot_composite_chaos_map`, `save_chaos_maps`) from
    `_GridChaosPlottingMixin`.
    """

    def __init__(
        self,
        potential: Any,
        E_0: float,
        y_0: float = 10.0,
        z_0: float = 10.0,
        v_y0: float = 10.0,
        v_z0: float = 10.0,
        grid_size: int = 10,
        x_search_range: Tuple[float, float] = (-10.0, 10.0),
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
        """Generate grid ICs and run orbit integrations across the grid.

        Parameters
        ----------
        potential : agama.Potential
            Agama gravitational potential object.
        E_0 : float
            Fixed total energy defining the accessible (x, v_x) surface.
        y_0, z_0 : float, default 0.0
            Fixed transverse position coordinates.
        v_y0, v_z0 : float, default 0.0
            Fixed transverse velocity components.
        grid_size : int, default 10
            Grid dimensions along each axis (grid_size x grid_size total orbits).
        x_search_range : tuple of float, default (-10.0, 10.0)
            Search bounds to scan for turning points at energy E_0.
        search_resolution : int, default 1000
            Number of points used during the turning-point search scan.
        omega : float, default 0.0
            Pattern speed of the rotating frame.
        iter_time : float, default 10.0
            Total time duration for orbit integrations.
        gali_threshold : float, default 1e-20
            Threshold limit to register chaos in GALI.
        sali_threshold : float, default 1e-3
            Threshold limit to register chaos in SALI.
        gali_window_size : int, default 50
            Sliding window size for GALI convergence.
        sali_window_size : int, default 25
            Sliding window size for SALI convergence.
        accuracy : float, default 1e-8
            Integration precision tracking for Agama.
        max_num_steps : int, default 100000000
            Maximum integration steps allowed per orbit.
        plotting_backend : str, default "matplotlib"
            Default plotting backend ('matplotlib' or 'plotly').
        keep_raw_deviations : bool, default False
            If True, preserve un-normalized deviation vectors in memory.
        """
        # 1. Store Grid Specific Configuration
        self.E_0: float = E_0
        self.y_0: float = y_0
        self.z_0: float = z_0
        self.v_y0: float = v_y0
        self.v_z0: float = v_z0
        self.grid_size: int = grid_size

        # 2. Generate Initial Conditions Matrix & Mask Metadata
        grid_ics = _generate_grid_ics(
            potential=potential,
            E_0=E_0,
            y_0=y_0,
            z_0=z_0,
            v_y0=v_y0,
            v_z0=v_z0,
            x_search_range=x_search_range,
            grid_size=grid_size,
            search_resolution=search_resolution,
        )
        self.ics_grid = grid_ics.ics
        self.x_grid: np.ndarray = grid_ics.x_vals
        self.vx_grid: np.ndarray = grid_ics.v_x_vals
        self.energy_remainder: np.ndarray = grid_ics.E_rem_vals
        self.unphysical_mask: np.ndarray = grid_ics.unphysical_mask

        # 3. Initialize Parent Class (Triggers Integration Automatically)
        super().__init__(
            ic=grid_ics.ics,
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

        # 4. Mask Unphysical Grid Points in Trajectory/Lyapunov Cache
        self._apply_unphysical_mask()

    def _apply_unphysical_mask(self) -> None:
        """Apply NaN mask to unphysical trajectories in raw array caches."""
        if not np.any(self.unphysical_mask):
            return

        if self._traj_arr is not None:
            self._traj_arr[self.unphysical_mask] = np.nan
        if self._lyap is not None:
            self._lyap[self.unphysical_mask] = np.nan

    # =========================================================================
    # GRID SPECIFIC PROPERTIES & RESHAPED ACCESSORS
    # =========================================================================
    @property
    def grid_ics(self) -> np.ndarray:
        """The grid intital conditions"""
        return self.ics_grid

    @property
    def sali_grid(self) -> np.ndarray:
        """Reshape final SALI chaos classifications into a 2D (grid_size, grid_size) map."""
        summary = self.detect_chaos(check_only=True)
        sali_check = np.array(summary.sali_check, dtype=float, copy=True)
        sali_check[self.unphysical_mask] = np.nan
        return sali_check.reshape((self.grid_size, self.grid_size)).T

    @property
    def gali_grid(self) -> np.ndarray:
        """Reshape final GALI chaos classifications into a 2D (grid_size, grid_size) map."""
        summary = self.detect_chaos(check_only=True)
        gali_check = np.array(summary.gali_check, dtype=float, copy=True)
        gali_check[self.unphysical_mask] = np.nan
        return gali_check.reshape((self.grid_size, self.grid_size)).T

    @property
    def lyapunov_grid(self) -> np.ndarray:
        """Reshape final Lyapunov values into a 2D (grid_size, grid_size) map."""
        summary = self.detect_chaos(check_only=True)
        lyap_check = np.array(summary.lyapunov_check, dtype=float, copy=True)
        lyap_check[self.unphysical_mask] = np.nan
        return lyap_check.reshape((self.grid_size, self.grid_size)).T

    @property
    def chaos_grids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bundle tuple of (sali_grid, gali_grid, lyapunov_grid) expected by _GridChaosPlottingMixin."""
        return self.sali_grid, self.gali_grid, self.lyapunov_grid

    @property
    def sali_time_series_grid(self) -> np.ndarray:
        """3D (grid_size, grid_size, n_steps) array for full SALI time evolution."""
        sali = np.array(self.sali_array, copy=True)
        sali[self.unphysical_mask] = np.nan
        return sali.reshape((self.grid_size, self.grid_size, -1))

    @property
    def gali_time_series_grid(self) -> np.ndarray:
        """3D (grid_size, grid_size, n_steps) array for full GALI time evolution."""
        gali = np.array(self.gali_array, copy=True)
        gali[self.unphysical_mask] = np.nan
        return gali.reshape((self.grid_size, self.grid_size, -1))

    # =========================================================================
    # INDEX HELPERS (2D Grid <-> 1D Orbit Index)
    # =========================================================================

    def grid_to_orbit_idx(self, i: int, j: int) -> int:
        """Convert 2D grid index (row i, column j) to flattened orbit index."""
        if not (0 <= i < self.grid_size and 0 <= j < self.grid_size):
            raise IndexError(
                f"Grid indices ({i}, {j}) out of bounds for grid size {self.grid_size}."
            )
        return i * self.grid_size + j

    def orbit_to_grid_idx(self, orbit_idx: int) -> Tuple[int, int]:
        """Convert flattened orbit index to 2D grid index (row i, column j)."""
        self._validate_index(orbit_idx)
        return divmod(orbit_idx, self.grid_size)
