"""
Orbit Chaos Detection Module.

Provides tools for simulating orbital trajectories and evaluating chaotic behavior
via Small Alignment Index (SALI), Generalized Alignment Index (GALI), and
Lyapunov exponents.
"""

from typing import Any, Optional, Tuple, Union
import numpy as np

import agama

from ._evaluate_chaos import evaluate_chaos
from ._sali_kernel import _sali_kernel
from ._plotting import _OrbitPlottingMixin
from .chaos_types import IntegrationCriteria, ChaosSummary, ChaosFullReport


class OrbitChaosDetector(_OrbitPlottingMixin):
    """Integrate orbits and analyze chaotic behavior using SALI/GALI indicators.

    Handles single or batch initial conditions seamlessly using vectorized
    computations and lazy evaluation properties. All `plot_*` methods are
    provided by `_OrbitPlottingMixin` (see `_plotting.py`); this class holds
    the integration, chaos-computation, and lookup logic.
    """

    def __init__(
        self,
        ic: Any,
        pot: Any,
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
        """Initialize detector and automatically run orbit integrations.

        Parameters
        ----------
        ic : array_like
            Initial conditions for coordinates and velocities.
        pot : agama.Potential
            Agama gravitational potential object.
        omega : float
            pattern speed of the rotating frame
        iter_time : float, default 10.0
            Total time duration for orbit integrations.
        gali_threshold : float, default 1e-16
            Threshold limits to register chaos in GALI calculations.
        sali_threshold : float, default 1e-2
            Threshold limits to register chaos in SALI calculations.
        gali_window_size : int, default 100
            The sliding window size required to confirm sustained convergence.
        sali_window_size : int, default 10
            The sliding window size required to confirm sustained convergence.
        accuracy : float, default 1e-8
            Integration precision tracking for Agama.
        max_num_steps : int, default 1e8
            Safety boundary cap for maximum integration steps allowed.
        plotting_backend: str, default "matplotlib"
            setup which plotting library to use.
        keep_raw_deviations : bool, default False
            If True, preserve the raw (un-normalized) deviation vectors so they
            remain accessible via `self._dev_arr` after normalisation — useful
            for plotting/diagnostics on small batches. If False (default),
            normalisation happens in place on `self._dev_arr` directly, avoiding
            an extra full-size array copy — recommended for large batches
            (thousands of orbits) where memory is the binding constraint.
        """

        # 1. Configuration Attributes
        self.ic: np.ndarray = np.atleast_2d(ic)
        self.pot: Any = pot
        self.omega: float = omega
        self.num_orbits: int = len(self.ic)

        self.iter_time: float = iter_time
        self.gali_threshold: float = gali_threshold
        self.sali_threshold: float = sali_threshold
        self.gali_window_size: int = gali_window_size
        self.sali_window_size: int = sali_window_size
        self.accuracy: float = accuracy
        self.max_num_steps: int = int(max_num_steps)
        self.plotting_backend: str = plotting_backend.lower()
        self.keep_raw_deviations: bool = keep_raw_deviations

        # 2. Raw Cached Simulation Data (Private)
        self._time_arr: Optional[np.ndarray] = None
        self._traj_arr: Optional[np.ndarray] = None
        self._dev_arr: Optional[np.ndarray] = None
        self._lyap: Optional[np.ndarray] = None

        # 3. Lazy Derived Attributes / Cache Layer
        self._dev_arr_normalized: Optional[np.ndarray] = None
        self._sali_arr: Optional[np.ndarray] = None
        self._gali_arr: Optional[np.ndarray] = None
        self._chaos_results_cache: Optional[Tuple[np.ndarray, ...]] = None

        # Automatically kick off the heavy simulation on creation
        _sali_kernel(np.array([[[[1]]]]), np.array([0]), np.array([1]))
        self._integrate_orbits()

    def _integrate_orbits(self) -> None:
        """Run the expensive orbit integration exactly once and cache results."""
        orbit = agama.orbit(
            ic=self.ic,
            potential=self.pot,
            Omega=self.omega,
            time=self.iter_time,
            der=True,
            separateTime=True,
            trajsize=5000,
            lyapunov=True,
            accuracy=self.accuracy,
            maxNumSteps=self.max_num_steps,
            dtype="float64",
        )
        self._time_arr, self._traj_arr, self._dev_arr, self._lyap = orbit

    def _normalize_deviation_vectors(self) -> np.ndarray:
        """Clean and unit-normalize deviation vectors safely.

        Behavior depends on `self.keep_raw_deviations`:
        - True: operates on a copy, leaving `self._dev_arr` untouched.
        - False (default): operates in place on `self._dev_arr`, saving one
        full-size array allocation — `self._dev_arr` is no longer usable
        afterward (it becomes the normalized array).
        """
        if self.keep_raw_deviations:
            dev = np.array(self._dev_arr, dtype=np.float64, copy=True)
        else:
            dev = self._dev_arr
            if dev.dtype != np.float64:
                dev = dev.astype(np.float64, copy=False)
                self._dev_arr = (
                    dev  # keep attribute consistent with what we're mutating
                )

        np.nan_to_num(dev, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)

        max_vals = np.abs(dev).max(axis=-1, keepdims=True)
        np.divide(dev, np.where(max_vals == 0.0, 1.0, max_vals), out=dev)

        norm = np.linalg.norm(dev, axis=-1, keepdims=True)
        np.divide(dev, np.where(norm == 0.0, 1.0, norm), out=dev)

        return dev

    def _compute_sali(self) -> np.ndarray:
        """Internal computation for smaller alignment index.

        Delegates to a Numba-jitted kernel that loops over the 15 (i, j)
        deviation-vector pairs directly, avoiding the (n, 15, timesteps, 6)
        fancy-indexed copies the numpy-vectorized version required.
        """
        arr = np.ascontiguousarray(self.deviation_vectors)
        idx_i, idx_j = np.triu_indices(6, k=1)
        return _sali_kernel(arr, idx_i.astype(np.int64), idx_j.astype(np.int64))

    def _compute_gali(self) -> np.ndarray:
        """Internal computation for generalized alignment index.

        GALI is the product of the singular values of the deviation-vector
        matrix at each timestep. For a square matrix, that product equals
        |det(matrix)|, so this uses np.linalg.det (LU-based) instead of a full
        SVD (compute_uv=False still does the full bidiagonalization + QR
        iteration under the hood) — same result, substantially less compute
        per matrix.
        """
        matrix_a = np.transpose(self.deviation_vectors, (0, 2, 1, 3))
        return np.abs(np.linalg.det(matrix_a))

    # =========================================================================
    # PUBLIC PROPERTIES
    # =========================================================================

    @property
    def criteria(self) -> IntegrationCriteria:
        """Get the integration and chaos indicator stopping criteria."""
        return IntegrationCriteria(
            iter_time=self.iter_time,
            gali_threshold=self.gali_threshold,
            sali_threshold=self.sali_threshold,
            gali_window_size=self.gali_window_size,
            sali_window_size=self.sali_window_size,
            accuracy=self.accuracy,
            max_num_steps=self.max_num_steps,
        )

    @property
    def timestamps(self) -> Optional[np.ndarray]:
        """Get the full integration time array."""
        return self._time_arr[0]

    @property
    def trajectories(self) -> Optional[np.ndarray]:
        """Get the integrated phase space trajectory paths."""
        return self._traj_arr

    @property
    def lyapunov_exponents(self) -> Optional[np.ndarray]:
        """Get calculated Lyapunov exponents for the system paths."""
        return self._lyap

    @property
    def deviation_vectors(self) -> np.ndarray:
        """Lazy-loaded property for normalized deviation vectors."""
        if self._dev_arr_normalized is None:
            self._dev_arr_normalized = self._normalize_deviation_vectors()
        return self._dev_arr_normalized

    @property
    def sali_array(self) -> np.ndarray:
        """Lazy-loaded property for the entire batch SALI matrix."""
        if self._sali_arr is None:
            self._sali_arr = self._compute_sali()
        return self._sali_arr

    @property
    def gali_array(self) -> np.ndarray:
        """Lazy-loaded property for the entire batch GALI matrix."""
        if self._gali_arr is None:
            self._gali_arr = self._compute_gali()
        return self._gali_arr

    # =========================================================================
    # PUBLIC ACCESS METHODS
    # =========================================================================

    def _validate_index(self, orbit_idx: Optional[int]) -> None:
        """Ensure provided lookup index is within bounds."""
        if orbit_idx is not None and (orbit_idx < 0 or orbit_idx >= self.num_orbits):
            raise IndexError(
                f"Orbit index {orbit_idx} is out of bounds for "
                f"{self.num_orbits} integrated orbits."
            )

    def _resolve_backend(self, backend_override: Optional[str] = None) -> str:
        """Determine backend priority: method argument > instance property."""
        chosen = backend_override if backend_override else self.plotting_backend
        chosen = chosen.lower()
        if chosen not in ("matplotlib", "plotly"):
            raise ValueError(
                f"Unsupported backend '{chosen}'. Must be 'matplotlib' or 'plotly'."
            )
        return chosen

    def get_trajectory(self, orbit_idx: Optional[int] = None) -> np.ndarray:
        """Return the full trajectory or specific targeted orbit index data."""
        self._validate_index(orbit_idx)
        return self.trajectories if orbit_idx is None else self.trajectories[orbit_idx]

    def get_sali(self, orbit_idx: Optional[int] = None) -> np.ndarray:
        """Return SALI calculation sequences filtered down to target orbit."""
        self._validate_index(orbit_idx)
        return self.sali_array if orbit_idx is None else self.sali_array[orbit_idx]

    def get_gali(self, orbit_idx: Optional[int] = None) -> np.ndarray:
        """Return GALI calculation sequences filtered down to target orbit."""
        self._validate_index(orbit_idx)
        return self.gali_array if orbit_idx is None else self.gali_array[orbit_idx]

    # =========================================================================
    # CORE ANALYSIS METHOD
    # =========================================================================

    def detect_chaos(
        self,
        orbit_idx: Optional[int] = None,
        separate_sali: bool = False,
        check_only: bool = True,
        sali_threshold_override: Optional[float] = None,
        gali_threshold_override: Optional[float] = None,
        sali_window_override: Optional[float] = None,
        gali_window_override: Optional[float] = None,
    ) -> Union[ChaosSummary, ChaosFullReport]:
        """Detect system deviations to distinguish chaotic from regular paths.

        Parameters
        ----------
        orbit_idx : int, optional
            A target index tracking one explicit orbit. Default extracts all.
        separate_sali : bool, default False
            Tracks cross evaluations individually if True (3D evaluation rule).
        check_only : bool, default True
            When True, provides a basic Summary data package. When False, wraps
            it inside a Full Diagnostic Report package.
        sali_threshold_override : float, optional
            Change the baseline SALI convergence check threshold parameter.
        gali_threshold_override : float, optional
            Change the baseline GALI convergence check threshold parameter.
        sali_window_override : float, optional
            Change the baseline SALI window size parameter.
        gali_window_override : float, optional
            Change the baseline GALI window size parameter.

        Returns
        -------
        Union[ChaosSummary, ChaosFullReport]
            The designated analysis container populated with convergence checks.
        """

        self._validate_index(orbit_idx)

        is_default_run = (
            (sali_threshold_override is None)
            and (gali_threshold_override is None)
            and (sali_window_override is None)
            and (gali_window_override is None)
        )

        if is_default_run and self._chaos_results_cache is not None:
            gali_check, gali_time, sali_check, sali_time = self._chaos_results_cache
        else:
            s_thresh = (
                sali_threshold_override
                if sali_threshold_override is not None
                else self.sali_threshold
            )
            g_thresh = (
                gali_threshold_override
                if gali_threshold_override is not None
                else self.gali_threshold
            )
            s_window = (
                sali_window_override
                if sali_window_override is not None
                else self.sali_window_size
            )
            g_window = (
                gali_window_override
                if gali_window_override is not None
                else self.gali_window_size
            )

            gali_check, gali_time = evaluate_chaos(
                self.gali_array,
                self.timestamps,
                threshold=g_thresh,
                window_size=g_window,
            )
            sali_check, sali_time = evaluate_chaos(
                self.sali_array,
                self.timestamps,
                threshold=s_thresh,
                separate=separate_sali,
                window_size=s_window,
            )

            if is_default_run:
                self._chaos_results_cache = (
                    gali_check,
                    gali_time,
                    sali_check,
                    sali_time,
                )

        lyap_array = self._lyap[:, 0]
        lyap_time = self._lyap[:, 1]
        lyap_check = np.where(lyap_array <= 0.1, 0, 1)

        if orbit_idx is not None:
            gali_c = gali_check[orbit_idx]
            gali_t = gali_time[orbit_idx]
            sali_c = sali_check[orbit_idx]
            sali_t = sali_time[orbit_idx]
            gali_d = self.gali_array[orbit_idx]
            sali_d = self.sali_array[orbit_idx]
            lyap_c = lyap_check[orbit_idx]
            lyap_t = lyap_time[orbit_idx]
            lyap_d = lyap_array[orbit_idx]
        else:
            gali_c, gali_t, sali_c, sali_t = (
                gali_check,
                gali_time,
                sali_check,
                sali_time,
            )
            lyap_c, lyap_t = (lyap_check, lyap_time)

            gali_d = self.gali_array
            sali_d = self.sali_array
            lyap_d = lyap_array

        summary_data = ChaosSummary(gali_c, gali_t, sali_c, sali_t, lyap_c, lyap_t)
        if check_only:
            return summary_data

        return ChaosFullReport(
            summary=summary_data,
            timestamps=self.timestamps,
            gali_array=gali_d,
            sali_array=sali_d,
            lyapunov_array=lyap_d,
        )
