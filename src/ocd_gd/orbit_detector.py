"""
Orbit Chaos Detection Module.

Provides tools for simulating orbital trajectories and evaluating chaotic behavior
via Small Alignment Index (SALI), Generalized Alignment Index (GALI), and
Lyapunov exponents.
"""

from typing import Any, NamedTuple, Optional, Tuple, Union
import agama
import numpy as np

from .evaluate_chaos import evaluate_chaos


class ChaosSummary(NamedTuple):
    """Structured container holding processed summary chaos classifications."""

    gali_check: np.ndarray
    gali_time: np.ndarray
    sali_check: np.ndarray
    sali_time: np.ndarray


class ChaosFullReport(NamedTuple):
    """Complete diagnostic bundle containing summaries alongside raw arrays."""

    summary: ChaosSummary
    timestamps: np.ndarray
    gali_array: np.ndarray
    sali_array: np.ndarray


class OrbitChaosDetector:
    """Integrate orbits and analyze chaotic behavior using SALI/GALI indicators.

    Handles single or batch initial conditions seamlessly using vectorized
    computations and lazy evaluation properties.
    """

    def __init__(
        self,
        ic: Any,
        pot: Any,
        omega: float = 0.0,
        iter_time: float = 10.0,
        gali_threshold: float = 1e-16,
        sali_threshold: float = 1e-8,
        window_size: int = 10,
        accuracy: float = 1e-8,
        max_num_steps: int = 100000000,
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
        sali_threshold : float, default 1e-8
            Threshold limits to register chaos in SALI calculations.
        window_size : int, default 10
            The sliding window size required to confirm sustained convergence.
        accuracy : float, default 1e-8
            Integration precision tracking for Agama.
        max_num_steps : int, default 1e8
            Safety boundary cap for maximum integration steps allowed.
        """
        # 1. Configuration Attributes
        self.ic: np.ndarray = np.atleast_2d(ic)
        self.pot: Any = pot
        self.omega: float = omega
        self.num_orbits: int = len(self.ic)

        self.iter_time: float = iter_time
        self.gali_threshold: float = gali_threshold
        self.sali_threshold: float = sali_threshold
        self.window_size: int = window_size
        self.accuracy: float = accuracy
        self.max_num_steps: int = max_num_steps

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
            trajsize=1000,
            lyapunov=True,
            accuracy=self.accuracy,
            maxNumSteps=self.max_num_steps,
        )
        self._time_arr, self._traj_arr, self._dev_arr, self._lyap = orbit

    def _normalize_deviation_vectors(self) -> np.ndarray:
        """Clean and unit-normalize deviation vectors safely."""
        dev_clean = np.nan_to_num(self._dev_arr, nan=0.0, posinf=1e30, neginf=-1e30)
        max_vals = np.abs(dev_clean).max(axis=-1, keepdims=True)
        max_vals_safe = np.where(max_vals == 0.0, 1.0, max_vals)
        scaled_dev = dev_clean / max_vals_safe

        scaled_norm = np.linalg.norm(scaled_dev, axis=-1, keepdims=True)
        scaled_norm_safe = np.where(scaled_norm == 0.0, 1.0, scaled_norm)

        return scaled_dev / scaled_norm_safe

    def _compute_sali(self) -> np.ndarray:
        """Internal computation for smaller alignment index."""
        arr = np.array(self.deviation_vectors, ndmin=4, copy=False)
        idx_i, idx_j = np.triu_indices(6, k=1)

        w1 = arr[:, idx_i, :, :]
        w2 = arr[:, idx_j, :, :]

        sum_norm = np.linalg.norm(w1 + w2, axis=-1)
        diff_norm = np.linalg.norm(w1 - w2, axis=-1)
        return np.minimum(sum_norm, diff_norm)

    def _compute_gali(self) -> np.ndarray:
        """Internal computation for generalized alignment index."""
        matrix_a = np.transpose(self.deviation_vectors, (0, 2, 1, 3))
        singular_values = np.linalg.svd(matrix_a, compute_uv=False)
        return np.prod(singular_values, axis=-1)

    # =========================================================================
    # PUBLIC PROPERTIES
    # =========================================================================

    @property
    def timestamps(self) -> Optional[np.ndarray]:
        """Get the full integration time array."""
        return self._time_arr

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
        sali_override: Optional[float] = None,
        gali_override: Optional[float] = None,
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
        sali_override : float, optional
            Change the baseline SALI convergence check threshold parameter.
        gali_override : float, optional
            Change the baseline GALI convergence check threshold parameter.

        Returns
        -------
        Union[ChaosSummary, ChaosFullReport]
            The designated analysis container populated with convergence checks.
        """
        self._validate_index(orbit_idx)

        # 1. Determine if we can use the default cache
        is_default_run = (sali_override is None) and (gali_override is None)

        if is_default_run and self._chaos_results_cache is not None:
            gali_check, gali_time, sali_check, sali_time = self._chaos_results_cache
        else:
            s_thresh = (
                sali_override if sali_override is not None else self.sali_threshold
            )
            g_thresh = (
                gali_override if gali_override is not None else self.gali_threshold
            )

            gali_check, gali_time = evaluate_chaos(
                self.gali_array,
                self.timestamps,
                threshold=g_thresh,
                window_size=self.window_size * 10,
            )
            sali_check, sali_time = evaluate_chaos(
                self.sali_array,
                self.timestamps,
                threshold=s_thresh,
                separate=separate_sali,
                window_size=self.window_size,
            )

            # Save to cache if it used the standard default settings
            if is_default_run:
                self._chaos_results_cache = (
                    gali_check,
                    gali_time,
                    sali_check,
                    sali_time,
                )

        # 2. Slice arrays for the requested orbit index after retrieval
        if orbit_idx is not None:
            gali_c = gali_check[orbit_idx]
            gali_t = gali_time[orbit_idx]
            sali_c = sali_check[orbit_idx]
            sali_t = sali_time[orbit_idx]
            gali_d = self.gali_array[orbit_idx]
            sali_d = self.sali_array[orbit_idx]
        else:
            gali_c, gali_t, sali_c, sali_t = (
                gali_check,
                gali_time,
                sali_check,
                sali_time,
            )
            gali_d = self.gali_array
            sali_d = self.sali_array

        summary_data = ChaosSummary(gali_c, gali_t, sali_c, sali_t)

        if check_only:
            return summary_data

        return ChaosFullReport(
            summary=summary_data,
            timestamps=self.timestamps,
            gali_array=gali_d,
            sali_array=sali_d,
        )
