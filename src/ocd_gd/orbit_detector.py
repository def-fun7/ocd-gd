"""
Orbit Chaos Detection Module.

Provides tools for simulating orbital trajectories and evaluating chaotic behavior
via Small Alignment Index (SALI), Generalized Alignment Index (GALI), and
Lyapunov exponents.
"""

__all__ = ["OrbitChaosDetector"]

import time
from typing import Any

import agama
import numpy as np
import numpy.typing as npt
from astropy.table import QTable

from ._decay_classifier import (
    classify_decay,
    get_preset,
    is_chaotic,
    is_regular,
)
from ._family_check import FamilyStats, classify_box_loop, summarize_family
from ._plotting import _OrbitPlottingMixin
from ._sali_kernel import sali_kernel
from ._terminal_config import get_logger
from ._types import (
    _UNSET,
    LYAPUNOV_CHAOS_THRESHOLD,
    ChaosAgreement,
    ChaosSurveySummary,
    DecayClassification,
    IntegrationCriteria,
    MethodChaosStats,
)
from ._units import AgamaUnits, tag_unit

logger = get_logger(__name__)

_METADATA_FIELD_UNITS: dict[str, str | None] = {
    "num_orbits": None,
    "n_periods": None,
    "iter_time": "time",
    "traj_size": None,
    "accuracy": None,
    "max_num_steps": None,
}


class OrbitChaosDetector(_OrbitPlottingMixin):
    """Integrate orbits and analyze chaotic behavior using SALI/GALI indicators.

    Handles single or batch initial conditions seamlessly using vectorized
    computations and lazy evaluation properties. All `plot_*` methods are
    provided by `_OrbitPlottingMixin` (see `_plotting.py`); this class holds
    the integration, chaos-computation, family-classification, and lookup
    logic.

    Besides chaos detection (`detect_chaos`/`chaos_summary`), this class
    also exposes a cheap orbit-family label via `orbit_family`/
    `family_summary` (box vs. loop, from the sign of the in-plane angular
    momentum -- see `_family_check.py`), reusing the same trajectory data
    with no extra integration.

    Examples
    --------
    >>> import agama
    >>> from ocd_gd.orbit_detector import OrbitChaosDetector
    >>> agama.setUnits(mass=1, length=1, velocity=1)
    >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
    >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
    >>> detector = OrbitChaosDetector(ic=ic, pot=pot, iter_time=1.0)
    >>> detector.num_orbits
    1
    >>> detector.orbit_family
    array(['loop'], dtype='<U4')
    """

    def __init__(
        self,
        ic: Any,
        pot: Any,
        omega: float = 0.0,
        n_periods: float | None = _UNSET,
        iter_time: float | None = None,
        traj_size: int = 5000,
        accuracy: float = 1e-8,
        max_num_steps: int = 100000000,
        plotting_backend: str = "matplotlib",
        keep_raw_deviations: bool = False,
        units: AgamaUnits | None = None,
    ) -> None:
        """Initialize detector and automatically run orbit integrations.

        Parameters
        ----------
        ic : array_like
            Initial conditions for coordinates and velocities.
        pot : agama.Potential
            Agama gravitational potential object.
        omega : float, default 0.0
            Pattern speed of the rotating frame.
        n_periods : float or None, optional
            Number of orbital periods to integrate each orbit. Mutually
            exclusive with ``iter_time``; provide at most one. When
            ``n_periods`` is given, the actual integration time is computed
            per-orbit as ``n_periods * pot.Tcirc(ic)``. Defaults to 30.0
            when neither ``n_periods`` nor ``iter_time`` is provided.
        iter_time : float or None, default None
            Direct integration time in Agama time units. Mutually exclusive
            with ``n_periods``; if provided, ``n_periods`` is computed via
            ``iter_time / pot.Tcirc(ic)``.
        traj_size : int, default 5000
            if given, turns on the recording of trajectory for each orbit (should be either a single integer or an array of integers with length N). The trajectory of each orbit is stored either at every timestep of the integrator (if trajsize=0) or at regular intervals of time (`dt=abs(time)/(trajsize-1)`, so that the number of points is `trajsize`; the last stored point is always at the end of integration period, and if trajsize>1, the first point is the initial conditions).
        accuracy : float, default 1e-8
            Integration precision tracking for Agama.
        max_num_steps : int, default 100000000
            Safety boundary cap for maximum integration steps allowed.
        plotting_backend : str, default "matplotlib"
            Plotting library to use. Must be "matplotlib" or "plotly".
        keep_raw_deviations : bool, default False
            If True, preserve the raw (un-normalized) deviation vectors so they
            remain accessible via `self._dev_arr` after normalisation.
        units : AgamaUnits, optional
            Unit system to tag physical `metadata_row()` columns (`iter_time`,
            and in `GridChaosDetector`, `R_0`/`E_0`/`omega`/...) with real
            astropy units. If None (default), falls back to
            `AgamaUnits.current()`.

        Raises
        ------
        ValueError
            If both ``n_periods`` and ``iter_time`` are provided.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [[8.0, 0.0, 0.05, 0.0, 180.0, 5.0]]
        >>> detector = OrbitChaosDetector(ic=ic, pot=pot, iter_time=5.0)
        >>> isinstance(detector.pot, agama.Potential)
        True
        """
        if n_periods is not _UNSET and n_periods is not None and iter_time is not None:
            raise ValueError("Provide either 'n_periods' or 'iter_time', not both.")

        self.ic: npt.NDArray[np.float64] = np.atleast_2d(ic)
        self.pot: Any = pot
        self.omega: float = omega
        self.num_orbits: int = len(self.ic)

        if iter_time is not None:
            self.iter_time: float | npt.NDArray[np.float64] = (
                np.asarray(iter_time, dtype=np.float64)
                if np.iterable(iter_time)
                else float(iter_time)
            )
            dyn_time = self.pot.Tcirc(self.ic)
            self.n_periods: float | npt.NDArray[np.float64] = self.iter_time / dyn_time
        else:
            _n_periods = (
                30.0 if (n_periods is _UNSET or n_periods is None) else float(n_periods)
            )
            self.n_periods = _n_periods
            self.iter_time = self._compute_iter_time(_n_periods)

        self.traj_size: int = traj_size
        self.accuracy: float = accuracy
        self.max_num_steps: int = int(max_num_steps)
        self.plotting_backend: str = plotting_backend.lower()
        self.keep_raw_deviations: bool = keep_raw_deviations
        self.units: AgamaUnits | None = (
            units if units is not None else AgamaUnits.current()
        )

        self._time_arr: npt.NDArray[np.float64] | None = None
        self._traj_arr: npt.NDArray[np.float64] | None = None
        self._dev_arr: npt.NDArray[np.float64] | None = None
        self._lyap: npt.NDArray[np.float64] | None = None

        self._dev_arr_normalized: npt.NDArray[np.float64] | None = None
        self._sali_arr: npt.NDArray[np.float64] | None = None
        self._gali_arr: npt.NDArray[np.float64] | None = None
        self._chaos_results_cache: DecayClassification | None = None
        self._orbit_family_cache: npt.NDArray[np.str_] | None = None

        sali_kernel(np.array([[[[1]]]]), np.array([0]), np.array([1]))
        self._integrate_orbits()

    def _integrate_orbits(self) -> None:
        """Run the expensive orbit integration exactly once and cache results."""
        if isinstance(self.iter_time, np.ndarray) and self.iter_time.size > 1:
            iter_time_repr = f"[{self.iter_time.min():.4g}..{self.iter_time.max():.4g}]"
        else:
            iter_time_repr = f"{float(np.mean(self.iter_time)):.4g}"
        logger.info(
            "Integrating %d orbit(s) for iter_time=%s (accuracy=%.1e, max_num_steps=%d) ...",
            self.num_orbits,
            iter_time_repr,
            self.accuracy,
            self.max_num_steps,
        )
        start = time.perf_counter()
        orbit = agama.orbit(
            ic=self.ic,
            potential=self.pot,
            Omega=self.omega,
            time=self.iter_time,
            der=True,
            separateTime=True,
            trajsize=self.traj_size,
            lyapunov=True,
            accuracy=self.accuracy,
            maxNumSteps=self.max_num_steps,
            dtype="float64",
        )
        self._time_arr, self._traj_arr, self._dev_arr, self._lyap = orbit
        elapsed = time.perf_counter() - start
        logger.info(
            "Finished integrating %d orbit(s) in %.3fs", self.num_orbits, elapsed
        )

    def _compute_iter_time(self, n_periods: float) -> npt.NDArray[np.float64]:
        """Computes the iteration time for a number of orbits based on dynamical time.

        Parameters
        ----------
        n_periods : float
            Total number of orbital periods.

        Returns
        -------
        npt.NDArray[np.float64]
            An array of required iteration time for that number of orbits.
        """
        dyn_time = self.pot.Tcirc(self.ic)
        return n_periods * dyn_time

    def _normalize_deviation_vectors(self) -> npt.NDArray[np.float64]:
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
                self._dev_arr = dev

        np.nan_to_num(dev, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)

        max_vals = np.abs(dev).max(axis=-1, keepdims=True)
        np.divide(dev, np.where(max_vals == 0.0, 1.0, max_vals), out=dev)

        norm = np.linalg.norm(dev, axis=-1, keepdims=True)
        np.divide(dev, np.where(norm == 0.0, 1.0, norm), out=dev)

        return dev

    def _compute_sali(self) -> npt.NDArray[np.float64]:
        """Internal computation for smaller alignment index.

        Delegates to a Numba-jitted kernel that loops over the 15 (i, j)
        deviation-vector pairs directly, avoiding the (n, 15, timesteps, 6)
        fancy-indexed copies the numpy-vectorized version required.
        """
        logger.info("Computing SALI for %d orbit(s) ...", self.num_orbits)
        start = time.perf_counter()
        arr = np.ascontiguousarray(self.deviation_vectors)
        idx_i, idx_j = np.triu_indices(6, k=1)
        result = sali_kernel(arr, idx_i.astype(np.int64), idx_j.astype(np.int64))
        # Transpose from (n_orbits, 15, n_time) to (15, n_orbits, n_time)
        # so each of the 15 pairs is shaped (num_orbits, time_steps), matching GALI.
        result = np.ascontiguousarray(np.transpose(result, (1, 0, 2)))
        elapsed = time.perf_counter() - start
        logger.info(
            "Finished computing SALI for %d orbit(s) in %.3fs", self.num_orbits, elapsed
        )
        return result

    def _compute_gali(self) -> npt.NDArray[np.float64]:
        """Internal computation for generalized alignment index.

        GALI is the product of the singular values of the deviation-vector
        matrix at each timestep. For a square matrix, that product equals
        |det(matrix)|, so this uses np.linalg.det (LU-based) instead of a full
        SVD (compute_uv=False still does the full bidiagonalization + QR
        iteration under the hood) — same result, substantially less compute
        per matrix.
        """
        logger.info("Computing GALI for %d orbit(s) ...", self.num_orbits)
        start = time.perf_counter()
        matrix_a = np.transpose(self.deviation_vectors, (0, 2, 1, 3))
        result = np.abs(np.linalg.det(matrix_a))
        elapsed = time.perf_counter() - start
        logger.info(
            "Finished computing GALI for %d orbit(s) in %.3fs", self.num_orbits, elapsed
        )
        return result

    @property
    def criteria(self) -> IntegrationCriteria:
        """Get the integration and chaos indicator stopping criteria.

        Returns
        -------
        IntegrationCriteria
            The configuration parameters for orbit integration and evaluation.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=2.0)
        >>> detector.criteria.iter_time
        2.0
        """
        return IntegrationCriteria(
            iter_time=self.iter_time,
            traj_size=self.traj_size,
            accuracy=self.accuracy,
            max_num_steps=self.max_num_steps,
        )

    @property
    def timestamps(self) -> npt.NDArray[np.float64] | None:
        """Get the full integration time array.

        Returns
        -------
        ndarray or None
            Array of timestamps for the integrated trajectory.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.timestamps.ndim
        2
        """
        return self._time_arr

    @property
    def trajectories(self) -> npt.NDArray[np.float64] | None:
        """Get the integrated phase space trajectory paths.

        Returns
        -------
        ndarray or None
            Phase space trajectories for integrated orbits with shape
            (num_orbits, time_steps, 6).

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.trajectories.shape[2]
        6
        """
        return self._traj_arr

    @property
    def lyapunov_exponents(self) -> npt.NDArray[np.float64] | None:
        """Get calculated Lyapunov exponents for the system paths.

        Returns
        -------
        ndarray or None
            The Lyapunov exponents and corresponding times of shape
            (num_orbits, 2).

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.lyapunov_exponents.shape
        (1, 2)
        """
        return self._lyap

    @property
    def deviation_vectors(self) -> npt.NDArray[np.float64]:
        """Lazy-loaded property for normalized deviation vectors.

        Returns
        -------
        ndarray
            Normalized deviation vectors for the integrated orbits of shape
            (num_orbits, 6, time_steps, 6).

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.deviation_vectors.ndim
        4
        """
        if self._dev_arr_normalized is None:
            self._dev_arr_normalized = self._normalize_deviation_vectors()
        return self._dev_arr_normalized

    @property
    def sali_array(self) -> npt.NDArray[np.float64]:
        """Lazy-loaded property for the entire batch SALI matrix.

        Returns
        -------
        ndarray
            The Small Alignment Index values for each orbit over time,
            shaped (15, num_orbits, time_steps).

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.sali_array.shape[0]
        15
        """
        if self._sali_arr is None:
            self._sali_arr = self._compute_sali()
        return self._sali_arr

    @property
    def gali_array(self) -> npt.NDArray[np.float64]:
        """Lazy-loaded property for the entire batch GALI matrix.

        Returns
        -------
        ndarray
            The Generalized Alignment Index values for each orbit over time,
            shaped (num_orbits, time_steps).

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.gali_array.ndim
        2
        """
        if self._gali_arr is None:
            self._gali_arr = self._compute_gali()
        return self._gali_arr

    @property
    def orbit_family(self) -> npt.NDArray[np.str_]:
        """Lazy-loaded per-orbit box/loop classification.

        Classifies each orbit as `"loop"` if the orbit's in-plane angular
        momentum L_z(t) keeps one sign for the whole integration, or `"box"`
        if it flips sign (never settles into circulating one way).

        Cheap: reuses the already-integrated `self.trajectories`, no new
        integration. Coarse: only distinguishes box vs. loop.

        Returns
        -------
        ndarray
            An array of strings ("box" or "loop") of shape (num_orbits,).

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.orbit_family[0] in ("box", "loop")
        True
        """
        if self._orbit_family_cache is None:
            self._orbit_family_cache = classify_box_loop(self.trajectories)
        return self._orbit_family_cache

    def _validate_index(self, orbit_idx: int | None) -> None:
        """Ensure provided lookup index is within bounds."""
        if orbit_idx is not None and (orbit_idx < 0 or orbit_idx >= self.num_orbits):
            raise IndexError(
                f"Orbit index {orbit_idx} is out of bounds for "
                f"{self.num_orbits} integrated orbits."
            )

    def _resolve_backend(self, backend_override: str | None = None) -> str:
        """Determine backend priority: method argument > instance property."""
        chosen = backend_override if backend_override else self.plotting_backend
        chosen = chosen.lower()
        if chosen not in ("matplotlib", "plotly"):
            raise ValueError(
                f"Unsupported backend '{chosen}'. Must be 'matplotlib' or 'plotly'."
            )
        return chosen

    def _tag_unit(self, name: str, value: Any, lookup: dict[str, str | None]) -> Any:
        """Attach a physical unit to a raw Agama-unit metadata value."""
        return tag_unit(self.units, name, value, lookup)

    def get_trajectory(self, orbit_idx: int | None = None) -> npt.NDArray[np.float64]:
        """Return the full trajectory or specific targeted orbit index data.

        Parameters
        ----------
        orbit_idx : int, optional
            Specific orbit index to retrieve. If None, returns all trajectories.

        Returns
        -------
        ndarray
            Phase space trajectories.

        Raises
        ------
        IndexError
            If orbit_idx is out of bounds.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.get_trajectory(0).shape
        (5000, 6)
        """
        self._validate_index(orbit_idx)
        return self.trajectories if orbit_idx is None else self.trajectories[orbit_idx]

    def get_sali(self, orbit_idx: int | None = None) -> npt.NDArray[np.float64]:
        """Return SALI calculation sequences filtered down to target orbit.

        Parameters
        ----------
        orbit_idx : int, optional
            Specific orbit index to retrieve. If None, returns all SALI arrays
            of shape (15, num_orbits, time_steps).

        Returns
        -------
        ndarray
            SALI values over time. Shape (15, time_steps) for a single orbit,
            or (15, num_orbits, time_steps) for all orbits.

        Raises
        ------
        IndexError
            If orbit_idx is out of bounds.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.get_sali(0).shape
        (15, 5000)
        """
        self._validate_index(orbit_idx)
        if orbit_idx is None:
            return self.sali_array
        if self.sali_array.ndim == 3 and self.sali_array.shape[0] == 15:
            return self.sali_array[:, orbit_idx]
        return self.sali_array[orbit_idx]

    def get_gali(self, orbit_idx: int | None = None) -> npt.NDArray[np.float64]:
        """Return GALI calculation sequences filtered down to target orbit.

        Parameters
        ----------
        orbit_idx : int, optional
            Specific orbit index to retrieve. If None, returns all GALI arrays.

        Returns
        -------
        ndarray
            GALI values over time.

        Raises
        ------
        IndexError
            If orbit_idx is out of bounds.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.get_gali(0).shape
        (5000,)
        """
        self._validate_index(orbit_idx)
        return self.gali_array if orbit_idx is None else self.gali_array[orbit_idx]

    def get_family(
        self, orbit_idx: int | None = None
    ) -> npt.NDArray[np.str_] | np.str_:
        """Return the box/loop classification, optionally for one orbit.

        Parameters
        ----------
        orbit_idx : int, optional
            Specific orbit index to retrieve. If None, returns classifications for all orbits.

        Returns
        -------
        ndarray or str
            The classification string(s).

        Raises
        ------
        IndexError
            If orbit_idx is out of bounds.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> detector.get_family(0) in ("box", "loop")
        True
        """
        self._validate_index(orbit_idx)
        return self.orbit_family if orbit_idx is None else self.orbit_family[orbit_idx]

    def detect_chaos(
        self,
        orbit_idx: int | None = None,
        separate_sali: bool = False,
        sali_preset: str | None = "sali",
        gali_preset: str | None = "gali6_3dof",
        sali_kwargs: dict | None = None,
        gali_kwargs: dict | None = None,
    ) -> DecayClassification:
        """Classify each orbit as regular or chaotic via Bayesian decay-model selection.

        Runs ``classify_decay`` independently on the SALI and GALI time series,
        selecting among five decay laws (plateau, power law, exponential,
        plateau→exponential, plateau→power law) using BIC-weighted model
        averaging. Results are returned as a :class:`DecayClassification`
        holding one dict per indicator (see ``_decay_classifier.LABELS`` for
        the label convention: odd = regular, even > 0 = chaotic,
        0 = undetermined).

        The default run (no overrides) is cached so that repeated calls cost
        nothing.

        Parameters
        ----------
        orbit_idx : int, optional
            If given, returns results sliced to that single orbit. If ``None``
            (default), returns results for all integrated orbits.
        separate_sali : bool, default False
            If False, evaluates the pointwise minimum envelope across all 15
            SALI pairs, producing a single classification per orbit. If True,
            evaluates each of the 15 deviation-vector pairs individually,
            producing (15, num_orbits) results in the SALI dictionary.
        sali_preset : str or None, default ``"sali"``
            Key into ``_decay_classifier.PRESETS`` for SALI-specific defaults
            (``floor``, ``max_power_exponent``). Pass ``None`` to use the
            ``classify_decay`` module defaults; pass a key from ``PRESETS``
            to override those defaults. Any ``sali_kwargs`` are merged on top.
        gali_preset : str or None, default ``"gali6_3dof"``
            Key into ``_decay_classifier.PRESETS`` for GALI-specific defaults.
        sali_kwargs : dict, optional
            Extra keyword arguments forwarded to ``classify_decay`` for SALI,
            merged with (and taking precedence over) ``sali_preset`` defaults.
        gali_kwargs : dict, optional
            Extra keyword arguments forwarded to ``classify_decay`` for GALI.

        Returns
        -------
        DecayClassification
            Named tuple with fields ``sali`` and ``gali``, each a dict of
            arrays (length ``num_orbits``, or 1/sliced when ``orbit_idx`` is given)
            with keys: ``label``, ``score``, ``rate``, ``rate_norm``,
            ``rate_lb``, ``rate_is_bound``, ``delta_bic``, ``t_onset``,
            ``exponent``, ``t_floor``, ``n_eff``, ``p_plateau``, ``p_power``,
            ``p_exp``, ``p_exp_onset``, ``p_power_onset``.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> from ocd_gd import LABELS
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> report = detector.detect_chaos()
        >>> int(report.sali["label"][0]) in LABELS
        True
        """
        detect_chaos_start = time.perf_counter()
        self._validate_index(orbit_idx)

        is_default_config = (
            not separate_sali
            and sali_preset == "sali"
            and gali_preset == "gali6_3dof"
            and sali_kwargs is None
            and gali_kwargs is None
        )
        is_default_run = is_default_config and orbit_idx is None
        if is_default_config and orbit_idx is not None:
            return self._slice_classification(self.detect_chaos(), orbit_idx)

        if is_default_run and self._chaos_results_cache is not None:
            logger.info("Using cached decay-classifier results for detect_chaos")
            return self._chaos_results_cache

        s_kw: dict = get_preset(sali_preset)
        if sali_kwargs:
            s_kw.update(sali_kwargs)

        g_kw: dict = get_preset(gali_preset)
        if gali_kwargs:
            g_kw.update(gali_kwargs)

        t_orb = self.pot.Tcirc(self.ic)

        if not separate_sali:
            sali_ts = self.sali_array.min(axis=0)
            sali_result = classify_decay(sali_ts, self.timestamps, t_orb=t_orb, **s_kw)
        else:
            n_pairs, n_orb, n_t = self.sali_array.shape
            sali_flat = self.sali_array.reshape(n_pairs * n_orb, n_t)
            t_orb_expanded = (
                np.tile(np.asarray(t_orb), n_pairs) if t_orb is not None else None
            )
            time_expanded = (
                np.tile(self.timestamps, (n_pairs, 1))
                if np.ndim(self.timestamps) == 2
                else self.timestamps
            )
            raw_sali_out = classify_decay(
                sali_flat, time_expanded, t_orb=t_orb_expanded, **s_kw
            )
            sali_result = {
                k: v.reshape(n_pairs, n_orb) for k, v in raw_sali_out.items()
            }

        gali_result = classify_decay(
            self.gali_array, self.timestamps, t_orb=t_orb, **g_kw
        )

        lyap_array = self._lyap[:, 0]
        lyap_time = self._lyap[:, 1]
        is_nan = np.isnan(lyap_array)
        lyap_c = np.where(lyap_array <= LYAPUNOV_CHAOS_THRESHOLD, 0.0, 1.0)
        lyap_c[is_nan] = np.nan
        lyap_t = np.where(is_nan, np.inf, lyap_time)

        result = DecayClassification(
            sali=sali_result,
            gali=gali_result,
            lyapunov_check=lyap_c,
            lyapunov_time=lyap_t,
        )

        if is_default_run:
            self._chaos_results_cache = result

        if orbit_idx is not None:
            result = self._slice_classification(result, orbit_idx, separate_sali)
        logger.info(
            "detect_chaos completed in %.3fs (orbit_idx=%s, separate_sali=%s)",
            time.perf_counter() - detect_chaos_start,
            orbit_idx,
            separate_sali,
        )
        return result

    @staticmethod
    def _slice_classification(result, orbit_idx, separate_sali=False):
        def take(values, pairs_first):
            values = np.asarray(values)
            return (
                values[:, orbit_idx]
                if pairs_first and values.ndim == 2
                else values[orbit_idx]
            )

        return DecayClassification(
            sali={k: take(v, separate_sali) for k, v in result.sali.items()},
            gali={k: take(v, False) for k, v in result.gali.items()},
            lyapunov_check=result.lyapunov_check[orbit_idx],
            lyapunov_time=result.lyapunov_time[orbit_idx],
        )

    def _method_stats(self, labels: npt.NDArray[np.int_]) -> MethodChaosStats:
        """Build a MethodChaosStats block from a decay-classifier label array.

        ``labels`` follows the ``_decay_classifier.LABELS`` convention:
        - odd > 0  → regular
        - even > 0 → chaotic
        - 0        → undetermined
        """
        labels = np.asarray(labels, dtype=int)
        if labels.ndim == 2:
            sali_chaotic = np.any(is_chaotic(labels), axis=0)
            sali_regular = np.all(is_regular(labels), axis=0)
            labels = np.where(sali_chaotic, 2, np.where(sali_regular, 1, 0))

        chaotic_indices = np.where(is_chaotic(labels))[0]
        regular_indices = np.where(is_regular(labels))[0]
        undetermined_indices = np.where(labels == 0)[0]
        n_chaotic = len(chaotic_indices)
        n_regular = len(regular_indices)
        n_undetermined = len(undetermined_indices)
        n_total = self.num_orbits
        n_classified = n_total - n_undetermined

        return MethodChaosStats(
            n_chaotic=n_chaotic,
            n_regular=n_regular,
            n_undetermined=n_undetermined,
            n_total=n_total,
            chaotic_fraction=n_chaotic / n_classified if n_classified else float("nan"),
            chaotic_indices=chaotic_indices,
            regular_indices=regular_indices,
            undetermined_indices=undetermined_indices,
            chaotic_ics=self.ic[chaotic_indices],
            regular_ics=self.ic[regular_indices],
        )

    def chaos_summary(self, **detect_chaos_kwargs: Any) -> ChaosSurveySummary:
        """Summarize chaos classification across every integrated orbit.

        Derives per-indicator statistics from the decay-model labels produced
        by :meth:`detect_chaos`, plus Lyapunov-based classification, and
        computes pairwise and three-way agreement between all three indicators.

        Parameters
        ----------
        **detect_chaos_kwargs : dict
            Forwarded to :meth:`detect_chaos` — e.g. ``separate_sali``,
            ``sali_preset``, ``gali_preset``, ``sali_kwargs``, ``gali_kwargs``.

        Returns
        -------
        ChaosSurveySummary
            ``n_total``, one :class:`MethodChaosStats` per indicator
            (``.sali``, ``.gali``, ``.lyapunov``), and a
            :class:`ChaosAgreement` block.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> summary = detector.chaos_summary()
        >>> summary.n_total
        1
        """
        dc = self.detect_chaos(**detect_chaos_kwargs)

        sali_labels = np.asarray(dc.sali["label"])
        gali_labels = np.asarray(dc.gali["label"])

        if sali_labels.ndim == 2:
            sali_chaotic = np.any(is_chaotic(sali_labels), axis=0)
            sali_regular = np.all(is_regular(sali_labels), axis=0)
            orbit_sali_labels = np.where(sali_chaotic, 2, np.where(sali_regular, 1, 0))
        else:
            orbit_sali_labels = sali_labels

        sali_stats = self._method_stats(orbit_sali_labels)
        gali_stats = self._method_stats(gali_labels)

        lyap_array = self._lyap[:, 0]
        lyap_labels = np.zeros(self.num_orbits, dtype=int)
        lyap_undetermined = np.isnan(lyap_array)
        lyap_labels[~lyap_undetermined & (lyap_array > LYAPUNOV_CHAOS_THRESHOLD)] = (
            2  # chaotic (even)
        )
        lyap_labels[~lyap_undetermined & (lyap_array <= LYAPUNOV_CHAOS_THRESHOLD)] = (
            1  # regular (odd)
        )
        lyap_stats = self._method_stats(lyap_labels)

        sali_bool = is_chaotic(orbit_sali_labels)
        gali_bool = is_chaotic(gali_labels)
        lyap_bool = is_chaotic(lyap_labels)
        classified = ~lyap_undetermined

        all_agree = (sali_bool == gali_bool) & (gali_bool == lyap_bool) & classified

        agreement = ChaosAgreement(
            sali_gali_agreement=float(np.mean(sali_bool == gali_bool)),
            sali_lyapunov_agreement=float(
                np.mean((sali_bool == lyap_bool)[classified])
                if np.any(classified)
                else float("nan")
            ),
            gali_lyapunov_agreement=float(
                np.mean((gali_bool == lyap_bool)[classified])
                if np.any(classified)
                else float("nan")
            ),
            all_agree_chaotic=int(np.sum(all_agree & sali_bool)),
            all_agree_regular=int(np.sum(all_agree & ~sali_bool)),
            disagreement=int(np.sum(~all_agree & classified)),
            n_undetermined=int(np.sum(lyap_undetermined)),
        )

        return ChaosSurveySummary(
            n_total=self.num_orbits,
            sali=sali_stats,
            gali=gali_stats,
            lyapunov=lyap_stats,
            agreement=agreement,
        )

    def family_summary(self) -> FamilyStats:
        """Summarize box/loop classification across every integrated orbit.

        Computes counts, fractions, and indices per family (see `orbit_family`).

        Returns
        -------
        FamilyStats
            A statistics container with counts and indices for box and loop families.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> summary = detector.family_summary()
        >>> hasattr(summary, "n_loop")
        True
        """
        return summarize_family(self.orbit_family)

    def metadata_row(self, extra: dict[str, Any | None] | None = None) -> QTable:
        """Build a single-row astropy QTable of this run's integration/chaos parameters.

        Parameters
        ----------
        extra : dict, optional
            Additional columns to attach — e.g. potential-construction
            parameters (`bar_strength`, `bh_mass`) that this class has no
            way to introspect.

        Returns
        -------
        QTable
            A single-row table containing parameters and metadata.

        Examples
        --------
        >>> import agama
        >>> from ocd_gd.orbit_detector import OrbitChaosDetector
        >>> agama.setUnits(mass=1, length=1, velocity=1)
        >>> pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0)
        >>> ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]
        >>> detector = OrbitChaosDetector(ic, pot, iter_time=1.0)
        >>> row = detector.metadata_row(extra={"run_id": 42})
        >>> "run_id" in row.colnames
        True
        """
        criteria = self.criteria
        raw_columns: dict[str, Any] = {
            "num_orbits": self.num_orbits,
            "n_periods": self.n_periods,
            "iter_time": criteria.iter_time,
            "traj_size": criteria.traj_size,
            "accuracy": criteria.accuracy,
            "max_num_steps": criteria.max_num_steps,
        }
        columns: dict[str, Any] = {
            name: [self._tag_unit(name, value, _METADATA_FIELD_UNITS)]
            for name, value in raw_columns.items()
        }
        if extra:
            for key, value in extra.items():
                columns[key] = [value]
        return QTable(columns)
