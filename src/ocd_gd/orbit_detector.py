"""
Orbit Chaos Detection Module.

Provides tools for simulating orbital trajectories and evaluating chaotic behavior
via Small Alignment Index (SALI), Generalized Alignment Index (GALI), and
Lyapunov exponents.
"""

from typing import Any, NamedTuple, Optional, Tuple, Union, List
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import agama

from .evaluate_chaos import evaluate_chaos
from .visualisation import (
    plot_sali_mpl,
    plot_sali_plotly,
    plot_gali_mpl,
    plot_gali_plotly,
    plot_trajectory_2d_mpl,
    plot_trajectory_2d_plotly,
    plot_trajectory_3d_mpl,
    plot_trajectory_3d_plotly,
    plot_phase_space_mpl,
    plot_phase_space_plotly,
    plot_energy_drift_mpl,
    plot_energy_drift_plotly,
    plot_colored_trajectory_2d_mpl,
    plot_colored_trajectory_2d_plotly,
    plot_dashboard_mpl,
    plot_dashboard_plotly,
    plot_gali_batch_mpl,
    plot_sali_batch_mpl,
    plot_sali_gali_dual_batch_mpl,
)


@dataclass(frozen=True)
class IntegrationCriteria:
    iter_time: float
    gali_threshold: float
    sali_threshold: float
    gali_window_size: int
    sali_window_size: int
    accuracy: float
    max_num_steps: int


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
        gali_threshold: float = 1e-20,
        sali_threshold: float = 1e-3,
        gali_window_size: int = 100,
        sali_window_size: int = 10,
        accuracy: float = 1e-8,
        max_num_steps: int = 100000000,
        plotting_backend: str = "matplotlib",
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
            trajsize=5000,
            lyapunov=True,
            accuracy=self.accuracy,
            maxNumSteps=self.max_num_steps,
            dtype="float64",
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

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def plot_sali(
        self,
        orbit_idx: int = 0,
        all_pairs: bool = False,
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, plt.Axes], go.Figure]:
        """Plot SALI vs Time for a target orbit.

        Parameters
        ----------
        orbit_idx : int, default 0
            Target orbit index.
        all_pairs : bool, default False
            If True and separate SALI pairs are computed, plots all 15 vector pair
            traces. If False, plots the minimum SALI envelope.
        backend : str, optional
            'matplotlib' or 'plotly' (overrides default).
        save_path : str, optional
            Path to export figure.
        show : bool, default True
            Display figure immediately.
        """
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        chaos_report = self.detect_chaos(orbit_idx=orbit_idx, check_only=True)

        # 1. Safely extract boolean check (0 = Regular, 1 = Chaotic)
        s_check = chaos_report.sali_check
        is_chaotic = bool(np.any(s_check)) if np.ndim(s_check) > 0 else bool(s_check)

        # 2. Safely extract scalar timestamp without triggering TypeError on arrays/inf
        s_time = chaos_report.sali_time
        if np.ndim(s_time) > 0:
            # If array has elements, extract first value or min finite time
            det_time = float(s_time.flat[0])
        else:
            det_time = float(s_time)

        # Approximate sliding window duration in time units
        dt = self.timestamps[1] - self.timestamps[0]
        window_time = self.sali_window_size * dt

        sali_data = np.squeeze(self.sali_array[orbit_idx])
        if sali_data.ndim > 1 and not all_pairs:
            sali_data = np.min(sali_data, axis=0)
        lyap_data = (
            self.lyapunov_exponents[orbit_idx]
            if self.lyapunov_exponents is not None
            else None
        )
        plot_fn = plot_sali_mpl if engine == "matplotlib" else plot_sali_plotly

        return plot_fn(
            t=self.timestamps,
            sali=sali_data,
            threshold=self.sali_threshold,
            is_chaotic=is_chaotic,
            detection_time=det_time,
            window_size_time=window_time,
            lyapunov=lyap_data,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_gali(
        self,
        orbit_idx: int = 0,
        k_orders: Optional[List[int]] = None,
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ):
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        # Extract chaos report & detection time
        chaos_report = self.detect_chaos(orbit_idx=orbit_idx, check_only=True)
        g_check = chaos_report.gali_check
        is_chaotic = bool(np.any(g_check)) if np.ndim(g_check) > 0 else bool(g_check)

        g_time = chaos_report.gali_time
        det_time = float(g_time.flat[0]) if np.ndim(g_time) > 0 else float(g_time)

        dt = self.timestamps[1] - self.timestamps[0]
        window_time = self.gali_window_size * dt

        # Extract orbit-specific data
        gali_data = np.squeeze(self.gali_array[orbit_idx])
        lyap_data = (
            self.lyapunov_exponents[orbit_idx]
            if self.lyapunov_exponents is not None
            else None
        )

        if engine == "matplotlib":
            return plot_gali_mpl(
                t=self.timestamps,
                gali=gali_data,
                k_orders=k_orders,
                threshold=self.gali_threshold,
                is_chaotic=is_chaotic,
                detection_time=det_time if np.isfinite(det_time) else None,
                window_size_time=window_time,
                lyapunov=lyap_data,
                save_path=save_path,
                show=show,
                **kwargs,
            )
        else:
            return plot_gali_plotly(...)

    def plot_trajectory_2d(
        self,
        orbit_idx: int = 0,
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, np.ndarray], go.Figure]:
        """Plot Face-On (X-Y) and Edge-On (X-Z) 2D orbit projections."""
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        pos = self.trajectories[orbit_idx][:, :3]

        if engine == "matplotlib":
            return plot_trajectory_2d_mpl(pos, save_path=save_path, show=show, **kwargs)
        else:
            return plot_trajectory_2d_plotly(
                pos, save_path=save_path, show=show, **kwargs
            )

    def plot_trajectory_3d(
        self,
        orbit_idx: int = 0,
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, plt.Axes], go.Figure]:
        """Plot 3D spatial orbit path."""
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        pos = self.trajectories[orbit_idx][:, :3]

        if engine == "matplotlib":
            return plot_trajectory_3d_mpl(pos, save_path=save_path, show=show, **kwargs)
        else:
            return plot_trajectory_3d_plotly(
                pos, save_path=save_path, show=show, **kwargs
            )

    def plot_phase_space(
        self,
        orbit_idx: int = 0,
        plane: str = "x",
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, plt.Axes], go.Figure]:
        """Plot 2D phase space scatter projection (Position vs Velocity)."""
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        pos = self.trajectories[orbit_idx][:, :3]
        vel = self.trajectories[orbit_idx][:, 3:6]

        if engine == "matplotlib":
            return plot_phase_space_mpl(
                pos, vel, plane=plane, save_path=save_path, show=show, **kwargs
            )
        else:
            return plot_phase_space_plotly(
                pos, vel, plane=plane, save_path=save_path, show=show, **kwargs
            )

    def plot_colored_trajectory(
        self,
        orbit_idx: int = 0,
        color_by: str = "time",
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, plt.Axes], go.Figure]:
        """Plot 2D Face-On trajectory colored dynamically by time or SALI."""
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        pos = self.trajectories[orbit_idx][:, :3]

        if color_by.lower() == "sali":
            sali = np.squeeze(self.sali_array[orbit_idx])
            c_values = np.log10(np.min(sali, axis=0) if sali.ndim > 1 else sali)
            c_label = "log10(SALI)"
        else:
            c_values = self.timestamps
            c_label = "Time"

        if engine == "matplotlib":
            return plot_colored_trajectory_2d_mpl(
                pos, c_values, c_label=c_label, save_path=save_path, show=show, **kwargs
            )
        else:
            return plot_colored_trajectory_2d_plotly(
                pos, c_values, c_label=c_label, save_path=save_path, show=show, **kwargs
            )

    def plot_dashboard(
        self,
        orbit_idx: int = 0,
        backend: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, np.ndarray], None]:
        """Plot a multi-panel diagnostic dashboard summarizing trajectory and chaos metrics."""
        self._validate_index(orbit_idx)
        engine = self._resolve_backend(backend)

        # 1. Extract Chaos Detection Metadata
        chaos_report = self.detect_chaos(orbit_idx=orbit_idx, check_only=True)

        s_check = chaos_report.sali_check
        sali_is_chaotic = (
            bool(np.any(s_check)) if np.ndim(s_check) > 0 else bool(s_check)
        )
        s_time = chaos_report.sali_time
        sali_det_time = float(s_time.flat[0]) if np.ndim(s_time) > 0 else float(s_time)

        g_check = chaos_report.gali_check
        gali_is_chaotic = (
            bool(np.any(g_check)) if np.ndim(g_check) > 0 else bool(g_check)
        )
        g_time = chaos_report.gali_time
        gali_det_time = float(g_time.flat[0]) if np.ndim(g_time) > 0 else float(g_time)

        dt = self.timestamps[1] - self.timestamps[0]

        # 2. Extract Data Arrays
        sali = np.squeeze(self.sali_array[orbit_idx])
        if sali.ndim > 1:
            sali = np.min(sali, axis=0)

        lyap_data = (
            self.lyapunov_exponents[orbit_idx]
            if self.lyapunov_exponents is not None
            else None
        )

        data = {
            "t": self.timestamps,
            "pos": self.trajectories[orbit_idx][:, :3],
            "vel": self.trajectories[orbit_idx][:, 3:6],
            "sali": sali,
            "gali": np.squeeze(self.gali_array[orbit_idx]),
            "lyapunov": lyap_data,
            # Detection parameters
            "sali_is_chaotic": sali_is_chaotic,
            "sali_det_time": sali_det_time if np.isfinite(sali_det_time) else None,
            "sali_window_time": self.sali_window_size * dt,
            "gali_is_chaotic": gali_is_chaotic,
            "gali_det_time": gali_det_time if np.isfinite(gali_det_time) else None,
            "gali_window_time": self.gali_window_size * dt,
        }

        if engine == "matplotlib":
            return plot_dashboard_mpl(
                data,
                sali_threshold=self.sali_threshold,
                gali_threshold=self.gali_threshold,
                save_path=save_path,
                show=show,
                **kwargs,
            )
        else:
            return plot_dashboard_plotly(
                data,
                sali_threshold=self.sali_threshold,
                gali_threshold=self.gali_threshold,
                save_path=save_path,
                show=show,
                **kwargs,
            )

    def plot_sali_batch(
        self,
        orbit_indices: Optional[List[int]] = None,
        max_per_page: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> List[plt.Figure]:
        """Plot a grid of SALI vs Time plots for multiple orbits (paginated).

        Parameters
        ----------
        orbit_indices : list of int, optional
            Selected orbit indices (e.g., [0, 1, 4, 7]). Defaults to all integrated orbits.
        max_per_page : int, default 10
            Maximum subplots rendered per figure page.
        save_path : str, optional
            Path to export image files. Multi-page figures append '_page1', '_page2'.
        """
        chaos_report = self.detect_chaos(check_only=True)
        dt = self.timestamps[1] - self.timestamps[0]

        return plot_sali_batch_mpl(
            t=self.timestamps,
            sali_array=self.sali_array,
            orbit_indices=orbit_indices,
            sali_checks=chaos_report.sali_check,
            sali_times=chaos_report.sali_time,
            lyapunov_array=self.lyapunov_exponents,
            threshold=self.sali_threshold,
            window_size_time=self.sali_window_size * dt,
            max_per_page=max_per_page,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_gali_batch(
        self,
        orbit_indices: Optional[List[int]] = None,
        k_orders: Optional[List[int]] = None,
        max_per_page: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> List[plt.Figure]:
        """Plot a grid of GALI vs Time plots for multiple orbits (paginated)."""
        chaos_report = self.detect_chaos(check_only=True)
        dt = self.timestamps[1] - self.timestamps[0]

        return plot_gali_batch_mpl(
            t=self.timestamps,
            gali_array=self.gali_array,
            orbit_indices=orbit_indices,
            gali_checks=chaos_report.gali_check,
            gali_times=chaos_report.gali_time,
            lyapunov_array=self.lyapunov_exponents,
            threshold=self.gali_threshold,
            window_size_time=self.gali_window_size * dt,
            k_orders=k_orders,
            max_per_page=max_per_page,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_sali_gali_batch(
        self,
        orbit_indices: Optional[List[int]] = None,
        k_orders: Optional[List[int]] = None,
        max_orbits_per_page: int = 5,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> List[plt.Figure]:
        """Plot side-by-side SALI (left) and GALI (right) for a batch of orbits.

        Parameters
        ----------
        orbit_indices : list of int, optional
            Selected orbit indices. Defaults to all integrated orbits.
        max_orbits_per_page : int, default 5
            Number of orbits per figure page (5 orbits = 10 subplots per page).
        save_path : str, optional
            Output file path for saving figures.
        """
        chaos_report = self.detect_chaos(check_only=True)
        dt = self.timestamps[1] - self.timestamps[0]

        return plot_sali_gali_dual_batch_mpl(
            t=self.timestamps,
            sali_array=self.sali_array,
            gali_array=self.gali_array,
            orbit_indices=orbit_indices,
            sali_checks=chaos_report.sali_check,
            sali_times=chaos_report.sali_time,
            gali_checks=chaos_report.gali_check,
            gali_times=chaos_report.gali_time,
            lyapunov_array=self.lyapunov_exponents,
            sali_threshold=self.sali_threshold,
            gali_threshold=self.gali_threshold,
            sali_window_time=self.sali_window_size * dt,
            gali_window_time=self.gali_window_size * dt,
            k_orders=k_orders,
            max_orbits_per_page=max_orbits_per_page,
            save_path=save_path,
            show=show,
            **kwargs,
        )
