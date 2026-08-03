"""
Plotting mixin for OrbitChaosDetector.

Isolated from the core detector module because visualization is a distinct
concern from integration/chaos-computation, and this is by far the largest
chunk of the class's surface area. `_OrbitPlottingMixin` is combined with the
core class via inheritance in `orbit_chaos_detector.py`; it assumes the host
class provides `_validate_index`, `_resolve_backend`, `detect_chaos`,
`timestamps`, `trajectories`, `sali_array`, `gali_array`,
`lyapunov_exponents`, `sali_threshold`, `gali_threshold`, `sali_window_size`,
and `gali_window_size`.
"""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from .visualisation import (
    plot_colored_trajectory_2d_mpl,
    plot_colored_trajectory_2d_plotly,
    plot_dashboard_mpl,
    plot_dashboard_plotly,
    plot_gali_batch_mpl,
    plot_gali_mpl,
    plot_gali_plotly,
    plot_phase_space_mpl,
    plot_phase_space_plotly,
    plot_sali_batch_mpl,
    plot_sali_gali_dual_batch_mpl,
    plot_sali_mpl,
    plot_sali_plotly,
    plot_trajectory_2d_mpl,
    plot_trajectory_2d_plotly,
    plot_trajectory_3d_mpl,
    plot_trajectory_3d_plotly,
)


class _OrbitPlottingMixin:
    """All `plot_*` methods and the small helpers that only exist to serve them."""

    # =========================================================================
    # SHARED PLOTTING HELPERS
    # =========================================================================

    def _dispatch_plot(
        self,
        mpl_fn: Callable[..., Any],
        plotly_fn: Callable[..., Any],
        backend: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve the active backend and call the matching mpl/plotly function.

        Centralizes the "pick matplotlib vs plotly implementation" branch that
        would otherwise be repeated in every plotting method. `mpl_fn` and
        `plotly_fn` are called with identical positional/keyword arguments, so
        this only applies where both implementations share a signature.
        """
        engine = self._resolve_backend(backend)
        plot_fn = mpl_fn if engine == "matplotlib" else plotly_fn
        return plot_fn(*args, **kwargs)

    def _get_dt(self) -> float:
        """Return the spacing between the first two recorded timestamps."""
        return self.timestamps[1] - self.timestamps[0]

    @staticmethod
    def _detection_info(
        check: npt.NDArray[np.float64], time_val: npt.NDArray[np.float64]
    ) -> tuple[bool, float]:
        """Reduce a chaos-check/detection-time pair to a plain (bool, float).

        `check` and `time_val` may either be scalars (single-orbit lookup) or
        arrays (batch lookup, or separate-pair SALI results); both cases are
        handled uniformly here instead of being re-derived in every plot
        method. For arrays, `is_chaotic` is True if any element flags chaos,
        and `detection_time` is taken from the first entry.
        """
        is_chaotic = bool(np.any(check)) if np.ndim(check) > 0 else bool(check)
        det_time = float(time_val.flat[0]) if np.ndim(time_val) > 0 else float(time_val)
        return is_chaotic, det_time

    # =========================================================================
    # SINGLE-ORBIT PLOTS
    # =========================================================================

    def plot_sali(
        self,
        orbit_idx: int = 0,
        all_pairs: bool = False,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes] | go.Figure:
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

        chaos_report = self.detect_chaos(orbit_idx=orbit_idx, check_only=True)
        is_chaotic, det_time = self._detection_info(
            chaos_report.sali_check, chaos_report.sali_time
        )

        # Approximate sliding window duration in time units
        window_time = self.sali_window_size * self._get_dt()

        sali_data = np.squeeze(self.sali_array[orbit_idx])
        if sali_data.ndim > 1 and not all_pairs:
            sali_data = np.min(sali_data, axis=0)
        lyap_data = (
            self.lyapunov_exponents[orbit_idx]
            if self.lyapunov_exponents is not None
            else None
        )

        return self._dispatch_plot(
            plot_sali_mpl,
            plot_sali_plotly,
            backend,
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
        k_orders: list[int | None] | None = None,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes] | go.Figure:
        """Plot GALI vs Time for a target orbit.

        Parameters
        ----------
        orbit_idx : int, default 0
            Target orbit index.
        k_orders : list of int, optional
            Which GALI orders to overlay on the plot. Defaults to whatever the
            underlying plot function chooses (typically all available orders).
        backend : str, optional
            'matplotlib' or 'plotly' (overrides default).
        save_path : str, optional
            Path to export figure.
        show : bool, default True
            Display figure immediately.
        """
        self._validate_index(orbit_idx)

        chaos_report = self.detect_chaos(orbit_idx=orbit_idx, check_only=True)
        is_chaotic, det_time = self._detection_info(
            chaos_report.gali_check, chaos_report.gali_time
        )

        window_time = self.gali_window_size * self._get_dt()

        gali_data = np.squeeze(self.gali_array[orbit_idx])
        lyap_data = (
            self.lyapunov_exponents[orbit_idx]
            if self.lyapunov_exponents is not None
            else None
        )

        return self._dispatch_plot(
            plot_gali_mpl,
            plot_gali_plotly,
            backend,
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

    def plot_trajectory_2d(
        self,
        orbit_idx: int = 0,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, npt.NDArray[np.float64]] | go.Figure:
        """Plot Face-On (X-Y) and Edge-On (X-Z) 2D orbit projections."""
        self._validate_index(orbit_idx)
        pos = self.trajectories[orbit_idx][:, :3]

        return self._dispatch_plot(
            plot_trajectory_2d_mpl,
            plot_trajectory_2d_plotly,
            backend,
            pos,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_trajectory_3d(
        self,
        orbit_idx: int = 0,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes] | go.Figure:
        """Plot 3D spatial orbit path."""
        self._validate_index(orbit_idx)
        pos = self.trajectories[orbit_idx][:, :3]

        return self._dispatch_plot(
            plot_trajectory_3d_mpl,
            plot_trajectory_3d_plotly,
            backend,
            pos,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_phase_space(
        self,
        orbit_idx: int = 0,
        plane: str = "x",
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes] | go.Figure:
        """Plot 2D phase space scatter projection (Position vs Velocity)."""
        self._validate_index(orbit_idx)
        pos = self.trajectories[orbit_idx][:, :3]
        vel = self.trajectories[orbit_idx][:, 3:6]

        return self._dispatch_plot(
            plot_phase_space_mpl,
            plot_phase_space_plotly,
            backend,
            pos,
            vel,
            plane=plane,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_colored_trajectory(
        self,
        orbit_idx: int = 0,
        color_by: str = "time",
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes] | go.Figure:
        """Plot 2D Face-On trajectory colored dynamically by time or SALI."""
        self._validate_index(orbit_idx)
        pos = self.trajectories[orbit_idx][:, :3]

        if color_by.lower() == "sali":
            sali = np.squeeze(self.sali_array[orbit_idx])
            c_values = np.log10(np.min(sali, axis=0) if sali.ndim > 1 else sali)
            c_label = "log10(SALI)"
        else:
            c_values = self.timestamps
            c_label = "Time"

        return self._dispatch_plot(
            plot_colored_trajectory_2d_mpl,
            plot_colored_trajectory_2d_plotly,
            backend,
            pos,
            c_values,
            c_label=c_label,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_dashboard(
        self,
        orbit_idx: int = 0,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, npt.NDArray[np.float64]] | None:
        """Plot a multi-panel diagnostic dashboard summarizing trajectory and chaos metrics."""
        self._validate_index(orbit_idx)

        # 1. Extract Chaos Detection Metadata
        chaos_report = self.detect_chaos(orbit_idx=orbit_idx, check_only=True)
        sali_is_chaotic, sali_det_time = self._detection_info(
            chaos_report.sali_check, chaos_report.sali_time
        )
        gali_is_chaotic, gali_det_time = self._detection_info(
            chaos_report.gali_check, chaos_report.gali_time
        )
        dt = self._get_dt()

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

        return self._dispatch_plot(
            plot_dashboard_mpl,
            plot_dashboard_plotly,
            backend,
            data,
            sali_threshold=self.sali_threshold,
            gali_threshold=self.gali_threshold,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    # =========================================================================
    # BATCH PLOTS (matplotlib only — no plotly batch implementation exists)
    # =========================================================================

    def plot_sali_batch(
        self,
        orbit_indices: list[int | None] | None = None,
        max_per_page: int = 10,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> list[plt.Figure]:
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
        dt = self._get_dt()

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
        orbit_indices: list[int | None] | None = None,
        k_orders: list[int | None] | None = None,
        max_per_page: int = 10,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> list[plt.Figure]:
        """Plot a grid of GALI vs Time plots for multiple orbits (paginated)."""
        chaos_report = self.detect_chaos(check_only=True)
        dt = self._get_dt()

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
        orbit_indices: list[int | None] | None = None,
        k_orders: list[int | None] | None = None,
        max_orbits_per_page: int = 5,
        save_path: str | None = None,
        show: bool = True,
        **kwargs,
    ) -> list[plt.Figure]:
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
        dt = self._get_dt()

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
