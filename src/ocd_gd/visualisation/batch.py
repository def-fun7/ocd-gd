"""
Batch visualization module for comparing multiple orbits simultaneously.
"""

from typing import List, Optional, Tuple, Union
import math
import numpy as np
import matplotlib.pyplot as plt

from .matplotlib_backend import plot_sali_mpl, plot_gali_mpl, _handle_save_show


def plot_sali_batch_mpl(
    t: np.ndarray,
    sali_array: np.ndarray,
    orbit_indices: Optional[List[int]] = None,
    sali_checks: Optional[np.ndarray] = None,
    sali_times: Optional[np.ndarray] = None,
    lyapunov_array: Optional[np.ndarray] = None,
    threshold: float = 1e-2,
    window_size_time: Optional[float] = None,
    max_per_page: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> List[plt.Figure]:
    """
    Plot grid of SALI vs Time graphs for multiple orbits. Paginated if total orbits > max_per_page.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    sali_array : np.ndarray
        Full SALI array of shape (N_orbits, ...).
    orbit_indices : list of int, optional
        Target orbit indices to plot. Defaults to all orbits in sali_array.
    max_per_page : int, default 10
        Maximum subplots per figure page.
    """
    num_total_orbits = sali_array.shape[0]
    if orbit_indices is None:
        orbit_indices = list(range(num_total_orbits))

    # Paginate indices into chunks of max_per_page
    pages = [
        orbit_indices[i : i + max_per_page]
        for i in range(0, len(orbit_indices), max_per_page)
    ]
    figures = []

    for page_idx, page_orbits in enumerate(pages):
        n_plots = len(page_orbits)
        ncols = 2 if n_plots > 1 else 1
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=kwargs.get("figsize", (6 * ncols, 3.5 * nrows)),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for i, idx in enumerate(page_orbits):
            ax = axes_flat[i]
            sali_data = np.squeeze(sali_array[idx])
            if sali_data.ndim > 1:
                sali_data = np.min(sali_data, axis=0)

            is_chaotic = bool(sali_checks[idx]) if sali_checks is not None else None
            det_time = None
            if sali_times is not None:
                t_val = np.squeeze(sali_times[idx])
                val_float = float(t_val.flat[0]) if t_val.ndim > 0 else float(t_val)
                if np.isfinite(val_float):
                    det_time = val_float
            lyap_data = lyapunov_array[idx] if lyapunov_array is not None else None

            plot_sali_mpl(
                t=t,
                sali=sali_data,
                threshold=threshold,
                is_chaotic=is_chaotic,
                detection_time=det_time,
                window_size_time=window_size_time,
                lyapunov=lyap_data,
                fig=fig,
                ax=ax,
                show=False,
                title=f"Orbit #{idx}",
                legend=(i == 0),  # Show legend only on first subplot to reduce clutter
            )

        # Hide remaining unused subplot grid cells on final page
        for j in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.tight_layout()

        # Handle saving paginated outputs (e.g. sali_batch_page1.png)
        page_save_path = None
        if save_path:
            if len(pages) > 1:
                base, ext = (
                    save_path.rsplit(".", 1) if "." in save_path else (save_path, "png")
                )
                page_save_path = f"{base}_page{page_idx + 1}.{ext}"
            else:
                page_save_path = save_path

        _handle_save_show(
            fig, save_path=page_save_path, show=show, backend="matplotlib", **kwargs
        )
        figures.append(fig)

    return figures


def plot_gali_batch_mpl(
    t: np.ndarray,
    gali_array: np.ndarray,
    orbit_indices: Optional[List[int]] = None,
    gali_checks: Optional[np.ndarray] = None,
    gali_times: Optional[np.ndarray] = None,
    lyapunov_array: Optional[np.ndarray] = None,
    threshold: float = 1e-16,
    window_size_time: Optional[float] = None,
    k_orders: Optional[List[int]] = None,
    max_per_page: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> List[plt.Figure]:
    """
    Plot grid of GALI vs Time graphs for multiple orbits. Paginated if total orbits > max_per_page.
    """
    num_total_orbits = gali_array.shape[0]
    if orbit_indices is None:
        orbit_indices = list(range(num_total_orbits))

    pages = [
        orbit_indices[i : i + max_per_page]
        for i in range(0, len(orbit_indices), max_per_page)
    ]
    figures = []

    for page_idx, page_orbits in enumerate(pages):
        n_plots = len(page_orbits)
        ncols = 2 if n_plots > 1 else 1
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=kwargs.get("figsize", (6 * ncols, 3.5 * nrows)),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for i, idx in enumerate(page_orbits):
            ax = axes_flat[i]
            gali_data = np.squeeze(gali_array[idx])

            is_chaotic = bool(gali_checks[idx]) if gali_checks is not None else None
            det_time = None
            if gali_times is not None:
                t_val = np.squeeze(gali_times[idx])
                val_float = float(t_val.flat[0]) if t_val.ndim > 0 else float(t_val)
                if np.isfinite(val_float):
                    det_time = val_float
            lyap_data = lyapunov_array[idx] if lyapunov_array is not None else None

            plot_gali_mpl(
                t=t,
                gali=gali_data,
                k_orders=k_orders,
                threshold=threshold,
                is_chaotic=is_chaotic,
                detection_time=det_time,
                window_size_time=window_size_time,
                lyapunov=lyap_data,
                fig=fig,
                ax=ax,
                show=False,
                title=f"Orbit #{idx}",
                legend=(i == 0),
            )

        for j in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.tight_layout()

        page_save_path = None
        if save_path:
            if len(pages) > 1:
                base, ext = (
                    save_path.rsplit(".", 1) if "." in save_path else (save_path, "png")
                )
                page_save_path = f"{base}_page{page_idx + 1}.{ext}"
            else:
                page_save_path = save_path

        _handle_save_show(
            fig, save_path=page_save_path, show=show, backend="matplotlib", **kwargs
        )
        figures.append(fig)

    return figures


def plot_sali_gali_dual_batch_mpl(
    t: np.ndarray,
    sali_array: np.ndarray,
    gali_array: np.ndarray,
    orbit_indices: Optional[List[int]] = None,
    sali_checks: Optional[np.ndarray] = None,
    sali_times: Optional[np.ndarray] = None,
    gali_checks: Optional[np.ndarray] = None,
    gali_times: Optional[np.ndarray] = None,
    lyapunov_array: Optional[np.ndarray] = None,
    sali_threshold: float = 1e-2,
    gali_threshold: float = 1e-16,
    sali_window_time: Optional[float] = None,
    gali_window_time: Optional[float] = None,
    k_orders: Optional[List[int]] = None,
    max_orbits_per_page: int = 5,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> List[plt.Figure]:
    """
    Plot side-by-side SALI (left) and GALI (right) for multiple orbits in a batch.

    Parameters
    ----------
    max_orbits_per_page : int, default 5
        Maximum number of orbits (rows = 5, total subplots = 10) per page.
    """
    num_total_orbits = sali_array.shape[0]
    if orbit_indices is None:
        orbit_indices = list(range(num_total_orbits))

    # Split target orbits into chunks (e.g. 5 orbits -> 10 subplots max per page)
    pages = [
        orbit_indices[i : i + max_orbits_per_page]
        for i in range(0, len(orbit_indices), max_orbits_per_page)
    ]
    figures = []

    for page_idx, page_orbits in enumerate(pages):
        nrows = len(page_orbits)
        ncols = 2

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=kwargs.get("figsize", (12, 3.2 * nrows)),
            squeeze=False,
        )

        for row_idx, idx in enumerate(page_orbits):
            ax_sali = axes[row_idx, 0]
            ax_gali = axes[row_idx, 1]

            # 1. Process SALI
            sali_data = np.squeeze(sali_array[idx])
            if sali_data.ndim > 1:
                sali_data = np.min(sali_data, axis=0)

            s_check = sali_checks[idx] if sali_checks is not None else None
            sali_is_chaotic = bool(np.squeeze(s_check)) if s_check is not None else None

            sali_det_time = None
            if sali_times is not None:
                st_val = np.squeeze(sali_times[idx])
                st_float = float(st_val.flat[0]) if st_val.ndim > 0 else float(st_val)
                if np.isfinite(st_float):
                    sali_det_time = st_float

            # 2. Process GALI
            gali_data = np.squeeze(gali_array[idx])

            g_check = gali_checks[idx] if gali_checks is not None else None
            gali_is_chaotic = bool(np.squeeze(g_check)) if g_check is not None else None

            gali_det_time = None
            if gali_times is not None:
                gt_val = np.squeeze(gali_times[idx])
                gt_float = float(gt_val.flat[0]) if gt_val.ndim > 0 else float(gt_val)
                if np.isfinite(gt_float):
                    gali_det_time = gt_float

            lyap_data = lyapunov_array[idx] if lyapunov_array is not None else None

            # Render Left Panel (SALI)
            plot_sali_mpl(
                t=t,
                sali=sali_data,
                threshold=sali_threshold,
                is_chaotic=sali_is_chaotic,
                detection_time=sali_det_time,
                window_size_time=sali_window_time,
                lyapunov=lyap_data,
                fig=fig,
                ax=ax_sali,
                show=False,
                title=f"Orbit #{idx} - SALI",
                legend=(row_idx == 0),
            )

            # Render Right Panel (GALI)
            plot_gali_mpl(
                t=t,
                gali=gali_data,
                k_orders=k_orders,
                threshold=gali_threshold,
                is_chaotic=gali_is_chaotic,
                detection_time=gali_det_time,
                window_size_time=gali_window_time,
                lyapunov=lyap_data,
                fig=fig,
                ax=ax_gali,
                show=False,
                title=f"Orbit #{idx} - GALI",
                legend=(row_idx == 0),
            )

        fig.tight_layout()

        # Multi-page filename formatting
        page_save_path = None
        if save_path:
            if len(pages) > 1:
                base, ext = (
                    save_path.rsplit(".", 1) if "." in save_path else (save_path, "png")
                )
                page_save_path = f"{base}_page{page_idx + 1}.{ext}"
            else:
                page_save_path = save_path

        _handle_save_show(
            fig, save_path=page_save_path, show=show, backend="matplotlib", **kwargs
        )
        figures.append(fig)

    return figures
