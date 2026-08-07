"""
Matplotlib backend for plotting galactic orbit dynamics and chaos indicators.
"""

__all__ = [
    "plot_colored_trajectory_2d_mpl",
    "plot_energy_drift_mpl",
    "plot_gali_mpl",
    "plot_phase_space_mpl",
    "plot_sali_mpl",
    "plot_trajectory_2d_mpl",
    "plot_trajectory_3d_mpl",
]


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

from .utils import resolve_save_path


def _setup_fig_ax(
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    projection: str | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Helper utility to manage figure and axes creation across standalone/subplot contexts."""
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if projection == "3d":
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    return fig, ax

def _handle_save_show(
    fig,
    save_path: str | None = None,
    show: bool = True,
    backend: str = "matplotlib",
    **kwargs,
) -> None:
    """Helper utility to handle saving and displaying figures."""
    if save_path:
        save_path = resolve_save_path(save_path, backend)
        dpi = kwargs.get("dpi", 300)
        bbox_inches = kwargs.get("bbox_inches", "tight")
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()

def _format_lyap_text(lyap_data: npt.NDArray[np.float64] | None) -> str | None:
    """Format AGAMA's Lyapunov array [lambda * Torb, Tchaos / Torb] cleanly."""
    if lyap_data is None:
        return None

    lyap_flat = np.squeeze(lyap_data)
    if lyap_flat.size < 2:
        return None

    lam_torb, t_chaos = lyap_flat[0], lyap_flat[1]

    if np.isnan(lam_torb) or np.isnan(t_chaos):
        return r"$\lambda T_{\mathrm{orb}}$: NaN (Unconverged)"
    elif lam_torb == 0.0 or np.isinf(t_chaos):
        return r"$\lambda T_{\mathrm{orb}} = 0.0$"
    else:
        return (
            rf"$\lambda T_{{\mathrm{{orb}}}} = {lam_torb:.3f}$"
            + "\n"
            + rf"$T_{{\mathrm{{chaos}}}} = {t_chaos:.1f} \, T_{{\mathrm{{orb}}}}$"
        )

def plot_sali_mpl(
    t: npt.NDArray[np.float64],
    sali: npt.NDArray[np.float64],
    threshold: float | None = 1e-8,
    is_chaotic: bool | None = None,
    detection_time: float | None = None,
    window_size_time: float | None = None,
    lyapunov: npt.NDArray[np.float64] | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot SALI vs Time on a log scale.

    Args:
        t: Time array.
        sali: SALI evolution array.
        threshold: Chaos detection threshold line value. Defaults to 1e-8.
        is_chaotic: Flag indicating if classified as chaotic. Defaults to None.
        detection_time: Time of chaos detection. Defaults to None.
        window_size_time: Size of the sliding detection window. Defaults to None.
        lyapunov: Lyapunov exponents/evolution data to format and show. Defaults to None.
        fig: Matplotlib Figure object to reuse. Defaults to None.
        ax: Matplotlib Axes object to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated/updated figure and axes objects.

    Examples:
        >>> import numpy as np
        >>> t = np.linspace(0, 100, 100)
        >>> sali = np.exp(-t)
        >>> fig, ax = plot_sali_mpl(t, sali, show=False)
    """
    fig, ax = _setup_fig_ax(fig, ax, figsize=kwargs.get("figsize", (7, 4.5)))

    base_title = kwargs.get("title", "SALI vs Time")
    if is_chaotic is not None:
        status_str = "Chaotic" if is_chaotic else "Regular"
        status_color = "crimson" if is_chaotic else "darkgreen"
        ax.set_title(
            f"{base_title} [{status_str}]", color=status_color, fontweight="bold"
        )
    else:
        ax.set_title(base_title)

    color = kwargs.get("color", "crimson")
    lw = kwargs.get("linewidth", kwargs.get("lw", 1.5))
    label = kwargs.get("label", "SALI")
    ax.plot(t, sali, color=color, lw=lw, label=label)

    if threshold is not None:
        thresh_color = kwargs.get("thresh_color", "black")
        thresh_ls = kwargs.get("thresh_ls", "--")
        ax.axhline(
            threshold,
            color=thresh_color,
            linestyle=thresh_ls,
            alpha=0.7,
            label=kwargs.get("thresh_label", f"Threshold ({threshold:.0e})"),
        )

    if (
        is_chaotic
        and detection_time is not None
        and np.isfinite(detection_time)
        and window_size_time is not None
    ):
        t_start = max(t[0], detection_time - window_size_time)
        t_end = detection_time

        ax.axvspan(
            t_start,
            t_end,
            ymin=0,
            ymax=1,
            color="orange",
            alpha=0.3,
            zorder=0,
            label=f"Detection Window (t={t_end:.2f})",
        )
        ax.axvline(t_end, color="orange", linestyle=":", lw=1.5, zorder=2)

    lyap_str = _format_lyap_text(lyapunov)
    if lyap_str and kwargs.get("show_lyapunov", True):
        ax.text(
            0.03,
            0.05,
            lyap_str,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
            },
        )

    ax.set_yscale("log")
    ax.set_xlabel(kwargs.get("xlabel", "Time"))
    ax.set_ylabel(kwargs.get("ylabel", "SALI"))
    ax.grid(kwargs.get("grid", True), which="both", linestyle=":", alpha=0.5)

    if kwargs.get("legend", True):
        ax.legend(loc=kwargs.get("legend_loc", "best"))

    _handle_save_show(
        fig, save_path=save_path, show=show, backend="matplotlib", **kwargs
    )
    return fig, ax

def plot_gali_mpl(
    t: npt.NDArray[np.float64],
    gali: npt.NDArray[np.float64],
    k_orders: list[int | None] | None = None,
    threshold: float | None = 1e-16,
    is_chaotic: bool | None = None,
    detection_time: float | None = None,
    window_size_time: float | None = None,
    lyapunov: npt.NDArray[np.float64] | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot GALI_k vs Time on a log scale.

    Args:
        t: Time array.
        gali: GALI evolution array of shape (N, k) or (N,).
        k_orders: GALI order list to plot. Defaults to None.
        threshold: Chaos detection threshold line value. Defaults to 1e-16.
        is_chaotic: Flag indicating if classified as chaotic. Defaults to None.
        detection_time: Time of chaos detection. Defaults to None.
        window_size_time: Size of the sliding detection window. Defaults to None.
        lyapunov: Lyapunov exponents/evolution data to format and show. Defaults to None.
        fig: Matplotlib Figure object to reuse. Defaults to None.
        ax: Matplotlib Axes object to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated/updated figure and axes objects.

    Examples:
        >>> import numpy as np
        >>> t = np.linspace(0, 100, 100)
        >>> gali = np.exp(-2 * t)
        >>> fig, ax = plot_gali_mpl(t, gali, show=False)
    """
    fig, ax = _setup_fig_ax(fig, ax, figsize=kwargs.get("figsize", (7, 4.5)))

    base_title = kwargs.get("title", "GALI vs Time")
    if is_chaotic is not None:
        status_str = "Chaotic" if is_chaotic else "Regular"
        status_color = "crimson" if is_chaotic else "darkgreen"
        ax.set_title(
            f"{base_title} [{status_str}]", color=status_color, fontweight="bold"
        )
    else:
        ax.set_title(base_title)

    if gali.ndim == 1:
        label = f"GALI_{k_orders[0]}" if k_orders else "GALI"
        ax.plot(t, gali, lw=kwargs.get("lw", 1.5), label=label)
    else:
        num_k = gali.shape[1]
        colors = kwargs.get("colors", plt.cm.plasma(np.linspace(0.1, 0.9, num_k)))
        for i in range(num_k):
            lbl = (
                f"GALI_{k_orders[i]}"
                if k_orders and i < len(k_orders)
                else f"GALI_{i+2}"
            )
            ax.plot(t, gali[:, i], color=colors[i], lw=kwargs.get("lw", 1.5), label=lbl)

    if threshold is not None:
        thresh_color = kwargs.get("thresh_color", "black")
        thresh_ls = kwargs.get("thresh_ls", "--")
        ax.axhline(
            threshold,
            color=thresh_color,
            linestyle=thresh_ls,
            alpha=0.7,
            label=kwargs.get("thresh_label", f"Threshold ({threshold:.0e})"),
        )

    if (
        is_chaotic
        and detection_time is not None
        and np.isfinite(detection_time)
        and window_size_time is not None
    ):
        t_start = max(t[0], detection_time - window_size_time)
        t_end = detection_time

        ax.axvspan(
            t_start,
            t_end,
            ymin=0,
            ymax=1,
            color="orange",
            alpha=0.3,
            zorder=0,
            label=f"Detection Window (t={t_end:.2f})",
        )
        ax.axvline(t_end, color="orange", linestyle=":", lw=1.5, zorder=2)

    lyap_str = _format_lyap_text(lyapunov)
    if lyap_str and kwargs.get("show_lyapunov", True):
        ax.text(
            0.03,
            0.05,
            lyap_str,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
            },
        )

    ax.set_yscale("log")
    ax.set_xlabel(kwargs.get("xlabel", "Time"))
    ax.set_ylabel(kwargs.get("ylabel", "GALI"))
    ax.grid(kwargs.get("grid", True), which="both", linestyle=":", alpha=0.5)

    if kwargs.get("legend", True):
        ax.legend(loc=kwargs.get("legend_loc", "best"))

    _handle_save_show(
        fig, save_path=save_path, show=show, backend="matplotlib", **kwargs
    )
    return fig, ax

def plot_trajectory_2d_mpl(
    pos: npt.NDArray[np.float64],
    fig: plt.Figure | None = None,
    axes: plt.Axes | (npt.NDArray[np.float64] | None) = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, npt.NDArray[np.float64]]:
    """Plot 2D projections of the orbit: Face-On (X-Y) and Edge-On (X-Z).

    Args:
        pos: Orbit trajectory position array of shape (N, 3).
        fig: Matplotlib Figure object to reuse. Defaults to None.
        axes: Axes object(s) to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, npt.NDArray[np.float64]]: The generated figure and axes array.

    Examples:
        >>> import numpy as np
        >>> pos = np.random.randn(100, 3)
        >>> fig, axes = plot_trajectory_2d_mpl(pos, show=False)
    """
    if axes is None:
        figsize = kwargs.get("figsize", (11, 5))
        fig, axes_arr = plt.subplots(1, 2, figsize=figsize)
    else:
        axes_arr = np.atleast_1d(axes)
        fig = axes_arr[0].get_figure()

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    color = kwargs.get("color", "navy")
    alpha = kwargs.get("alpha", 0.7)
    lw = kwargs.get("lw", 0.8)

    axes_arr[0].plot(x, y, color=color, alpha=alpha, lw=lw)
    axes_arr[0].set_xlabel("X")
    axes_arr[0].set_ylabel("Y")
    axes_arr[0].set_title(kwargs.get("title_faceon", "Face-On (X - Y)"))
    axes_arr[0].set_aspect("equal", adjustable="datalim")
    axes_arr[0].grid(kwargs.get("grid", True), linestyle=":", alpha=0.5)

    axes_arr[1].plot(x, z, color=color, alpha=alpha, lw=lw)
    axes_arr[1].set_xlabel("X")
    axes_arr[1].set_ylabel("Z")
    axes_arr[1].set_title(kwargs.get("title_edgeon", "Edge-On (X - Z)"))
    axes_arr[1].set_aspect("equal", adjustable="datalim")
    axes_arr[1].grid(kwargs.get("grid", True), linestyle=":", alpha=0.5)

    if kwargs.get("suptitle"):
        fig.suptitle(kwargs.get("suptitle"))

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig, axes_arr

def plot_trajectory_3d_mpl(
    pos: npt.NDArray[np.float64],
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 3D spatial orbit trajectory.

    Args:
        pos: Orbit trajectory position array of shape (N, 3).
        fig: Matplotlib Figure object to reuse. Defaults to None.
        ax: Matplotlib Axes object to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated/updated figure and axes objects.

    Examples:
        >>> import numpy as np
        >>> pos = np.random.randn(100, 3)
        >>> fig, ax = plot_trajectory_3d_mpl(pos, show=False)
    """
    fig, ax = _setup_fig_ax(
        fig, ax, projection="3d", figsize=kwargs.get("figsize", (8, 7))
    )

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    color = kwargs.get("color", "teal")
    lw = kwargs.get("lw", 0.8)
    alpha = kwargs.get("alpha", 0.8)

    ax.plot(
        x,
        y,
        z,
        color=color,
        lw=lw,
        alpha=alpha,
        label=kwargs.get("label", "Trajectory"),
    )

    if kwargs.get("mark_endpoints", True):
        ax.scatter(x[0], y[0], z[0], color="green", s=30, label="Start", zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], color="red", s=30, label="End", zorder=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(kwargs.get("title", "3D Orbit Trajectory"))

    if kwargs.get("legend", True):
        ax.legend(loc=kwargs.get("legend_loc", "best"))

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig, ax

def plot_phase_space_mpl(
    pos: npt.NDArray[np.float64],
    vel: npt.NDArray[np.float64],
    plane: str = "x",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D phase space projection (e.g., X vs V_x, Y vs V_y, or Z vs V_z).

    Args:
        pos: Orbit trajectory position array of shape (N, 3).
        vel: Orbit trajectory velocity array of shape (N, 3).
        plane: 'x', 'y', or 'z' specifying which component axis to plot. Defaults to "x".
        fig: Matplotlib Figure object to reuse. Defaults to None.
        ax: Matplotlib Axes object to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated/updated figure and axes objects.

    Examples:
        >>> import numpy as np
        >>> pos = np.random.randn(100, 3)
        >>> vel = np.random.randn(100, 3)
        >>> fig, ax = plot_phase_space_mpl(pos, vel, plane="x", show=False)
    """
    fig, ax = _setup_fig_ax(fig, ax, figsize=kwargs.get("figsize", (7, 5)))

    idx = {"x": 0, "y": 1, "z": 2}.get(plane.lower(), 0)
    q = pos[:, idx]
    v = vel[:, idx]

    color = kwargs.get("color", "purple")
    alpha = kwargs.get("alpha", 0.5)
    s = kwargs.get("s", 1)

    ax.scatter(
        q,
        v,
        color=color,
        alpha=alpha,
        s=s,
        label=kwargs.get("label", f"{plane.upper()} Phase Space"),
    )

    ax.set_xlabel(kwargs.get("xlabel", f"{plane.upper()}"))
    ax.set_ylabel(kwargs.get("ylabel", f"V_{plane.lower()}"))
    ax.set_title(
        kwargs.get("title", f"Phase Space ({plane.upper()} vs V_{plane.lower()})")
    )
    ax.grid(kwargs.get("grid", True), linestyle=":", alpha=0.5)

    if kwargs.get("legend", False):
        ax.legend(loc=kwargs.get("legend_loc", "best"))

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig, ax

def plot_energy_drift_mpl(
    t: npt.NDArray[np.float64],
    energy: npt.NDArray[np.float64],
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot fractional energy conservation error: |(E(t) - E_0) / E_0| over time.

    Verifies numerical integration accuracy from AGAMA.

    Args:
        t: Time array.
        energy: Total energy evolution array.
        fig: Matplotlib Figure object to reuse. Defaults to None.
        ax: Matplotlib Axes object to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated/updated figure and axes objects.

    Examples:
        >>> import numpy as np
        >>> t = np.linspace(0, 100, 100)
        >>> energy = -1.0 + 1e-6 * np.random.randn(100)
        >>> fig, ax = plot_energy_drift_mpl(t, energy, show=False)
    """
    fig, ax = _setup_fig_ax(fig, ax, figsize=kwargs.get("figsize", (7, 4.5)))

    e0 = energy[0]
    drift = np.abs((energy - e0) / e0)

    color = kwargs.get("color", "darkgreen")
    lw = kwargs.get("lw", 1.2)

    ax.plot(t, drift, color=color, lw=lw, label=kwargs.get("label", "|ΔE/E₀|"))

    if kwargs.get("log_scale", True):
        ax.set_yscale("log")

    ax.set_xlabel(kwargs.get("xlabel", "Time"))
    ax.set_ylabel(kwargs.get("ylabel", "|(E(t) - E₀) / E₀|"))
    ax.set_title(kwargs.get("title", "Fractional Energy Drift"))
    ax.grid(kwargs.get("grid", True), which="both", linestyle=":", alpha=0.5)

    if kwargs.get("legend", True):
        ax.legend(loc=kwargs.get("legend_loc", "best"))

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig, ax

def plot_colored_trajectory_2d_mpl(
    pos: npt.NDArray[np.float64],
    c_values: npt.NDArray[np.float64],
    c_label: str = "Time",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D Face-On (X-Y) trajectory colored continuously by a scalar array.

    E.g. colored by Time or log(SALI).

    Args:
        pos: Orbit trajectory position array of shape (N, 3).
        c_values: Scalar values array of shape (N,) to map to segment colors.
        c_label: Label for the colorbar. Defaults to "Time".
        fig: Matplotlib Figure object to reuse. Defaults to None.
        ax: Matplotlib Axes object to reuse. Defaults to None.
        save_path: Path to export the figure. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated/updated figure and axes objects.

    Examples:
        >>> import numpy as np
        >>> pos = np.random.randn(100, 3)
        >>> t = np.linspace(0, 100, 100)
        >>> fig, ax = plot_colored_trajectory_2d_mpl(pos, t, show=False)
    """
    fig, ax = _setup_fig_ax(fig, ax, figsize=kwargs.get("figsize", (8, 6.5)))

    x, y = pos[:, 0], pos[:, 1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = kwargs.get("cmap", "viridis")
    norm = kwargs.get("norm", None)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c_values)
    lc.set_linewidth(kwargs.get("lw", 1.5))
    lc.set_alpha(kwargs.get("alpha", 0.9))

    line = ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label(c_label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(kwargs.get("title", f"Trajectory Colored by {c_label}"))
    ax.grid(kwargs.get("grid", True), linestyle=":", alpha=0.5)

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig, ax
