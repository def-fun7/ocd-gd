"""
Multi-panel diagnostic dashboard for orbit dynamics and chaos indicators.
"""

__all__ = ["plot_dashboard_mpl", "plot_dashboard_plotly"]

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .mpl_backend import (
    _handle_save_show,
    plot_gali_mpl,
    plot_sali_mpl,
    plot_trajectory_3d_mpl,
)
from .plotly_backend import (
    plot_sali_plotly,
    plot_trajectory_2d_plotly,
    plot_trajectory_3d_plotly,
)


def plot_dashboard_mpl(
    data: dict[str, Any],
    sali_threshold: float = 1e-2,
    gali_threshold: float = 1e-16,
    k_orders: list[int | None] | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, npt.NDArray[np.float64]]:
    """Generate a 4-panel Matplotlib summary dashboard.

    The panels show the 3D trajectory, 2D face-on projection, SALI evolution,
    and GALI evolution.

    Args:
        data: Dict containing orbit integration and chaos results. Must contain:
            - "t": Time array.
            - "pos": Position array of shape (N, 3).
            - "sali": SALI evolution array.
            - "gali": GALI evolution array/dict.
            - "sali_is_chaotic" (optional): Chaotic classification from SALI.
            - "sali_det_time" (optional): SALI detection time.
            - "sali_window_time" (optional): SALI window time size.
            - "gali_is_chaotic" (optional): Chaotic classification from GALI.
            - "gali_det_time" (optional): GALI detection time.
            - "gali_window_time" (optional): GALI window time size.
            - "lyapunov" (optional): Lyapunov exponent/evolution.
        sali_threshold: Chaos detection threshold for SALI. Defaults to 1e-2.
        gali_threshold: Chaos detection threshold for GALI. Defaults to 1e-16.
        k_orders: GALI order list to plot. Defaults to None.
        save_path: Path to save the figure. If None, it is not saved. Defaults to None.
        show: If True, calls `plt.show()`. Defaults to True.
        **kwargs: Additional plotting options. Supported options include:
            - figsize (tuple): Size of the figure. Defaults to (14, 10).
            - suptitle (bool): Whether to show the main title. Defaults to True.
            - title (str): Custom main title.

    Returns:
        tuple[plt.Figure, npt.NDArray[np.float64]]: The generated figure and its axes.
    """
    fig = plt.figure(figsize=kwargs.get("figsize", (14, 10)))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_2d_face = fig.add_subplot(gs[0, 1])
    ax_sali = fig.add_subplot(gs[1, 0])
    ax_gali = fig.add_subplot(gs[1, 1])

    t = data["t"]
    pos = data["pos"]

    plot_trajectory_3d_mpl(
        pos,
        fig=fig,
        ax=ax_3d,
        show=False,
        mark_endpoints=True,
        title="3D Orbit Trajectory",
    )

    ax_2d_face.plot(pos[:, 0], pos[:, 1], color="navy", lw=0.8, alpha=0.7)
    ax_2d_face.set_xlabel("X")
    ax_2d_face.set_ylabel("Y")
    ax_2d_face.set_title("Face-On Projection (X - Y)")
    ax_2d_face.set_aspect("equal", adjustable="datalim")
    ax_2d_face.grid(True, linestyle=":", alpha=0.5)

    plot_sali_mpl(
        t=t,
        sali=data["sali"],
        threshold=sali_threshold,
        is_chaotic=data.get("sali_is_chaotic"),
        detection_time=data.get("sali_det_time"),
        window_size_time=data.get("sali_window_time"),
        lyapunov=data.get("lyapunov"),
        fig=fig,
        ax=ax_sali,
        show=False,
    )

    plot_gali_mpl(
        t=t,
        gali=data["gali"],
        k_orders=k_orders,
        threshold=gali_threshold,
        is_chaotic=data.get("gali_is_chaotic"),
        detection_time=data.get("gali_det_time"),
        window_size_time=data.get("gali_window_time"),
        lyapunov=data.get("lyapunov"),
        fig=fig,
        ax=ax_gali,
        show=False,
    )
    if kwargs.get("suptitle", True):
        status_str = (
            "Chaotic"
            if data.get("sali_is_chaotic") or data.get("gali_is_chaotic")
            else "Regular"
        )
        fig.suptitle(
            kwargs.get("title", f"Orbit Chaos Diagnostic Summary [{status_str}]"),
            fontsize=14,
            fontweight="bold",
        )

    _handle_save_show(
        fig, save_path=save_path, show=show, backend="matplotlib", **kwargs
    )
    return fig, fig.axes


def plot_dashboard_plotly(
    data: dict[str, npt.NDArray[np.float64]],
    threshold: float = 1e-8,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> None:
    """Generates interactive Plotly plots as sequential views or combined views.

    Args:
        data: Dict containing orbit integration and chaos results. Must contain:
            - "pos": Position array of shape (N, 3).
            - "t": Time array.
            - "sali": SALI evolution array.
        threshold: Chaos detection threshold for SALI. Defaults to 1e-8.
        save_path: Path to save the interactive plots. If set, saves three files with
            prefixes _3d, _2d, and _sali appended to the base filename. Defaults to None.
        show: If True, opens the plots in a browser. Defaults to True.
        **kwargs: Additional keyword arguments.
    """

    fig_3d = plot_trajectory_3d_plotly(data["pos"], show=False)
    fig_2d = plot_trajectory_2d_plotly(data["pos"], show=False)
    fig_sali = plot_sali_plotly(
        data["t"], data["sali"], threshold=threshold, show=False
    )

    if show:
        fig_3d.show()
        fig_2d.show()
        fig_sali.show()

    if save_path:
        base_name = save_path.rsplit(".", 1)[0]
        ext = save_path.rsplit(".", 1)[1] if "." in save_path else "html"
        fig_3d.write_html(f"{base_name}_3d.{ext}")
        fig_2d.write_html(f"{base_name}_2d.{ext}")
        fig_sali.write_html(f"{base_name}_sali.{ext}")
