"""
Multi-panel diagnostic dashboard for orbit dynamics and chaos indicators.
"""

from typing import Optional, Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from .matplotlib_backend import (
    plot_sali_mpl,
    plot_gali_mpl,
    plot_trajectory_2d_mpl,
    plot_trajectory_3d_mpl,
    plot_energy_drift_mpl,
    _handle_save_show,
)
from .plotly_backend import (
    plot_sali_plotly,
    plot_gali_plotly,
    plot_trajectory_2d_plotly,
    plot_trajectory_3d_plotly,
    plot_energy_drift_plotly,
)


def plot_dashboard_mpl(
    data: Dict[str, Any],
    sali_threshold: float = 1e-2,
    gali_threshold: float = 1e-16,
    k_orders: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Generate a 4-panel Matplotlib summary dashboard."""
    fig = plt.figure(figsize=kwargs.get("figsize", (14, 10)))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_2d_face = fig.add_subplot(gs[0, 1])
    ax_sali = fig.add_subplot(gs[1, 0])
    ax_gali = fig.add_subplot(gs[1, 1])

    t = data["t"]
    pos = data["pos"]

    # Panel 1: 3D Trajectory
    plot_trajectory_3d_mpl(
        pos,
        fig=fig,
        ax=ax_3d,
        show=False,
        mark_endpoints=True,
        title="3D Orbit Trajectory",
    )

    # Panel 2: Face-On (X-Y) Trajectory
    ax_2d_face.plot(pos[:, 0], pos[:, 1], color="navy", lw=0.8, alpha=0.7)
    ax_2d_face.set_xlabel("X")
    ax_2d_face.set_ylabel("Y")
    ax_2d_face.set_title("Face-On Projection (X - Y)")
    ax_2d_face.set_aspect("equal", adjustable="datalim")
    ax_2d_face.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: SALI vs Time (Log scale + Window Box + Lyapunov Box)
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

    # Panel 4: GALI vs Time (Log scale + Window Box + Lyapunov Box)
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
    data: Dict[str, np.ndarray],
    threshold: float = 1e-8,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> None:
    """
    Generates interactive Plotly plots as sequential views or combined views.
    """
    # Note: Plotly multi-panel dashboards mixing 3D scenes with 2D Cartesian axes
    # are cleanest when triggered or saved per figure panel.
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
