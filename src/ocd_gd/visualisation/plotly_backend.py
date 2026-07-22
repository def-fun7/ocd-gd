"""
Plotly backend for interactive visualization of galactic orbit dynamics and chaos indicators.
"""

from typing import Optional, List, Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import resolve_save_path


def _handle_save_show(
    fig: go.Figure,
    save_path: Optional[str] = None,
    backend: str = "plotly",
    show: bool = True,
    **kwargs,
) -> None:
    """Helper utility to save static images (via kaleido or html) and trigger display."""
    if save_path:
        save_path = resolve_save_path(save_path, backend)
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            # Requires `kaleido` package for static exports (png, pdf, svg)
            width = kwargs.get("width", 900)
            height = kwargs.get("height", 600)
            scale = kwargs.get("scale", 2)
            fig.write_image(save_path, width=width, height=height, scale=scale)
    if show:
        fig.show()


# ==============================================================================
# 1. SALI vs Time
# ==============================================================================
def plot_sali_plotly(
    t: np.ndarray,
    sali: np.ndarray,
    threshold: Optional[float] = 1e-8,
    is_chaotic: Optional[bool] = None,
    detection_time: Optional[float] = None,
    window_size_time: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot interactive SALI vs Time on a log scale."""
    fig = go.Figure()

    color = kwargs.get("color", "crimson")
    name = kwargs.get("name", "SALI")

    fig.add_trace(
        go.Scatter(
            x=t,
            y=sali,
            mode="lines",
            name=name,
            line=dict(color=color, width=kwargs.get("width", 2)),
        )
    )

    if threshold is not None:
        thresh_color = kwargs.get("thresh_color", "gray")
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=thresh_color,
            annotation_text=f"Threshold ({threshold:.0e})",
            annotation_position="bottom right",
        )
    if (
        is_chaotic
        and detection_time is not None
        and np.isfinite(detection_time)
        and window_size_time is not None
    ):
        t_start = max(t[0], detection_time - window_size_time)
        t_end = detection_time

        fig.add_vrect(
            x0=t_start,
            x1=t_end,
            fillcolor="orange",
            opacity=0.3,
            line_width=1,
            line_color="orange",
            annotation_text="Detection Window",
            annotation_position="top left",
        )
    base_title = kwargs.get("title", "SALI vs Time")
    if is_chaotic is not None:
        status_str = "Chaotic" if is_chaotic else "Regular"
        title_text = f"{base_title} <b>[{status_str}]</b>"
    else:
        title_text = base_title
    fig.update_layout(
        title=title_text,
        xaxis_title=kwargs.get("xlabel", "Time"),
        yaxis_title=kwargs.get("ylabel", "SALI"),
        yaxis_type="log",
        template=kwargs.get("template", "plotly_white"),
        width=kwargs.get("fig_width", 800),
        height=kwargs.get("fig_height", 500),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig


# ==============================================================================
# 2. GALI vs Time
# ==============================================================================
def plot_gali_plotly(
    t: np.ndarray,
    gali: np.ndarray,
    k_orders: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot interactive GALI_k vs Time on a log scale."""
    fig = go.Figure()

    if gali.ndim == 1:
        name = f"GALI_{k_orders[0]}" if k_orders else "GALI"
        fig.add_trace(
            go.Scatter(x=t, y=gali, mode="lines", name=name, line=dict(width=2))
        )
    else:
        num_k = gali.shape[1]
        for i in range(num_k):
            lbl = (
                f"GALI_{k_orders[i]}"
                if k_orders and i < len(k_orders)
                else f"GALI_{i+2}"
            )
            fig.add_trace(
                go.Scatter(
                    x=t, y=gali[:, i], mode="lines", name=lbl, line=dict(width=2)
                )
            )

    fig.update_layout(
        title=kwargs.get("title", "GALI vs Time"),
        xaxis_title=kwargs.get("xlabel", "Time"),
        yaxis_title=kwargs.get("ylabel", "GALI"),
        yaxis_type="log",
        template=kwargs.get("template", "plotly_white"),
        width=kwargs.get("fig_width", 800),
        height=kwargs.get("fig_height", 500),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig


# ==============================================================================
# 3. Orbit Trajectory Plot (Face-On & Edge-On Side-by-Side)
# ==============================================================================
def plot_trajectory_2d_plotly(
    pos: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot 2D projections of the orbit side-by-side: Face-On (X-Y) and Edge-On (X-Z)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            kwargs.get("title_faceon", "Face-On (X - Y)"),
            kwargs.get("title_edgeon", "Edge-On (X - Z)"),
        ),
    )

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    color = kwargs.get("color", "navy")

    # Face-On
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="Face-On", line=dict(color=color, width=1)
        ),
        row=1,
        col=1,
    )
    # Edge-On
    fig.add_trace(
        go.Scatter(
            x=x, y=z, mode="lines", name="Edge-On", line=dict(color=color, width=1)
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="X", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)

    fig.update_xaxes(title_text="X", scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=2)

    fig.update_layout(
        title_text=kwargs.get("title", "Orbit 2D Projections"),
        template=kwargs.get("template", "plotly_white"),
        showlegend=False,
        width=kwargs.get("fig_width", 1000),
        height=kwargs.get("fig_height", 500),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig


# ==============================================================================
# 4. 3D Orbit Plot
# ==============================================================================
def plot_trajectory_3d_plotly(
    pos: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot interactive 3D spatial orbit trajectory."""
    fig = go.Figure()

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    color = kwargs.get("color", "teal")

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            name="Trajectory",
            line=dict(color=color, width=3),
        )
    )

    if kwargs.get("mark_endpoints", True):
        fig.add_trace(
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                name="Start",
                marker=dict(color="green", size=6),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x[-1]],
                y=[y[-1]],
                z=[z[-1]],
                mode="markers",
                name="End",
                marker=dict(color="red", size=6),
            )
        )

    fig.update_layout(
        title=kwargs.get("title", "3D Orbit Trajectory"),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        template=kwargs.get("template", "plotly_white"),
        width=kwargs.get("fig_width", 800),
        height=kwargs.get("fig_height", 700),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig


# ==============================================================================
# 5. Phase Space Plot (Position vs Velocity)
# ==============================================================================
def plot_phase_space_plotly(
    pos: np.ndarray,
    vel: np.ndarray,
    plane: str = "x",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot 2D phase space scatter projection (X vs V_x, Y vs V_y, or Z vs V_z)."""
    fig = go.Figure()

    idx = {"x": 0, "y": 1, "z": 2}.get(plane.lower(), 0)
    q = pos[:, idx]
    v = vel[:, idx]

    color = kwargs.get("color", "purple")

    fig.add_trace(
        go.Scatter(
            x=q,
            y=v,
            mode="markers",
            name=f"{plane.upper()} Phase Space",
            marker=dict(
                color=color,
                size=kwargs.get("size", 3),
                opacity=kwargs.get("opacity", 0.6),
            ),
        )
    )

    fig.update_layout(
        title=kwargs.get(
            "title", f"Phase Space ({plane.upper()} vs V_{plane.lower()})"
        ),
        xaxis_title=kwargs.get("xlabel", f"{plane.upper()}"),
        yaxis_title=kwargs.get("ylabel", f"V_{plane.lower()}"),
        template=kwargs.get("template", "plotly_white"),
        width=kwargs.get("fig_width", 800),
        height=kwargs.get("fig_height", 500),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig


# ==============================================================================
# 6. Fractional Energy Drift vs Time
# ==============================================================================
def plot_energy_drift_plotly(
    t: np.ndarray,
    energy: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot fractional energy conservation error: |(E(t) - E_0) / E_0| over time."""
    fig = go.Figure()

    e0 = energy[0]
    drift = np.abs((energy - e0) / e0)
    color = kwargs.get("color", "darkgreen")

    fig.add_trace(
        go.Scatter(
            x=t,
            y=drift,
            mode="lines",
            name="|(E(t) - E₀) / E₀|",
            line=dict(color=color, width=kwargs.get("width", 2)),
        )
    )

    yaxis_type = "log" if kwargs.get("log_scale", True) else "linear"

    fig.update_layout(
        title=kwargs.get("title", "Fractional Energy Drift"),
        xaxis_title=kwargs.get("xlabel", "Time"),
        yaxis_title=kwargs.get("ylabel", "|(E(t) - E₀) / E₀|"),
        yaxis_type=yaxis_type,
        template=kwargs.get("template", "plotly_white"),
        width=kwargs.get("fig_width", 800),
        height=kwargs.get("fig_height", 500),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig


# ==============================================================================
# 7. Color-Coded Trajectory Plot (Colored by Time or SALI)
# ==============================================================================
def plot_colored_trajectory_2d_plotly(
    pos: np.ndarray,
    c_values: np.ndarray,
    c_label: str = "Time",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot 2D Face-On (X-Y) trajectory colored continuously by a scalar array."""
    fig = go.Figure()

    x, y = pos[:, 0], pos[:, 1]
    colorscale = kwargs.get("colorscale", "Viridis")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(
                size=2,
                color=c_values,
                colorscale=colorscale,
                colorbar=dict(title=c_label),
                showscale=True,
            ),
            line=dict(color="rgba(150,150,150,0.3)", width=1),
        )
    )

    fig.update_layout(
        title=kwargs.get("title", f"Trajectory Colored by {c_label}"),
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        template=kwargs.get("template", "plotly_white"),
        width=kwargs.get("fig_width", 800),
        height=kwargs.get("fig_height", 650),
    )

    _handle_save_show(fig, save_path=save_path, show=show, **kwargs)
    return fig
