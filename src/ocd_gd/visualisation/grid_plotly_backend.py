"""
Plotly backend for chaos-map plotting.
"""

__all__ = [
    "plot_chaos_maps_plotly",
    "plot_composite_chaos_map_plotly",
    "plot_consensus_chaos_map_plotly",
]

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from matplotlib.colors import to_hex, to_rgb
from plotly.subplots import make_subplots

from .grid_constants import (
    COMPOSITE_LAYOUT_PLOTLY,
    CONSENSUS_LABELS,
    CONSENSUS_LAYOUT_PLOTLY,
    FAMILY_BOUNDARY_LABEL,
    FAMILY_BOUNDARY_LINESTYLE_PLOTLY,
    FAMILY_BOUNDARY_LINEWIDTH_PLOTLY,
    RESONANCE_LINESTYLE_PLOTLY,
    RESONANCE_LINEWIDTH,
    SIDE_BY_SIDE_LAYOUT_PLOTLY,
    SIDE_BY_SIDE_LEGEND_LABELS,
    ZVC_LABEL,
    ZVC_LINESTYLE_PLOTLY,
    ZVC_LINEWIDTH,
)
from .grid_helpers import (
    _binary_grid_to_rgb,
    _binary_grids_to_composite_rgb,
    _composite_legend_entries,
    _compute_consensus_grid,
    _compute_zvc,
    _consensus_grid_to_rgb,
    _family_boundary_field,
    _get_consensus_colors,
    _has_family_boundary,
    _resonance_overlay_specs,
)
from .grid_themes import DEFAULT_THEME, ChaosMapTheme, get_theme

# =============================================================================
# Chaos Maps: SHARED HELPERS
# =============================================================================


def _save_plotly_figure(fig: go.Figure, save_path: str) -> None:
    """Save a plotly figure as interactive HTML (`.html`) or a static image
    (any other extension, via kaleido)."""
    if str(save_path).lower().endswith(".html"):
        fig.write_html(save_path)
    else:
        fig.write_image(save_path)


def _finalize_plotly_figure(fig: go.Figure, save_path: str | None, show: bool) -> None:
    """Shared save/show handling for the plotly chaos-map plots."""
    if save_path:
        _save_plotly_figure(fig, save_path)
    if show:
        fig.show()


def _add_zvc_plotly(
    fig: go.Figure,
    x_vals: npt.NDArray[np.float64],
    v_zvc: npt.NDArray[np.float64],
    color: str,
    *,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    subplot_kwargs = {"row": row, "col": col} if row is not None else {}
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=v_zvc,
            mode="lines",
            line={"color": color, "dash": ZVC_LINESTYLE_PLOTLY, "width": ZVC_LINEWIDTH},
            name=ZVC_LABEL,
            showlegend=showlegend,
        ),
        **subplot_kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=-v_zvc,
            mode="lines",
            line={"color": color, "dash": ZVC_LINESTYLE_PLOTLY, "width": ZVC_LINEWIDTH},
            showlegend=False,
        ),
        **subplot_kwargs,
    )


def _add_resonance_plotly(
    fig: go.Figure,
    resonance_specs: list[tuple[float, str, str]],
    *,
    row: int | None = None,
    col: int | None = None,
    linestyle: str = RESONANCE_LINESTYLE_PLOTLY,
    linewidth: float = RESONANCE_LINEWIDTH,
) -> None:
    subplot_kwargs = {"row": row, "col": col} if row is not None else {}
    for radius, _, color in resonance_specs:
        fig.add_vline(
            x=radius,
            line={"color": color, "dash": linestyle, "width": linewidth},
            **subplot_kwargs,
        )
        fig.add_vline(
            x=-radius,
            line={"color": color, "dash": linestyle, "width": linewidth},
            **subplot_kwargs,
        )


def _add_family_boundary_plotly(
    fig: go.Figure,
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    family_field: npt.NDArray[np.float64] | None,
    color: str,
    *,
    row: int | None = None,
    col: int | None = None,
) -> bool:
    """Draw the box/loop boundary as a contour-line trace. Returns True if a
    boundary was actually drawn."""
    if not _has_family_boundary(family_field):
        return False
    subplot_kwargs = {"row": row, "col": col} if row is not None else {}
    fig.add_trace(
        go.Contour(
            x=x_vals,
            y=v_x_vals,
            z=family_field,
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 1,
                "coloring": "lines",
                "showlabels": False,
            },
            line={"color": color, "width": FAMILY_BOUNDARY_LINEWIDTH_PLOTLY},
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
            name=FAMILY_BOUNDARY_LABEL,
        ),
        **subplot_kwargs,
    )
    return True


def _legend_proxy_traces(
    labels_and_colors: Sequence[tuple[str, str | tuple[float, float, float]]],
) -> list:
    """Build invisible marker traces solely to populate a plotly legend.

    `go.Image` traces (used for the actual maps) carry no legend entry, so
    the discrete color key is faked with zero-data marker traces instead.
    """
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "color": to_hex(color), "symbol": "square"},
            name=label,
            showlegend=True,
        )
        for label, color in labels_and_colors
    ]


def _line_legend_proxy_traces(
    labels_and_colors: Sequence[tuple[str, str]],
    *,
    dash: str = RESONANCE_LINESTYLE_PLOTLY,
    width: float = RESONANCE_LINEWIDTH,
) -> list:
    """Same as `_legend_proxy_traces` but styled as a line, for overlays whose
    real trace carries no legend entry (`add_vline` shapes, `go.Contour` with
    `showlegend=False`)."""
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": to_hex(color), "dash": dash, "width": width},
            name=label,
            showlegend=True,
        )
        for label, color in labels_and_colors
    ]


# =============================================================================
# A. CHAOS MAPS: Side By Side
# =============================================================================


def plot_chaos_maps_plotly(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    orbit_family_grid: npt.NDArray[np.str_] | None = None,
    theme: str | ChaosMapTheme = DEFAULT_THEME,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plotly counterpart of `plot_chaos_maps_mpl`.

    Renders each map as an RGB image (rather than a `Heatmap`) so the
    unphysical-region masking matches the matplotlib version exactly. See
    `plot_chaos_maps_mpl` for parameter and color/legend semantics.
    """
    th = get_theme(theme)
    grid_size_y, grid_size_x = sali_grid.shape

    panels = [
        (sali_grid, f"SALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (gali_grid, f"GALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (lyapunov_grid, f"Lyapunov Map ({grid_size_x}x{grid_size_y})"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for _, title in panels])

    dx = (x_vals[-1] - x_vals[0]) / max(grid_size_x - 1, 1)
    dy = (v_x_vals[-1] - v_x_vals[0]) / max(grid_size_y - 1, 1)
    v_zvc = _compute_zvc(E_rem_vals)
    resonance_specs = _resonance_overlay_specs(resonance_radii, th)
    family_field = _family_boundary_field(orbit_family_grid)
    family_drawn = False

    for col, (grid, _) in enumerate(panels, start=1):
        rgb = (
            _binary_grid_to_rgb(
                grid, th.color_regular, th.color_chaotic, th.color_masked
            )
            * 255
        ).astype(np.uint8)
        fig.add_trace(
            go.Image(z=rgb, x0=x_vals[0], dx=dx, y0=v_x_vals[0], dy=dy),
            row=1,
            col=col,
        )
        if v_zvc is not None:
            _add_zvc_plotly(
                fig, x_vals, v_zvc, th.zvc_color, row=1, col=col, showlegend=(col == 1)
            )
        _add_resonance_plotly(fig, resonance_specs, row=1, col=col)
        if _add_family_boundary_plotly(
            fig,
            x_vals,
            v_x_vals,
            family_field,
            th.family_boundary_color,
            row=1,
            col=col,
        ):
            family_drawn = True

    legend_labels = [
        (SIDE_BY_SIDE_LEGEND_LABELS["regular"], th.color_regular),
        (SIDE_BY_SIDE_LEGEND_LABELS["chaotic"], th.color_chaotic),
        (SIDE_BY_SIDE_LEGEND_LABELS["masked"], th.color_masked),
    ]
    for trace in _legend_proxy_traces(legend_labels):
        fig.add_trace(trace, row=1, col=1)
    if family_drawn:
        for trace in _line_legend_proxy_traces(
            [(FAMILY_BOUNDARY_LABEL, th.family_boundary_color)],
            dash=FAMILY_BOUNDARY_LINESTYLE_PLOTLY,
            width=FAMILY_BOUNDARY_LINEWIDTH_PLOTLY,
        ):
            fig.add_trace(trace, row=1, col=1)
    for trace in _line_legend_proxy_traces(
        [(label, color) for _, label, color in resonance_specs]
    ):
        fig.add_trace(trace, row=1, col=1)

    fig.update_layout(title="Chaos Maps", **SIDE_BY_SIDE_LAYOUT_PLOTLY)

    _finalize_plotly_figure(fig, save_path, show)
    return fig


# =============================================================================
# B. CHAOS MAP: Composite
# =============================================================================


def plot_composite_chaos_map_plotly(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    orbit_family_grid: npt.NDArray[np.str_] | None = None,
    theme: str | ChaosMapTheme = DEFAULT_THEME,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plotly counterpart of `plot_composite_chaos_map_mpl` — see its
    docstring for the channel-mapping and legend semantics."""
    th = get_theme(theme)
    grid_size_y, grid_size_x = sali_grid.shape
    rgb = (
        _binary_grids_to_composite_rgb(sali_grid, gali_grid, lyapunov_grid, th) * 255
    ).astype(np.uint8)

    dx = (x_vals[-1] - x_vals[0]) / max(grid_size_x - 1, 1)
    dy = (v_x_vals[-1] - v_x_vals[0]) / max(grid_size_y - 1, 1)
    v_zvc = _compute_zvc(E_rem_vals)
    resonance_specs = _resonance_overlay_specs(resonance_radii, th)
    family_field = _family_boundary_field(orbit_family_grid)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb, x0=x_vals[0], dx=dx, y0=v_x_vals[0], dy=dy))

    if v_zvc is not None:
        _add_zvc_plotly(fig, x_vals, v_zvc, th.zvc_color)
    _add_resonance_plotly(fig, resonance_specs)
    family_drawn = _add_family_boundary_plotly(
        fig, x_vals, v_x_vals, family_field, th.family_boundary_color
    )

    for trace in _legend_proxy_traces(_composite_legend_entries(th)):
        fig.add_trace(trace)
    if family_drawn:
        for trace in _line_legend_proxy_traces(
            [(FAMILY_BOUNDARY_LABEL, th.family_boundary_color)],
            dash=FAMILY_BOUNDARY_LINESTYLE_PLOTLY,
            width=FAMILY_BOUNDARY_LINEWIDTH_PLOTLY,
        ):
            fig.add_trace(trace)
    for trace in _line_legend_proxy_traces(
        [(label, color) for _, label, color in resonance_specs]
    ):
        fig.add_trace(trace)

    fig.update_layout(
        title=f"Composite Chaos Overlay ({grid_size_x}x{grid_size_y})",
        xaxis_title="x",
        yaxis_title="v_x",
        **COMPOSITE_LAYOUT_PLOTLY,
    )

    _finalize_plotly_figure(fig, save_path, show)
    return fig


# =============================================================================
# C. CHAOS MAP: Consensus
# =============================================================================


def plot_consensus_chaos_map_plotly(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    orbit_family_grid: npt.NDArray[np.str_] | None = None,
    theme: str | ChaosMapTheme = DEFAULT_THEME,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Plotly counterpart of `plot_consensus_chaos_map_mpl`."""
    th = get_theme(theme)
    grid_size_y, grid_size_x = sali_grid.shape
    consensus_grid = _compute_consensus_grid(sali_grid, gali_grid, lyapunov_grid)

    rgb = (_consensus_grid_to_rgb(consensus_grid, th) * 255).astype(np.uint8)

    dx = (x_vals[-1] - x_vals[0]) / max(grid_size_x - 1, 1)
    dy = (v_x_vals[-1] - v_x_vals[0]) / max(grid_size_y - 1, 1)
    v_zvc = _compute_zvc(E_rem_vals)
    resonance_specs = _resonance_overlay_specs(resonance_radii, th)
    family_field = _family_boundary_field(orbit_family_grid)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb, x0=x_vals[0], dx=dx, y0=v_x_vals[0], dy=dy))

    if v_zvc is not None:
        _add_zvc_plotly(fig, x_vals, v_zvc, th.zvc_color)
    _add_resonance_plotly(fig, resonance_specs)
    family_drawn = _add_family_boundary_plotly(
        fig, x_vals, v_x_vals, family_field, th.family_boundary_color
    )

    # Consensus level color swatches in legend
    colors = _get_consensus_colors(th)
    consensus_legend_entries = [(CONSENSUS_LABELS[i], colors[i]) for i in range(4)]
    consensus_legend_entries.append(
        ("Unphysical Domain", to_rgb(th.composite_masked_color))
    )

    for trace in _legend_proxy_traces(consensus_legend_entries):
        fig.add_trace(trace)

    if family_drawn:
        for trace in _line_legend_proxy_traces(
            [(FAMILY_BOUNDARY_LABEL, th.family_boundary_color)],
            dash=FAMILY_BOUNDARY_LINESTYLE_PLOTLY,
            width=FAMILY_BOUNDARY_LINEWIDTH_PLOTLY,
        ):
            fig.add_trace(trace)
    for trace in _line_legend_proxy_traces(
        [(label, color) for _, label, color in resonance_specs]
    ):
        fig.add_trace(trace)

    fig.update_layout(
        title=f"Chaos Consensus Map ({grid_size_x}x{grid_size_y})",
        xaxis_title="x",
        yaxis_title="v_x",
        **CONSENSUS_LAYOUT_PLOTLY,
    )

    _finalize_plotly_figure(fig, save_path, show)
    return fig
