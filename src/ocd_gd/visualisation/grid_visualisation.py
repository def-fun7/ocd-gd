"""
Chaos-map rendering functions for GridChaosDetector.

Provides both matplotlib and plotly implementations of:
- the side-by-side SALI/GALI/Lyapunov map (`plot_chaos_maps_*`)
- the single RGB composite overlay of all three (`plot_composite_chaos_map_*`)

Kept as a standalone module (mirroring `.visualisation`) so `_grid_plotting.py`
can dispatch to either backend the same way the rest of the package does.
"""

__all__ = [
    "plot_chaos_maps_mpl",
    "plot_chaos_maps_plotly",
    "plot_composite_chaos_map_mpl",
    "plot_composite_chaos_map_plotly",
]

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plotly.subplots import make_subplots

# =============================================================================
# SHARED HELPERS
# =============================================================================


def _binary_grid_to_rgb(
    grid: npt.NDArray[np.float64],
    color_regular: str | tuple[float, float, float],
    color_chaotic: str | tuple[float, float, float],
    color_masked: str | tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    """Convert a single 0 (regular) / 1 (chaotic) / NaN (unphysical) grid into
    an (H, W, 3) RGB image array (float, range [0, 1])."""
    rgb = np.tile(np.array(to_rgb(color_masked)), (*grid.shape, 1))
    rgb[grid == 0] = to_rgb(color_regular)
    rgb[grid == 1] = to_rgb(color_chaotic)
    return rgb


def _binary_grids_to_composite_rgb(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    masked_color: str | tuple[float, float, float] = (0.2, 0.2, 0.2),
) -> npt.NDArray[np.float64]:
    """Vectorized construction of the composite RGB image from three 0/1/NaN
    grids (replaces a per-cell Python double loop with array operations).

    Channel mapping per pixel, matching `plot_composite_chaos_map_mpl`:
    Red = Lyapunov flag, Green = GALI flag, Blue = SALI flag, with a fixed
    "regular" background color where none of the three fired.
    """
    shape = sali_grid.shape
    nan_mask = np.isnan(sali_grid) | np.isnan(gali_grid) | np.isnan(lyapunov_grid)

    s = np.nan_to_num(sali_grid, nan=0).astype(bool)
    g = np.nan_to_num(gali_grid, nan=0).astype(bool)
    l = np.nan_to_num(lyapunov_grid, nan=0).astype(bool)
    any_flag = s | g | l

    rgb = np.tile(np.array([0.12, 0.31, 0.47]), (*shape, 1))
    r_channel = np.where(l, 0.9, 0.2)
    g_channel = np.where(g, 0.8, 0.2)
    b_channel = np.where(s, 0.9, 0.1)
    rgb[any_flag] = np.stack([r_channel, g_channel, b_channel], axis=-1)[any_flag]

    rgb[nan_mask] = to_rgb(masked_color)
    return rgb


def _finalize_mpl_figure(fig: plt.Figure, save_path: str | None, show: bool) -> None:
    """Shared save/show/close handling for the matplotlib chaos-map plots.

    Closes the figure when `show` is False so repeated calls (e.g. from
    `save_chaos_maps`) don't leave stray figures accumulating in memory.
    """
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


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


def _legend_proxy_traces(labels_and_colors: Sequence[tuple[str, str]]) -> list:
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


def _line_legend_proxy_traces(labels_and_colors: Sequence[tuple[str, str]]) -> list:
    """Same as `_legend_proxy_traces` but styled as a line, for the resonance
    radius overlays (`add_vline` shapes carry no legend entry either)."""
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": to_hex(color), "dash": "dot", "width": 1.5},
            name=label,
            showlegend=True,
        )
        for label, color in labels_and_colors
    ]


# Resonance field name -> (legend label, line color). Order controls legend order.
_RESONANCE_STYLES = {
    "corotation": ("Corotation Radius", "#2ca02c"),
    "inner_lindblad": ("Inner Lindblad (ILR)", "#9467bd"),
    "outer_lindblad": ("Outer Lindblad (OLR)", "#8c564b"),
}


def _resonance_overlay_specs(resonance_radii) -> list:
    """Return (radius, label, color) for each resonance radius that was
    actually found (skips any that are None — e.g. a non-rotating potential
    has no corotation/Lindblad radii at all)."""
    if resonance_radii is None:
        return []
    specs = []
    for field, (label, color) in _RESONANCE_STYLES.items():
        radius = getattr(resonance_radii, field, None)
        if radius is not None:
            specs.append((radius, label, color))
    return specs


# =============================================================================
# SIDE-BY-SIDE CHAOS MAPS
# =============================================================================


def plot_chaos_maps_mpl(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    cmap_colors: Sequence[str] = ("#1f4e78", "#f2c811"),  # [Regular (0), Chaotic (1)]
    masked_color: str = "#333333",  # unphysical (NaN) regions
    save_path: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, npt.NDArray[np.float64]]:
    """Plot SALI, GALI, and Lyapunov 2D chaos maps side-by-side (1x3).

    Uses a clean, unified legend instead of a colorbar since each map is a
    binary regular/chaotic classification rather than a continuous quantity.

    Parameters
    ----------
    sali_grid, gali_grid, lyapunov_grid : ndarray
        (grid_size, grid_size) arrays of 0 (regular) / 1 (chaotic) / NaN
        (unphysical) classifications.
    x_vals, v_x_vals : ndarray
        Grid axis coordinates.
    E_rem_vals : ndarray, optional
        Residual energy at each x value; if given, overlays the
        zero-velocity curve (ZVC).
    resonance_radii : ResonanceRadii, optional
        Corotation/Lindblad radii to overlay as mirrored (±R) vertical
        lines; any field left as None is simply skipped.
    cmap_colors : sequence of str, default ("#1f4e78", "#f2c811")
        [regular color, chaotic color].
    masked_color : str, default "#333333"
        Color for unphysical (NaN) grid cells.
    save_path : str, optional
        Path to export the figure.
    show : bool, default True
        Display the figure immediately.
    """
    grid_size_y, grid_size_x = sali_grid.shape
    extent = [x_vals[0], x_vals[-1], v_x_vals[0], v_x_vals[-1]]

    cmap = ListedColormap(list(cmap_colors))
    cmap.set_bad(color=masked_color)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)

    v_zvc = (
        np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0)) if E_rem_vals is not None else None
    )
    resonance_specs = _resonance_overlay_specs(resonance_radii)

    panel_configs = [
        (axes[0], sali_grid, f"SALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (axes[1], gali_grid, f"GALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (axes[2], lyapunov_grid, f"Lyapunov Map ({grid_size_x}x{grid_size_y})"),
    ]

    for ax, grid, title in panel_configs:
        ax.imshow(
            grid,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        if v_zvc is not None:
            ax.plot(x_vals, v_zvc, color="red", linestyle="--", linewidth=1.5)
            ax.plot(x_vals, -v_zvc, color="red", linestyle="--", linewidth=1.5)
        for radius, _, color in resonance_specs:
            ax.axvline(radius, color=color, linestyle=":", linewidth=1.3)
            ax.axvline(-radius, color=color, linestyle=":", linewidth=1.3)
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    axes[0].set_ylabel("$v_x$", fontsize=12)

    legend_elements = [
        Patch(facecolor=cmap_colors[0], edgecolor="none", label="Regular Orbit"),
        Patch(facecolor=cmap_colors[1], edgecolor="none", label="Chaotic Orbit"),
        Patch(facecolor=masked_color, edgecolor="none", label="Unphysical Domain"),
    ]
    if v_zvc is not None:
        legend_elements.append(
            Line2D([0], [0], color="red", lw=1.5, ls="--", label="Zero-Velocity Curve")
        )
    for radius, label, color in resonance_specs:
        legend_elements.append(
            Line2D([0], [0], color=color, lw=1.3, ls=":", label=label)
        )

    axes[2].legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        bbox_to_anchor=(1.0, 1.0),
    )

    _finalize_mpl_figure(fig, save_path, show)
    return fig, axes


def plot_chaos_maps_plotly(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    cmap_colors: Sequence[str] = ("#1f4e78", "#f2c811"),
    masked_color: str = "#333333",
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Plotly counterpart of `plot_chaos_maps_mpl`.

    Renders each map as an RGB image (rather than a `Heatmap`) so the
    unphysical-region masking matches the matplotlib version exactly. See
    `plot_chaos_maps_mpl` for parameter and color/legend semantics.
    """
    grid_size_y, grid_size_x = sali_grid.shape
    color_regular, color_chaotic = cmap_colors

    panels = [
        (sali_grid, f"SALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (gali_grid, f"GALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (lyapunov_grid, f"Lyapunov Map ({grid_size_x}x{grid_size_y})"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for _, title in panels])

    dx = (x_vals[-1] - x_vals[0]) / max(grid_size_x - 1, 1)
    dy = (v_x_vals[-1] - v_x_vals[0]) / max(grid_size_y - 1, 1)
    v_zvc = (
        np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0)) if E_rem_vals is not None else None
    )
    resonance_specs = _resonance_overlay_specs(resonance_radii)

    for col, (grid, _) in enumerate(panels, start=1):
        rgb = (
            _binary_grid_to_rgb(grid, color_regular, color_chaotic, masked_color) * 255
        ).astype(np.uint8)
        fig.add_trace(
            go.Image(z=rgb, x0=x_vals[0], dx=dx, y0=v_x_vals[0], dy=dy),
            row=1,
            col=col,
        )
        if v_zvc is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=v_zvc,
                    mode="lines",
                    line={"color": "red", "dash": "dash", "width": 1.5},
                    name="Zero-Velocity Curve",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=-v_zvc,
                    mode="lines",
                    line={"color": "red", "dash": "dash", "width": 1.5},
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        for radius, _, color in resonance_specs:
            fig.add_vline(
                x=radius,
                line={"color": color, "dash": "dot", "width": 1.3},
                row=1,
                col=col,
            )
            fig.add_vline(
                x=-radius,
                line={"color": color, "dash": "dot", "width": 1.3},
                row=1,
                col=col,
            )

    legend_labels = [
        ("Regular Orbit", color_regular),
        ("Chaotic Orbit", color_chaotic),
        ("Unphysical Domain", masked_color),
    ]
    for trace in _legend_proxy_traces(legend_labels):
        fig.add_trace(trace, row=1, col=1)
    for trace in _line_legend_proxy_traces(
        [(label, color) for _, label, color in resonance_specs]
    ):
        fig.add_trace(trace, row=1, col=1)

    fig.update_layout(title="Chaos Maps", height=550, width=1400)

    _finalize_plotly_figure(fig, save_path, show)
    return fig


# =============================================================================
# COMPOSITE RGB CHAOS MAP
# =============================================================================


def plot_composite_chaos_map_mpl(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    masked_color: tuple[float, float, float] = (0.2, 0.2, 0.2),
    save_path: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay SALI, GALI, and Lyapunov indicators into a single RGB composite
    chaos map.

    Channel mapping: Red = Lyapunov exponent flag, Green = GALI flag,
    Blue = SALI flag. A grid cell where none of the three fired keeps the
    regular-orbit background color; unphysical (NaN) cells use `masked_color`.

    Parameters
    ----------
    sali_grid, gali_grid, lyapunov_grid : ndarray
        (grid_size, grid_size) arrays of 0/1/NaN classifications.
    x_vals, v_x_vals : ndarray
        Grid axis coordinates.
    E_rem_vals : ndarray, optional
        Residual energy at each x value; if given, overlays the ZVC.
    resonance_radii : ResonanceRadii, optional
        Corotation/Lindblad radii to overlay as mirrored (±R) vertical
        lines; any field left as None is simply skipped.
    masked_color : tuple of float, default (0.2, 0.2, 0.2)
        RGB color for unphysical (NaN) grid cells.
    save_path : str, optional
        Path to export the figure.
    show : bool, default True
        Display the figure immediately.
    """
    rgb_map = _binary_grids_to_composite_rgb(
        sali_grid, gali_grid, lyapunov_grid, masked_color
    )
    grid_shape = sali_grid.shape
    extent = [x_vals[0], x_vals[-1], v_x_vals[0], v_x_vals[-1]]
    resonance_specs = _resonance_overlay_specs(resonance_radii)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        rgb_map, extent=extent, origin="lower", aspect="auto", interpolation="nearest"
    )

    if E_rem_vals is not None:
        v_zvc = np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0))
        ax.plot(x_vals, v_zvc, color="red", linestyle="--", linewidth=1.5)
        ax.plot(x_vals, -v_zvc, color="red", linestyle="--", linewidth=1.5)

    for radius, _, color in resonance_specs:
        ax.axvline(radius, color=color, linestyle=":", linewidth=1.3)
        ax.axvline(-radius, color=color, linestyle=":", linewidth=1.3)

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$v_x$", fontsize=12)
    ax.set_title(
        f"Composite Chaos Overlay ({grid_shape[1]}x{grid_shape[0]})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    legend_elements = [
        Patch(facecolor=(0.12, 0.31, 0.47), label="Regular Orbit (All 0)"),
        Patch(facecolor=(0.9, 0.8, 0.9), label="All Indicators Agree (1,1,1)"),
        Patch(facecolor=(0.2, 0.8, 0.1), label="GALI Only"),
        Patch(facecolor=(0.9, 0.2, 0.1), label="Lyapunov Only"),
        Patch(facecolor=masked_color, label="Unphysical Domain"),
    ]
    if E_rem_vals is not None:
        legend_elements.append(
            Line2D([0], [0], color="red", lw=1.5, ls="--", label="Zero-Velocity Curve")
        )
    for radius, label, color in resonance_specs:
        legend_elements.append(
            Line2D([0], [0], color=color, lw=1.3, ls=":", label=label)
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        facecolor="#ffffff",
        edgecolor="#cccccc",
    )

    _finalize_mpl_figure(fig, save_path, show)
    return fig, ax


def plot_composite_chaos_map_plotly(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    E_rem_vals: npt.NDArray[np.float64] | None = None,
    resonance_radii=None,
    masked_color: tuple[float, float, float] = (0.2, 0.2, 0.2),
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Plotly counterpart of `plot_composite_chaos_map_mpl` — see its
    docstring for the channel-mapping and legend semantics."""
    grid_size_y, grid_size_x = sali_grid.shape
    rgb = (
        _binary_grids_to_composite_rgb(
            sali_grid, gali_grid, lyapunov_grid, masked_color
        )
        * 255
    ).astype(np.uint8)

    dx = (x_vals[-1] - x_vals[0]) / max(grid_size_x - 1, 1)
    dy = (v_x_vals[-1] - v_x_vals[0]) / max(grid_size_y - 1, 1)
    resonance_specs = _resonance_overlay_specs(resonance_radii)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb, x0=x_vals[0], dx=dx, y0=v_x_vals[0], dy=dy))

    if E_rem_vals is not None:
        v_zvc = np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=v_zvc,
                mode="lines",
                line={"color": "red", "dash": "dash", "width": 1.5},
                name="Zero-Velocity Curve",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=-v_zvc,
                mode="lines",
                line={"color": "red", "dash": "dash", "width": 1.5},
                showlegend=False,
            )
        )

    for radius, _, color in resonance_specs:
        fig.add_vline(x=radius, line={"color": color, "dash": "dot", "width": 1.3})
        fig.add_vline(x=-radius, line={"color": color, "dash": "dot", "width": 1.3})

    legend_labels = [
        ("Regular Orbit (All 0)", (0.12, 0.31, 0.47)),
        ("All Indicators Agree (1,1,1)", (0.9, 0.8, 0.9)),
        ("GALI Only", (0.2, 0.8, 0.1)),
        ("Lyapunov Only", (0.9, 0.2, 0.1)),
        ("Unphysical Domain", masked_color),
    ]
    for trace in _legend_proxy_traces(legend_labels):
        fig.add_trace(trace)
    for trace in _line_legend_proxy_traces(
        [(label, color) for _, label, color in resonance_specs]
    ):
        fig.add_trace(trace)

    fig.update_layout(
        title=f"Composite Chaos Overlay ({grid_size_x}x{grid_size_y})",
        xaxis_title="x",
        yaxis_title="v_x",
        height=650,
        width=750,
    )

    _finalize_plotly_figure(fig, save_path, show)
    return fig
