"""
Matplotlib backend for chaos-map plotting.
"""

__all__ = [
    "plot_chaos_maps_mpl",
    "plot_composite_chaos_map_mpl",
    "plot_consensus_chaos_map_mpl",
]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .grid_constants import (
    COMPOSITE_FIGSIZE_MPL,
    COMPOSITE_RESONANCE_LINESTYLE_MPL,
    COMPOSITE_RESONANCE_LINEWIDTH_MPL,
    CONSENSUS_FIGSIZE_MPL,
    CONSENSUS_LABELS,
    FAMILY_BOUNDARY_LABEL,
    FAMILY_BOUNDARY_LINESTYLE_MPL,
    FAMILY_BOUNDARY_LINEWIDTH_MPL,
    MPL_LEGEND_KWARGS,
    RESONANCE_LINESTYLE_MPL,
    RESONANCE_LINEWIDTH,
    SIDE_BY_SIDE_FIGSIZE_MPL,
    SIDE_BY_SIDE_LEGEND_LABELS,
    ZVC_LABEL,
    ZVC_LINESTYLE_MPL,
    ZVC_LINEWIDTH,
)
from .grid_helpers import (
    _binary_grids_to_composite_rgb,
    _composite_legend_entries,
    _compute_consensus_grid,
    _compute_zvc,
    _family_boundary_field,
    _get_consensus_colors,
    _has_family_boundary,
    _resonance_overlay_specs,
)
from .grid_themes import DEFAULT_THEME, ChaosMapTheme, get_theme


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


def _add_zvc_mpl(
    ax: plt.Axes,
    x_vals: npt.NDArray[np.float64],
    v_zvc: npt.NDArray[np.float64],
    color: str,
) -> None:
    ax.plot(
        x_vals, v_zvc, color=color, linestyle=ZVC_LINESTYLE_MPL, linewidth=ZVC_LINEWIDTH
    )
    ax.plot(
        x_vals,
        -v_zvc,
        color=color,
        linestyle=ZVC_LINESTYLE_MPL,
        linewidth=ZVC_LINEWIDTH,
    )


def _add_resonance_mpl(
    ax: plt.Axes,
    resonance_specs: list[tuple[float, str, str]],
    *,
    linestyle: str = RESONANCE_LINESTYLE_MPL,
    linewidth: float = RESONANCE_LINEWIDTH,
) -> None:
    for radius, _, color in resonance_specs:
        ax.axvline(radius, color=color, linestyle=linestyle, linewidth=linewidth)
        ax.axvline(-radius, color=color, linestyle=linestyle, linewidth=linewidth)


def _add_family_boundary_mpl(
    ax: plt.Axes,
    x_vals: npt.NDArray[np.float64],
    v_x_vals: npt.NDArray[np.float64],
    family_field: npt.NDArray[np.float64] | None,
    color: str,
) -> bool:
    """Draw the box/loop boundary as a contour line. Returns True if a
    boundary was actually drawn."""
    if not _has_family_boundary(family_field):
        return False
    ax.contour(
        x_vals,
        v_x_vals,
        np.ma.masked_invalid(family_field),
        levels=[0.5],
        colors=[color],
        linewidths=FAMILY_BOUNDARY_LINEWIDTH_MPL,
        linestyles=FAMILY_BOUNDARY_LINESTYLE_MPL,
    )
    return True


def _zvc_legend_handle(color: str) -> Line2D:
    return Line2D(
        [0], [0], color=color, lw=ZVC_LINEWIDTH, ls=ZVC_LINESTYLE_MPL, label=ZVC_LABEL
    )


def _family_boundary_legend_handle(color: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=color,
        lw=FAMILY_BOUNDARY_LINEWIDTH_MPL,
        ls=FAMILY_BOUNDARY_LINESTYLE_MPL,
        label=FAMILY_BOUNDARY_LABEL,
    )


def _resonance_legend_handles(
    resonance_specs: list[tuple[float, str, str]],
    *,
    linestyle: str = RESONANCE_LINESTYLE_MPL,
    linewidth: float = RESONANCE_LINEWIDTH,
) -> list[Line2D]:
    return [
        Line2D([0], [0], color=color, lw=linewidth, ls=linestyle, label=label)
        for _, label, color in resonance_specs
    ]


def _build_side_by_side_legend_elements(
    theme: ChaosMapTheme,
    has_zvc: bool,
    resonance_specs: list[tuple[float, str, str]],
    has_family_boundary: bool,
) -> list:
    elements = [
        Patch(
            facecolor=theme.color_regular,
            edgecolor="none",
            label=SIDE_BY_SIDE_LEGEND_LABELS["regular"],
        ),
        Patch(
            facecolor=theme.color_chaotic,
            edgecolor="none",
            label=SIDE_BY_SIDE_LEGEND_LABELS["chaotic"],
        ),
        Patch(
            facecolor=theme.color_masked,
            edgecolor="none",
            label=SIDE_BY_SIDE_LEGEND_LABELS["masked"],
        ),
    ]
    if has_zvc:
        elements.append(_zvc_legend_handle(theme.zvc_color))
    if has_family_boundary:
        elements.append(_family_boundary_legend_handle(theme.family_boundary_color))
    elements.extend(_resonance_legend_handles(resonance_specs))
    return elements


def _build_composite_legend_elements(
    theme: ChaosMapTheme,
    has_zvc: bool,
    resonance_specs: list[tuple[float, str, str]],
    has_family_boundary: bool,
) -> list:
    elements = [
        Patch(facecolor=color, label=label)
        for label, color in _composite_legend_entries(theme)
    ]
    if has_zvc:
        elements.append(_zvc_legend_handle(theme.zvc_color))
    if has_family_boundary:
        elements.append(_family_boundary_legend_handle(theme.family_boundary_color))
    elements.extend(
        _resonance_legend_handles(
            resonance_specs,
            linestyle=COMPOSITE_RESONANCE_LINESTYLE_MPL,
            linewidth=COMPOSITE_RESONANCE_LINEWIDTH_MPL,
        )
    )
    return elements


def plot_chaos_maps_mpl(
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
) -> tuple[plt.Figure, npt.NDArray[np.float64]]:
    """Plot SALI, GALI, and Lyapunov 2D chaos maps side-by-side (1x3).

    Uses a clean, unified legend instead of a colorbar since each map is a
    binary regular/chaotic classification rather than a continuous quantity.

    Args:
        sali_grid: (H, W) array of SALI indicators (0 for regular, 1 for chaotic, NaN for masked).
        gali_grid: (H, W) array of GALI indicators (0 for regular, 1 for chaotic, NaN for masked).
        lyapunov_grid: (H, W) array of Lyapunov indicators (0 for regular, 1 for chaotic, NaN for masked).
        x_vals: Grid x coordinates.
        v_x_vals: Grid v_x coordinates.
        E_rem_vals: Residual energy at each x value. If provided, overlays the ZVC.
        resonance_radii: Resonance radii object containing attributes like corotation, etc.
        orbit_family_grid: Grid of box/loop family classifications.
        theme: Visual theme name (e.g. "magma") or a theme instance. Defaults to "magma".
        save_path: Path to export the figure. Defaults to None.
        show: If True, displays the figure. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[plt.Figure, npt.NDArray[np.float64]]: The generated figure and axes array.

    Examples:
        >>> import numpy as np
        >>> sali = np.zeros((10, 10))
        >>> gali = np.zeros((10, 10))
        >>> lyap = np.zeros((10, 10))
        >>> x = np.linspace(-1, 1, 10)
        >>> vx = np.linspace(-1, 1, 10)
        >>> fig, axes = plot_chaos_maps_mpl(sali, gali, lyap, x, vx, show=False)
    """
    th = get_theme(theme)
    grid_size_y, grid_size_x = sali_grid.shape
    dx = x_vals[1] - x_vals[0]
    dy = v_x_vals[1] - v_x_vals[0]
    extent = [
        x_vals[0] - dx / 2,
        x_vals[-1] + dx / 2,
        v_x_vals[0] - dy / 2,
        v_x_vals[-1] + dy / 2,
    ]

    cmap = ListedColormap([th.color_regular, th.color_chaotic])
    cmap.with_extremes(bad=th.color_masked)

    fig, axes = plt.subplots(
        1, 3, figsize=SIDE_BY_SIDE_FIGSIZE_MPL, sharex=True, sharey=True
    )

    v_zvc = _compute_zvc(E_rem_vals)
    resonance_specs = _resonance_overlay_specs(resonance_radii, th)
    family_field = _family_boundary_field(orbit_family_grid)
    family_drawn = False

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
            _add_zvc_mpl(ax, x_vals, v_zvc, th.zvc_color)
        _add_resonance_mpl(ax, resonance_specs)
        if _add_family_boundary_mpl(
            ax, x_vals, v_x_vals, family_field, th.family_boundary_color
        ):
            family_drawn = True
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_ylim(v_x_vals[0], v_x_vals[-1])

    axes[0].set_ylabel("$v_x$", fontsize=12)

    legend_elements = _build_side_by_side_legend_elements(
        th, v_zvc is not None, resonance_specs, family_drawn
    )
    axes[2].legend(
        handles=legend_elements, bbox_to_anchor=(1.0, 1.0), **MPL_LEGEND_KWARGS
    )

    _finalize_mpl_figure(fig, save_path, show)
    return fig, axes


def plot_composite_chaos_map_mpl(
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
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay SALI, GALI, and Lyapunov indicators into a single RGB composite chaos map.

    Channel mapping: Red = Lyapunov exponent flag, Green = GALI flag,
    Blue = SALI flag. A grid cell where none of the three fired keeps the
    regular-orbit background color; unphysical (NaN) cells use the theme's
    composite masked color.

    Args:
        sali_grid: (H, W) array of SALI indicators.
        gali_grid: (H, W) array of GALI indicators.
        lyapunov_grid: (H, W) array of Lyapunov indicators.
        x_vals: Grid x coordinates.
        v_x_vals: Grid v_x coordinates.
        E_rem_vals: Residual energy at each x value. If provided, overlays the ZVC.
        resonance_radii: Resonance radii object containing attributes like corotation, etc.
        orbit_family_grid: Grid of box/loop family classifications.
        theme: Visual theme name or theme instance. Defaults to "magma".
        save_path: Path to export the figure. Defaults to None.
        show: If True, displays the figure. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated figure and axes object.

    Examples:
        >>> import numpy as np
        >>> sali = np.zeros((10, 10))
        >>> gali = np.zeros((10, 10))
        >>> lyap = np.zeros((10, 10))
        >>> x = np.linspace(-1, 1, 10)
        >>> vx = np.linspace(-1, 1, 10)
        >>> fig, ax = plot_composite_chaos_map_mpl(sali, gali, lyap, x, vx, show=False)
    """
    th = get_theme(theme)
    rgb_map = _binary_grids_to_composite_rgb(sali_grid, gali_grid, lyapunov_grid, th)
    grid_shape = sali_grid.shape
    dx = x_vals[1] - x_vals[0]
    dy = v_x_vals[1] - v_x_vals[0]
    extent = [
        x_vals[0] - dx / 2,
        x_vals[-1] + dx / 2,
        v_x_vals[0] - dy / 2,
        v_x_vals[-1] + dy / 2,
    ]
    v_zvc = _compute_zvc(E_rem_vals)
    resonance_specs = _resonance_overlay_specs(resonance_radii, th)
    family_field = _family_boundary_field(orbit_family_grid)

    fig, ax = plt.subplots(figsize=COMPOSITE_FIGSIZE_MPL)
    ax.imshow(
        rgb_map, extent=extent, origin="lower", aspect="auto", interpolation="nearest"
    )

    if v_zvc is not None:
        _add_zvc_mpl(ax, x_vals, v_zvc, th.zvc_color)
    _add_resonance_mpl(
        ax,
        resonance_specs,
        linestyle=COMPOSITE_RESONANCE_LINESTYLE_MPL,
        linewidth=COMPOSITE_RESONANCE_LINEWIDTH_MPL,
    )
    family_drawn = _add_family_boundary_mpl(
        ax, x_vals, v_x_vals, family_field, th.family_boundary_color
    )

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$v_x$", fontsize=12)
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(v_x_vals[0], v_x_vals[-1])
    ax.set_title(
        f"Composite Chaos Overlay ({grid_shape[1]}x{grid_shape[0]})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    legend_elements = _build_composite_legend_elements(
        th, v_zvc is not None, resonance_specs, family_drawn
    )
    ax.legend(handles=legend_elements, **MPL_LEGEND_KWARGS)

    _finalize_mpl_figure(fig, save_path, show)
    return fig, ax


def plot_consensus_chaos_map_mpl(
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
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an 8-state discrete classification map showing exact (L, S, G) combinations.

    Args:
        sali_grid: (H, W) array of SALI indicators.
        gali_grid: (H, W) array of GALI indicators.
        lyapunov_grid: (H, W) array of Lyapunov indicators.
        x_vals: Grid x coordinates.
        v_x_vals: Grid v_x coordinates.
        E_rem_vals: Residual energy at each x value. If provided, overlays the ZVC.
        resonance_radii: Resonance radii object containing attributes like corotation, etc.
        orbit_family_grid: Grid of box/loop family classifications.
        theme: Visual theme name or theme instance. Defaults to "magma".
        save_path: Path to export the figure. Defaults to None.
        show: If True, displays the figure. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[plt.Figure, plt.Axes]: The generated figure and axes object.

    Examples:
        >>> import numpy as np
        >>> sali = np.zeros((10, 10))
        >>> gali = np.zeros((10, 10))
        >>> lyap = np.zeros((10, 10))
        >>> x = np.linspace(-1, 1, 10)
        >>> vx = np.linspace(-1, 1, 10)
        >>> fig, ax = plot_consensus_chaos_map_mpl(sali, gali, lyap, x, vx, show=False)
    """
    th = get_theme(theme)
    consensus_grid = _compute_consensus_grid(sali_grid, gali_grid, lyapunov_grid)
    grid_shape = sali_grid.shape
    dx = x_vals[1] - x_vals[0]
    dy = v_x_vals[1] - v_x_vals[0]
    extent = [
        x_vals[0] - dx / 2,
        x_vals[-1] + dx / 2,
        v_x_vals[0] - dy / 2,
        v_x_vals[-1] + dy / 2,
    ]

    v_zvc = _compute_zvc(E_rem_vals)
    resonance_specs = _resonance_overlay_specs(resonance_radii, th)
    family_field = _family_boundary_field(orbit_family_grid)

    colors = _get_consensus_colors(th)
    cmap = ListedColormap(colors)
    cmap.with_extremes(bad=th.composite_masked_color)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, 8.5, 1.0), ncolors=8)

    fig, ax = plt.subplots(figsize=CONSENSUS_FIGSIZE_MPL)

    im = ax.imshow(
        consensus_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    if v_zvc is not None:
        _add_zvc_mpl(ax, x_vals, v_zvc, th.zvc_color)
    _add_resonance_mpl(
        ax,
        resonance_specs,
        linestyle=COMPOSITE_RESONANCE_LINESTYLE_MPL,
        linewidth=COMPOSITE_RESONANCE_LINEWIDTH_MPL,
    )
    family_drawn = _add_family_boundary_mpl(
        ax, x_vals, v_x_vals, family_field, th.family_boundary_color
    )

    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(v_x_vals[0], v_x_vals[-1])

    cbar = fig.colorbar(im, ax=ax, ticks=list(range(8)), pad=0.03, fraction=0.046)
    cbar.ax.set_yticklabels([CONSENSUS_LABELS[i] for i in range(8)])
    cbar.set_label(
        "Chaos Indicator State (Lyapunov, SALI, GALI)", fontsize=11, labelpad=10
    )

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$v_x$", fontsize=12)
    ax.set_title(
        f"Chaos State Classification ({grid_shape[1]}x{grid_shape[0]})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    line_legend_elements = []
    if v_zvc is not None:
        line_legend_elements.append(_zvc_legend_handle(th.zvc_color))
    if family_drawn:
        line_legend_elements.append(
            _family_boundary_legend_handle(th.family_boundary_color)
        )
    line_legend_elements.extend(
        _resonance_legend_handles(
            resonance_specs,
            linestyle=COMPOSITE_RESONANCE_LINESTYLE_MPL,
            linewidth=COMPOSITE_RESONANCE_LINEWIDTH_MPL,
        )
    )
    if line_legend_elements:
        ax.legend(handles=line_legend_elements, **MPL_LEGEND_KWARGS)

    _finalize_mpl_figure(fig, save_path, show)
    return fig, ax
