"""
Chaos-map rendering functions for GridChaosDetector.

Provides both matplotlib and plotly implementations of:
- the side-by-side SALI/GALI/Lyapunov map (`plot_chaos_maps_*`)
- the single RGB composite overlay of all three (`plot_composite_chaos_map_*`)

Kept as a standalone module (mirroring `.visualisation`) so `_grid_plotting.py`
can dispatch to either backend the same way the rest of the package does.

Visual styling (colors, resonance/ZVC/family-boundary line colors) is picked
via the `theme` parameter on each public function and resolved through
`theme.py` — see that module to add a new look. Everything here that is
*not* color (legend text, figure sizing, dash patterns) lives in the
STYLE CONSTANTS block below and is the same for every theme.
"""

__all__ = [
    "_binary_grid_to_rgb",
    "_binary_grids_to_composite_rgb",
    "_composite_flag_rgb",
    "_composite_legend_entries",
    "_compute_consensus_grid",
    "_compute_zvc",
    "_consensus_grid_to_rgb",
    "_family_boundary_field",
    "_get_consensus_colors",
    "_has_family_boundary",
    "_resonance_overlay_specs",
]


import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_rgb

from .grid_constants import (
    _COMPOSITE_LEGEND_FLAGS,
    _FAMILY_BOX_LABEL,
    _FAMILY_LOOP_LABEL,
    COMPOSITE_LEGEND_LABELS,
    RESONANCE_LABELS,
)
from .grid_themes import ChaosMapTheme, _hex_to_rgb01


def _binary_grid_to_rgb(
    grid: npt.NDArray[np.float64],
    color_regular: str | tuple[float, float, float],
    color_chaotic: str | tuple[float, float, float],
    color_masked: str | tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    """Convert a single 0 (regular) / 1 (chaotic) / NaN (unphysical) grid into an (H, W, 3) RGB image array.

    Args:
        grid: A 2D array representing the orbit chaos mapping (0, 1, or NaN).
        color_regular: Color for regular orbits.
        color_chaotic: Color for chaotic orbits.
        color_masked: Color for unphysical/masked domains.

    Returns:
        npt.NDArray[np.float64]: An (H, W, 3) RGB float array in the range [0, 1].
    """
    rgb = np.tile(np.array(to_rgb(color_masked)), (*grid.shape, 1))
    rgb[grid == 0] = to_rgb(color_regular)
    rgb[grid == 1] = to_rgb(color_chaotic)
    return rgb


def _composite_flag_rgb(
    sali: bool, gali: bool, lyapunov: bool, theme: ChaosMapTheme
) -> tuple[float, float, float]:
    """Compute the RGB color for a combination of (SALI, GALI, Lyapunov) flags.

    Uses the theme's light(regular)->dark(chaotic) vote ramp.

    Args:
        sali: Whether the SALI indicator flagged the orbit as chaotic.
        gali: Whether the GALI indicator flagged the orbit as chaotic.
        lyapunov: Whether the Lyapunov exponent flagged the orbit as chaotic.
        theme: The visualization theme object.

    Returns:
        tuple[float, float, float]: The calculated RGB color as a tuple of floats.
    """
    return theme.get_state_color(sali, gali, lyapunov)


def _binary_grids_to_composite_rgb(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    theme: ChaosMapTheme,
) -> npt.NDArray[np.float64]:
    """Construct the themed RGB composite chaos map.

    Vectorized version of `_composite_flag_rgb`. Each cell's color is interpolated
    along the theme's regular->chaotic ramp by how many of the three indicators
    (SALI, GALI, Lyapunov) flagged it chaotic.

    Args:
        sali_grid: Binary/NaN grid for SALI.
        gali_grid: Binary/NaN grid for GALI.
        lyapunov_grid: Binary/NaN grid for Lyapunov.
        theme: The visualization theme object.

    Returns:
        npt.NDArray[np.float64]: An (H, W, 3) RGB float array.
    """
    nan_mask = np.isnan(sali_grid) | np.isnan(gali_grid) | np.isnan(lyapunov_grid)

    s = np.nan_to_num(sali_grid, nan=0).astype(bool)
    g = np.nan_to_num(gali_grid, nan=0).astype(bool)
    l = np.nan_to_num(lyapunov_grid, nan=0).astype(bool)

    n_chaotic = s.astype(np.int8) + g.astype(np.int8) + l.astype(np.int8)

    lo = np.array(_hex_to_rgb01(theme.composite_regular_color))
    hi = np.array(_hex_to_rgb01(theme.composite_chaotic_color))

    t = (n_chaotic / 3.0)[..., np.newaxis]
    rgb = lo + (hi - lo) * t

    rgb[nan_mask] = to_rgb(theme.composite_masked_color)
    return rgb


def _compute_zvc(
    E_rem_vals: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64] | None:
    """Zero-velocity curve from residual energy.

    Args:
        E_rem_vals: Residual energy array or None.

    Returns:
        npt.NDArray[np.float64] | None: Zero-velocity curve coordinates, or None.
    """
    if E_rem_vals is None:
        return None
    return np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0))


def _resonance_overlay_specs(
    resonance_radii, theme: ChaosMapTheme
) -> list[tuple[float, str, str]]:
    """Return (radius, label, color) for each resonance radius.

    Skips any that are None (e.g. a non-rotating potential has no corotation/Lindblad
    radii at all).

    Args:
        resonance_radii: Resonance radii object containing attributes like corotation, etc.
        theme: The visualization theme object.

    Returns:
        list[tuple[float, str, str]]: List of tuples containing radius, label, and color.
    """
    if resonance_radii is None:
        return []
    specs = []
    for field_name, label in RESONANCE_LABELS.items():
        radius = getattr(resonance_radii, field_name, None)
        if radius is not None:
            color = theme.resonance_colors.get(field_name, "#000000")
            specs.append((radius, label, color))
    return specs


def _family_boundary_field(
    orbit_family_grid: npt.NDArray[np.str_] | None,
) -> npt.NDArray[np.float64] | None:
    """Numeric 0 (box) / 1 (loop) / NaN field used to trace the box/loop boundary.

    Any entry that is neither `"box"` nor `"loop"` (e.g. an unphysical cell
    with no family classification) becomes NaN and is simply left as a gap
    in the boundary line, same as unphysical cells elsewhere on these maps.

    Args:
        orbit_family_grid: String grid of orbit family classifications.

    Returns:
        npt.NDArray[np.float64] | None: The family boundary numeric field, or None.
    """
    if orbit_family_grid is None:
        return None
    field = np.full(orbit_family_grid.shape, np.nan)
    field[orbit_family_grid == _FAMILY_LOOP_LABEL] = 1.0
    field[orbit_family_grid == _FAMILY_BOX_LABEL] = 0.0
    return field


def _has_family_boundary(family_field: npt.NDArray[np.float64] | None) -> bool:
    """Whether a box/loop boundary actually exists to draw.

    Both families must be present somewhere in the grid.

    Args:
        family_field: The numeric family boundary field.

    Returns:
        bool: True if boundary exists, False otherwise.
    """
    if family_field is None:
        return False
    finite = family_field[np.isfinite(family_field)]
    return finite.size > 0 and finite.min() != finite.max()


def _compute_consensus_grid(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Encode binary indicator grids into an integer state code from 0 to 7.

    Bit encoding order: (Lyapunov, SALI, GALI)
    code = (Lyapunov << 2) | (SALI << 1) | GALI
    Unphysical cells (containing NaN) remain NaN.

    Args:
        sali_grid: Binary/NaN grid for SALI.
        gali_grid: Binary/NaN grid for GALI.
        lyapunov_grid: Binary/NaN grid for Lyapunov.

    Returns:
        npt.NDArray[np.float64]: The encoded state code grid.
    """
    nan_mask = np.isnan(sali_grid) | np.isnan(gali_grid) | np.isnan(lyapunov_grid)

    s = np.nan_to_num(sali_grid, nan=0.0).astype(int)
    g = np.nan_to_num(gali_grid, nan=0.0).astype(int)
    l = np.nan_to_num(lyapunov_grid, nan=0.0).astype(int)

    code = (l << 2) | (s << 1) | g

    consensus = code.astype(np.float64)
    consensus[nan_mask] = np.nan
    return consensus


def _get_consensus_colors(theme: ChaosMapTheme) -> list[tuple[float, float, float]]:
    """Return RGB colors for all 8 discrete states (0..7) derived from `theme`.

    Calls through `_composite_flag_rgb` so the exact RGB channel definition
    is evaluated for each combination.

    Args:
        theme: The visualization theme object.

    Returns:
        list[tuple[float, float, float]]: RGB colors for each discrete state.
    """
    colors = []
    for code in range(8):
        l = bool(code & 4)
        s = bool(code & 2)
        g = bool(code & 1)

        colors.append(_composite_flag_rgb(sali=s, gali=g, lyapunov=l, theme=theme))
    return colors


def _consensus_grid_to_rgb(
    consensus_grid: npt.NDArray[np.float64],
    theme: ChaosMapTheme,
) -> npt.NDArray[np.float64]:
    """Convert a 0..7 / NaN state grid into an (H, W, 3) RGB image array.

    Args:
        consensus_grid: The 0..7/NaN consensus state grid.
        theme: The visualization theme object.

    Returns:
        npt.NDArray[np.float64]: An (H, W, 3) RGB image array.
    """
    colors = _get_consensus_colors(theme)
    rgb = np.tile(
        np.array(to_rgb(theme.composite_masked_color)), (*consensus_grid.shape, 1)
    )

    for val in range(8):
        rgb[consensus_grid == val] = colors[val]
    return rgb


def _composite_legend_entries(
    theme: ChaosMapTheme,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Return (label, RGB color) pairs for the composite legend/key.

    Colors are computed via `_composite_flag_rgb` — the same mapping used to render the
    map — so the swatches always match the actual pixel colors.

    Args:
        theme: The visualization theme object.

    Returns:
        list[tuple[str, tuple[float, float, float]]]: List of (label, RGB color) pairs.
    """
    entries = [
        (COMPOSITE_LEGEND_LABELS[key], _composite_flag_rgb(*flags, theme))
        for key, flags in _COMPOSITE_LEGEND_FLAGS.items()
    ]
    entries.append(
        (COMPOSITE_LEGEND_LABELS["masked"], to_rgb(theme.composite_masked_color))
    )
    return entries
