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
    "_composite_flag_rgb",
    "_binary_grids_to_composite_rgb",
    "_compute_zvc",
    "_resonance_overlay_specs",
    "_family_boundary_field",
    "_has_family_boundary",
    "_compute_consensus_grid",
    "_get_consensus_colors",
    "_consensus_grid_to_rgb",
]


import numpy as np
import numpy.typing as npt

from matplotlib.colors import to_rgb

from .grid_themes import ChaosMapTheme, DEFAULT_THEME, get_theme
from .grid_constants import RESONANCE_LABELS, _FAMILY_BOX_LABEL, _FAMILY_LOOP_LABEL

# =============================================================================
# SHARED HELPERS — data to RGB
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


def _composite_flag_rgb(
    sali: bool, gali: bool, lyapunov: bool, theme: ChaosMapTheme
) -> tuple[float, float, float]:
    """Compute exact RGB tuple for a combination of (SALI, GALI, Lyapunov) flags.

    Channel mapping:
    - Red   <- Lyapunov
    - Green <- GALI
    - Blue  <- SALI
    """
    r_base, g_base, b_base = theme.composite_base_rgb

    r = theme.composite_on_r if lyapunov else r_base
    g = theme.composite_on_g if gali else g_base
    b = theme.composite_on_b if sali else b_base

    return (r, g, b)


def _binary_grids_to_composite_rgb(
    sali_grid: npt.NDArray[np.float64],
    gali_grid: npt.NDArray[np.float64],
    lyapunov_grid: npt.NDArray[np.float64],
    theme: ChaosMapTheme,
) -> npt.NDArray[np.float64]:
    """Construct the themed RGB composite chaos map (vectorized version of
    `_composite_flag_rgb`)."""
    shape = sali_grid.shape
    nan_mask = np.isnan(sali_grid) | np.isnan(gali_grid) | np.isnan(lyapunov_grid)

    s = np.nan_to_num(sali_grid, nan=0).astype(bool)
    g = np.nan_to_num(gali_grid, nan=0).astype(bool)
    l = np.nan_to_num(lyapunov_grid, nan=0).astype(bool)
    any_flag = s | g | l

    base = np.array(theme.composite_base_rgb)
    rgb = np.tile(base, (*shape, 1))

    r_channel = np.where(l, theme.composite_on_r, base[0])
    g_channel = np.where(g, theme.composite_on_g, base[1])
    b_channel = np.where(s, theme.composite_on_b, base[2])

    rgb[any_flag] = np.stack([r_channel, g_channel, b_channel], axis=-1)[any_flag]
    rgb[nan_mask] = to_rgb(theme.composite_masked_color)
    return rgb


def _compute_zvc(
    E_rem_vals: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64] | None:
    """Zero-velocity curve from residual energy, or None if not supplied."""
    if E_rem_vals is None:
        return None
    return np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0))


def _resonance_overlay_specs(
    resonance_radii, theme: ChaosMapTheme
) -> list[tuple[float, str, str]]:
    """Return (radius, label, color) for each resonance radius that was
    actually found (skips any that are None — e.g. a non-rotating potential
    has no corotation/Lindblad radii at all)."""
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
    """Numeric 0 (box) / 1 (loop) / NaN field used to trace the box/loop
    boundary as a contour line. None if no family grid was supplied.

    Any entry that is neither `"box"` nor `"loop"` (e.g. an unphysical cell
    with no family classification) becomes NaN and is simply left as a gap
    in the boundary line, same as unphysical cells elsewhere on these maps.
    """
    if orbit_family_grid is None:
        return None
    field = np.full(orbit_family_grid.shape, np.nan)
    field[orbit_family_grid == _FAMILY_LOOP_LABEL] = 1.0
    field[orbit_family_grid == _FAMILY_BOX_LABEL] = 0.0
    return field


def _has_family_boundary(family_field: npt.NDArray[np.float64] | None) -> bool:
    """Whether a box/loop boundary actually exists to draw (both families
    must be present somewhere in the grid)."""
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
    """
    nan_mask = np.isnan(sali_grid) | np.isnan(gali_grid) | np.isnan(lyapunov_grid)

    s = np.nan_to_num(sali_grid, nan=0.0).astype(int)
    g = np.nan_to_num(gali_grid, nan=0.0).astype(int)
    l = np.nan_to_num(lyapunov_grid, nan=0.0).astype(int)

    # 3-bit integer mapping: 4*L + 2*S + 1*G
    code = (l << 2) | (s << 1) | g

    consensus = code.astype(np.float64)
    consensus[nan_mask] = np.nan
    return consensus


def _get_consensus_colors(theme: ChaosMapTheme) -> list[tuple[float, float, float]]:
    """Return RGB colors for all 8 discrete states (0..7) derived from `theme`.

    Calls through `_composite_flag_rgb` so the exact RGB channel definition
    is evaluated for each combination.
    """
    colors = []
    for code in range(8):
        l = bool(code & 4)  # Bit 2: Lyapunov
        s = bool(code & 2)  # Bit 1: SALI
        g = bool(code & 1)  # Bit 0: GALI

        # Note the exact argument order: sali, gali, lyapunov
        colors.append(_composite_flag_rgb(sali=s, gali=g, lyapunov=l, theme=theme))
    return colors


def _consensus_grid_to_rgb(
    consensus_grid: npt.NDArray[np.float64],
    theme: ChaosMapTheme,
) -> npt.NDArray[np.float64]:
    """Convert a 0..7 / NaN state grid into an (H, W, 3) RGB image array."""
    colors = _get_consensus_colors(theme)
    rgb = np.tile(
        np.array(to_rgb(theme.composite_masked_color)), (*consensus_grid.shape, 1)
    )

    for val in range(8):
        rgb[consensus_grid == val] = colors[val]
    return rgb
