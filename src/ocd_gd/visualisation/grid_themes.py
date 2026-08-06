"""
Visual themes for chaos-map plotting (see `chaos_map_plots.py`).

A theme bundles every *color* and *color-adjacent* constant the plotting
functions need — regular/chaotic colors, the composite RGB channel mapping,
resonance-line colors, etc. Domain text (legend labels, titles) and layout
(figure size, line dash *patterns*) stay in `chaos_map_plots.py` since those
aren't really "look" choices — they're the same regardless of theme.

Add a new theme by adding a `ChaosMapTheme` entry to `THEMES` below; nothing
in the plotting module needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_THEME = "magma"


@dataclass(frozen=True)
class ChaosMapTheme:
    name: str
    display_name: str

    # --- side-by-side (binary regular/chaotic) maps -------------------------
    color_regular: str
    color_chaotic: str
    color_masked: str

    # --- composite / 8-state RGB overlay ------------------------------------
    # Channel mapping: Red <- Lyapunov, Green <- GALI, Blue <- SALI.
    # Base RGB defines state (0, 0, 0). Turning a flag ON sets that channel
    # to its corresponding `composite_on_*` value.
    composite_base_rgb: tuple[float, float, float]
    composite_on_r: float
    composite_on_g: float
    composite_on_b: float
    composite_masked_color: tuple[float, float, float]

    # --- overlays -----------------------------------------------------------
    zvc_color: str
    resonance_colors: dict[str, str]  # ResonanceRadii field name -> color
    family_boundary_color: str

    def get_state_color(
        self, sali: bool, gali: bool, lyapunov: bool
    ) -> tuple[float, float, float]:
        """Compute the exact RGB color for any of the 8 (Lyapunov, SALI, GALI) states."""
        r_base, g_base, b_base = self.composite_base_rgb
        r = self.composite_on_r if lyapunov else r_base
        g = self.composite_on_g if gali else g_base
        b = self.composite_on_b if sali else b_base
        return (r, g, b)


THEMES: dict[str, ChaosMapTheme] = {}


def _register(theme: ChaosMapTheme) -> None:
    THEMES[theme.name] = theme


# =============================================================================
# MAGMA (default) — dark violet -> gold / vibrant primary additive channels
# =============================================================================
_register(
    ChaosMapTheme(
        name="magma",
        display_name="Magma",
        color_regular="#1f4e78",
        color_chaotic="#f2c811",
        color_masked="#333333",
        composite_base_rgb=(0.08, 0.05, 0.15),  # Dark Night Violet
        composite_on_r=0.92,  # Lyapunov (Red)
        composite_on_g=0.75,  # GALI (Green)
        composite_on_b=0.85,  # SALI (Blue - High intensity)
        composite_masked_color=(0.2, 0.2, 0.2),
        zvc_color="red",
        resonance_colors={
            "corotation": "#2ca02c",
            "inner_lindblad": "#9467bd",
            "outer_lindblad": "#8c564b",
        },
        family_boundary_color="#39ff14",
    )
)

# =============================================================================
# VIRIDIS — deep purple background, high contrast channels
# =============================================================================
_register(
    ChaosMapTheme(
        name="viridis",
        display_name="Viridis",
        color_regular="#440154",
        color_chaotic="#fde725",
        color_masked="#3f3f3f",
        composite_base_rgb=(0.12, 0.02, 0.18),
        composite_on_r=0.95,
        composite_on_g=0.85,
        composite_on_b=0.90,
        composite_masked_color=(0.25, 0.25, 0.25),
        zvc_color="#ff4d4d",
        resonance_colors={
            "corotation": "#ff7f0e",
            "inner_lindblad": "#e377c2",
            "outer_lindblad": "#8c564b",
        },
        family_boundary_color="#ffffff",
    )
)

# =============================================================================
# OCEAN — cool dark blue background, bright RGB channel highlights
# =============================================================================
_register(
    ChaosMapTheme(
        name="ocean",
        display_name="Ocean",
        color_regular="#0b3d59",
        color_chaotic="#2ec4b6",
        color_masked="#607d8b",
        composite_base_rgb=(0.04, 0.10, 0.20),
        composite_on_r=0.90,
        composite_on_g=0.85,
        composite_on_b=0.95,
        composite_masked_color=(0.30, 0.34, 0.36),
        zvc_color="#ff6b6b",
        resonance_colors={
            "corotation": "#f4a261",
            "inner_lindblad": "#e76f51",
            "outer_lindblad": "#e9c46a",
        },
        family_boundary_color="#ffe66d",
    )
)

# =============================================================================
# SUNSET — dark charcoal background, warm vibrant highlights
# =============================================================================
_register(
    ChaosMapTheme(
        name="sunset",
        display_name="Sunset",
        color_regular="#2b2d42",
        color_chaotic="#ef8354",
        color_masked="#8d99ae",
        composite_base_rgb=(0.10, 0.08, 0.15),
        composite_on_r=0.95,
        composite_on_g=0.75,
        composite_on_b=0.85,
        composite_masked_color=(0.35, 0.37, 0.40),
        zvc_color="#118ab2",
        resonance_colors={
            "corotation": "#06d6a0",
            "inner_lindblad": "#ffd166",
            "outer_lindblad": "#ef476f",
        },
        family_boundary_color="#ffffff",
    )
)


def list_themes() -> list[str]:
    """Names of all registered themes, e.g. for building a CLI choice list."""
    return list(THEMES.keys())


def get_theme(theme: str | ChaosMapTheme) -> ChaosMapTheme:
    """Resolve a theme name (case-insensitive) to a `ChaosMapTheme`.

    Passing an already-resolved `ChaosMapTheme` is also accepted, so a caller
    that has one on hand doesn't need to round-trip through its name.
    """
    if isinstance(theme, ChaosMapTheme):
        return theme
    try:
        return THEMES[str(theme).lower()]
    except KeyError:
        available = ", ".join(sorted(THEMES))
        raise ValueError(
            f"Unknown chaos-map theme {theme!r}. Available themes: {available}"
        ) from None
