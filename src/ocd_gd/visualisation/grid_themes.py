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

from dataclasses import dataclass

__all__ = ["DEFAULT_THEME", "ChaosMapTheme", "get_theme"]

DEFAULT_THEME = "magma"


_STATE_ORDER: list[tuple[bool, bool, bool]] = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]
_STATE_RANK: dict[tuple[bool, bool, bool], int] = {
    s: i for i, s in enumerate(_STATE_ORDER)
}


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

def _lerp_rgb(
    lo: tuple[float, float, float], hi: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    return tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))

@dataclass(frozen=True)
class ChaosMapTheme:
    name: str
    display_name: str

    color_regular: str
    color_chaotic: str
    color_masked: str

    composite_regular_color: str
    composite_chaotic_color: str
    composite_masked_color: tuple[float, float, float]

    zvc_color: str
    resonance_colors: dict[str, str]
    family_boundary_color: str

    def get_vote_color(
        self, n_chaotic: int, n_indicators: int = 3
    ) -> tuple[float, float, float]:
        """Continuous light->dark ramp by vote count alone (0..n_indicators).
        Used by the composite RGB overlay, where only 'how many agreed'
        matters, not which specific indicators."""
        t = n_chaotic / n_indicators
        lo = _hex_to_rgb01(self.composite_regular_color)
        hi = _hex_to_rgb01(self.composite_chaotic_color)
        return _lerp_rgb(lo, hi, t)

    def get_state_color(
        self, sali: bool, gali: bool, lyapunov: bool
    ) -> tuple[float, float, float]:
        """One of 8 distinct shades along the same light->dark ramp, one
        per exact (sali, gali, lyapunov) combination — monotonic in vote
        count, but same-vote states no longer collapse onto one color."""
        rank = _STATE_RANK[(sali, gali, lyapunov)]
        t = rank / (len(_STATE_ORDER) - 1)
        lo = _hex_to_rgb01(self.composite_regular_color)
        hi = _hex_to_rgb01(self.composite_chaotic_color)
        return _lerp_rgb(lo, hi, t)

    @property
    def composite_base_rgb(self) -> tuple[float, float, float]:
        return _hex_to_rgb01(self.composite_regular_color)

    @property
    def composite_on_r(self) -> float:
        return _hex_to_rgb01(self.composite_chaotic_color)[0]

    @property
    def composite_on_g(self) -> float:
        return _hex_to_rgb01(self.composite_chaotic_color)[1]

    @property
    def composite_on_b(self) -> float:
        return _hex_to_rgb01(self.composite_chaotic_color)[2]

THEMES: dict[str, ChaosMapTheme] = {}


def _register(theme: ChaosMapTheme) -> None:
    THEMES[theme.name] = theme

_register(
    ChaosMapTheme(
        name="magma",
        display_name="Magma",
        color_regular="#f6d98a",
        color_chaotic="#3b0f36",
        color_masked="#e6e6e6",
        composite_regular_color="#f6d98a",
        composite_chaotic_color="#3b0f36",
        composite_masked_color=(0.90, 0.90, 0.90),
        zvc_color="red",
        resonance_colors={
            "corotation": "#2ca02c",
            "inner_lindblad": "#9467bd",
            "outer_lindblad": "#8c564b",
        },
        family_boundary_color="#39ff14",
    )
)


_register(
    ChaosMapTheme(
        name="viridis",
        display_name="Viridis",
        color_regular="#eef79a",
        color_chaotic="#440154",
        color_masked="#e6e6e6",
        composite_regular_color="#eef79a",
        composite_chaotic_color="#440154",
        composite_masked_color=(0.90, 0.90, 0.90),
        zvc_color="#ff4d4d",
        resonance_colors={
            "corotation": "#ff7f0e",
            "inner_lindblad": "#e377c2",
            "outer_lindblad": "#8c564b",
        },
        family_boundary_color="#2c2c2c",
    )
)


_register(
    ChaosMapTheme(
        name="ocean",
        display_name="Ocean",
        color_regular="#a9e8e3",
        color_chaotic="#0b1f3a",
        color_masked="#e6e6e6",
        composite_regular_color="#a9e8e3",
        composite_chaotic_color="#0b1f3a",
        composite_masked_color=(0.90, 0.90, 0.90),
        zvc_color="#ff6b6b",
        resonance_colors={
            "corotation": "#f4a261",
            "inner_lindblad": "#e76f51",
            "outer_lindblad": "#e9c46a",
        },
        family_boundary_color="#118ab2",
    )
)


_register(
    ChaosMapTheme(
        name="sunset",
        display_name="Sunset",
        color_regular="#fbcfa8",
        color_chaotic="#3a0d1f",
        color_masked="#e6e6e6",
        composite_regular_color="#fbcfa8",
        composite_chaotic_color="#3a0d1f",
        composite_masked_color=(0.90, 0.90, 0.90),
        zvc_color="#118ab2",
        resonance_colors={
            "corotation": "#06d6a0",
            "inner_lindblad": "#ffd166",
            "outer_lindblad": "#ef476f",
        },
        family_boundary_color="#2b2d42",
    )
)


def list_themes() -> list[str]:
    return list(THEMES.keys())

def get_theme(theme: str | ChaosMapTheme) -> ChaosMapTheme:
    if isinstance(theme, ChaosMapTheme):
        return theme
    try:
        return THEMES[str(theme).lower()]
    except KeyError:
        available = ", ".join(sorted(THEMES))
        raise ValueError(
            f"Unknown chaos-map theme {theme!r}. Available themes: {available}"
        ) from None
