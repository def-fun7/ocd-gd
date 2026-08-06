__all__ = [
    "SIDE_BY_SIDE_LEGEND_LABELS",
    "SIDE_BY_SIDE_FIGSIZE_MPL",
    "SIDE_BY_SIDE_LAYOUT_PLOTLY",
    "COMPOSITE_LEGEND_LABELS",
    "_COMPOSITE_LEGEND_FLAGS",
    "COMPOSITE_FIGSIZE_MPL",
    "COMPOSITE_LAYOUT_PLOTLY",
    "COMPOSITE_RESONANCE_LINESTYLE_MPL",
    "COMPOSITE_RESONANCE_LINEWIDTH_MPL",
    "ZVC_LABEL",
    "ZVC_LINESTYLE_MPL",
    "ZVC_LINESTYLE_PLOTLY",
    "ZVC_LINEWIDTH",
    "RESONANCE_LINESTYLE_MPL",
    "RESONANCE_LINESTYLE_PLOTLY",
    "RESONANCE_LINEWIDTH",
    "RESONANCE_LABELS",
    "FAMILY_BOUNDARY_LABEL",
    "FAMILY_BOUNDARY_LINESTYLE_MPL",
    "FAMILY_BOUNDARY_LINEWIDTH_MPL",
    "FAMILY_BOUNDARY_LINESTYLE_PLOTLY",
    "FAMILY_BOUNDARY_LINEWIDTH_PLOTLY",
    "_FAMILY_BOX_LABEL",
    "_FAMILY_LOOP_LABEL",
    "MPL_LEGEND_KWARGS",
    "CONSENSUS_FIGSIZE_MPL",
    "CONSENSUS_LABELS",
    "CONSENSUS_LAYOUT_PLOTLY",
]

# =============================================================================
# STYLE CONSTANTS (theme-independent)
# =============================================================================

# Text, layout, and dash *patterns* — the same regardless of which theme is
# selected. Per-theme colors live in theme.py; look there to change a color.

SIDE_BY_SIDE_LEGEND_LABELS = {
    "regular": "Regular Orbit",
    "chaotic": "Chaotic Orbit",
    "masked": "Unphysical Domain",
}
SIDE_BY_SIDE_FIGSIZE_MPL = (16, 5.5)
SIDE_BY_SIDE_LAYOUT_PLOTLY = {"height": 550, "width": 1400}

COMPOSITE_LEGEND_LABELS = {
    "regular": "Regular Orbit (All 0)",
    "all": "Chaotic Orbit (All 1)",
    "sali_only": "SALI Chaotic",
    "gali_only": "GALI Chaotic",
    "lyapunov_only": "Lyapunov Chaotic",
    "masked": "Unphysical Domain",
}
# (sali_flag, gali_flag, lyapunov_flag) per legend entry, in display order.
_COMPOSITE_LEGEND_FLAGS: dict[str, tuple[bool, bool, bool]] = {
    "regular": (False, False, False),
    "all": (True, True, True),
    "sali_only": (True, False, False),
    "gali_only": (False, True, False),
    "lyapunov_only": (False, False, True),
}
COMPOSITE_FIGSIZE_MPL = (8, 7)
COMPOSITE_LAYOUT_PLOTLY = {"height": 650, "width": 750}
# Composite overlay draws resonance lines thicker/dashed rather than dotted,
# to stand out against the busier RGB background.
COMPOSITE_RESONANCE_LINESTYLE_MPL = "--"
COMPOSITE_RESONANCE_LINEWIDTH_MPL = 2

ZVC_LABEL = "Zero-Velocity Curve"
ZVC_LINESTYLE_MPL = "--"
ZVC_LINESTYLE_PLOTLY = "dash"
ZVC_LINEWIDTH = 1.5

RESONANCE_LINESTYLE_MPL = ":"
RESONANCE_LINESTYLE_PLOTLY = "dot"
RESONANCE_LINEWIDTH = 1.3
# ResonanceRadii field name -> legend label (physics text, not a color choice
# -> stays fixed across themes). Order controls legend order.
RESONANCE_LABELS: dict[str, str] = {
    "corotation": "Corotation Radius",
    "inner_lindblad": "Inner Lindblad (ILR)",
    "outer_lindblad": "Outer Lindblad (OLR)",
}

# Box/loop orbit-family boundary (see `orbit_family_grid` param below).
FAMILY_BOUNDARY_LABEL = "Box / Loop Boundary"
FAMILY_BOUNDARY_LINESTYLE_MPL = "-"
FAMILY_BOUNDARY_LINEWIDTH_MPL = 1.6
FAMILY_BOUNDARY_LINESTYLE_PLOTLY = "solid"
FAMILY_BOUNDARY_LINEWIDTH_PLOTLY = 1.6
_FAMILY_BOX_LABEL = "box"
_FAMILY_LOOP_LABEL = "loop"

# --- legend chrome (mpl) -----------------------------------------------------
MPL_LEGEND_KWARGS = {
    "loc": "upper right",
    "fontsize": 9,
    "framealpha": 0.9,
    "facecolor": "#ffffff",
    "edgecolor": "#cccccc",
}

# --- Style Constants for 8-State Classification Map ---
CONSENSUS_FIGSIZE_MPL = (9, 8)
CONSENSUS_LAYOUT_PLOTLY = {"height": 700, "width": 850}
CONSENSUS_LABELS: dict[int, str] = {
    0: "(0, 0, 0)",
    1: "(0, 0, 1)",
    2: "(0, 1, 0)",
    3: "(0, 1, 1)",
    4: "(1, 0, 0)",
    5: "(1, 0, 1)",
    6: "(1, 1, 0)",
    7: "(1, 1, 1)",
}
