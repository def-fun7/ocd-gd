"""
Shared visual and layout style constants for matplotlib and plotly backends.
"""

__all__ = [
    "COMPOSITE_FIGSIZE_MPL",
    "COMPOSITE_LAYOUT_PLOTLY",
    "COMPOSITE_LEGEND_LABELS",
    "COMPOSITE_RESONANCE_LINESTYLE_MPL",
    "COMPOSITE_RESONANCE_LINEWIDTH_MPL",
    "CONSENSUS_FIGSIZE_MPL",
    "CONSENSUS_LABELS",
    "CONSENSUS_LAYOUT_PLOTLY",
    "FAMILY_BOUNDARY_LABEL",
    "FAMILY_BOUNDARY_LINESTYLE_MPL",
    "FAMILY_BOUNDARY_LINESTYLE_PLOTLY",
    "FAMILY_BOUNDARY_LINEWIDTH_MPL",
    "FAMILY_BOUNDARY_LINEWIDTH_PLOTLY",
    "MPL_LEGEND_KWARGS",
    "RESONANCE_LABELS",
    "RESONANCE_LINESTYLE_MPL",
    "RESONANCE_LINESTYLE_PLOTLY",
    "RESONANCE_LINEWIDTH",
    "SIDE_BY_SIDE_FIGSIZE_MPL",
    "SIDE_BY_SIDE_LAYOUT_PLOTLY",
    "SIDE_BY_SIDE_LEGEND_LABELS",
    "ZVC_LABEL",
    "ZVC_LINESTYLE_MPL",
    "ZVC_LINESTYLE_PLOTLY",
    "ZVC_LINEWIDTH",
    "_COMPOSITE_LEGEND_FLAGS",
    "_FAMILY_BOX_LABEL",
    "_FAMILY_LOOP_LABEL",
]


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

_COMPOSITE_LEGEND_FLAGS: dict[str, tuple[bool, bool, bool]] = {
    "regular": (False, False, False),
    "all": (True, True, True),
    "sali_only": (True, False, False),
    "gali_only": (False, True, False),
    "lyapunov_only": (False, False, True),
}
COMPOSITE_FIGSIZE_MPL = (8, 7)
COMPOSITE_LAYOUT_PLOTLY = {"height": 650, "width": 750}


COMPOSITE_RESONANCE_LINESTYLE_MPL = "--"
COMPOSITE_RESONANCE_LINEWIDTH_MPL = 2

ZVC_LABEL = "Zero-Velocity Curve"
ZVC_LINESTYLE_MPL = "--"
ZVC_LINESTYLE_PLOTLY = "dash"
ZVC_LINEWIDTH = 1.5

RESONANCE_LINESTYLE_MPL = ":"
RESONANCE_LINESTYLE_PLOTLY = "dot"
RESONANCE_LINEWIDTH = 1.3


RESONANCE_LABELS: dict[str, str] = {
    "corotation": "Corotation Radius",
    "inner_lindblad": "Inner Lindblad (ILR)",
    "outer_lindblad": "Outer Lindblad (OLR)",
}


FAMILY_BOUNDARY_LABEL = "Box / Loop Boundary"
FAMILY_BOUNDARY_LINESTYLE_MPL = "-"
FAMILY_BOUNDARY_LINEWIDTH_MPL = 1.6
FAMILY_BOUNDARY_LINESTYLE_PLOTLY = "solid"
FAMILY_BOUNDARY_LINEWIDTH_PLOTLY = 1.6
_FAMILY_BOX_LABEL = "box"
_FAMILY_LOOP_LABEL = "loop"


MPL_LEGEND_KWARGS = {
    "loc": "upper right",
    "fontsize": 9,
    "framealpha": 0.9,
    "facecolor": "#ffffff",
    "edgecolor": "#cccccc",
}


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
