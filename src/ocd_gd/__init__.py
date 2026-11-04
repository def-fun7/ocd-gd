"""
Chaos Analysis Package.

A package for evaluating orbit integration data, checking convergence,
and detecting chaotic vs. regular dynamical behavior using SALI/GALI metrics.
"""

__all__ = [
    "HAS_RICH",
    "AgamaUnits",
    "ChaosAgreement",
    "ChaosSurveySummary",
    "CorotationSetup",
    "DecayClassification",
    "FamilyStats",
    "GridChaosDetector",
    "IntegrationCriteria",
    "LABELS",
    "MethodChaosStats",
    "OrbitChaosDetector",
    "ResonanceRadii",
    "chaos_summary_row",
    "classify_decay",
    "console",
    "get_logger",
    "omega_for_corotation_ratio",
    "print_banner",
    "print_dataframe_table",
    "print_kv_table",
    "set_output_dir",
    "set_publication_style",
    "setup_logging",
    "tag_unit",
]

from ._decay_classifier import LABELS, classify_decay
from ._family_check import FamilyStats
from ._resonance import CorotationSetup, ResonanceRadii, omega_for_corotation_ratio
from ._terminal_config import (
    HAS_RICH,
    console,
    get_logger,
    print_banner,
    print_dataframe_table,
    print_kv_table,
    setup_logging,
)
from ._types import (
    ChaosAgreement,
    ChaosSurveySummary,
    DecayClassification,
    IntegrationCriteria,
    MethodChaosStats,
    chaos_summary_row,
)
from ._units import AgamaUnits, tag_unit
from .grid_detector import GridChaosDetector
from .orbit_detector import OrbitChaosDetector
from .visualisation import set_output_dir, set_publication_style
