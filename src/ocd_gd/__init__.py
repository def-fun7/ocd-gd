"""
Chaos Analysis Package.

A package for evaluating orbit integration data, checking convergence,
and detecting chaotic vs. regular dynamical behavior using SALI/GALI metrics.
"""

__all__ = [
    "ChaosAgreement",
    "ChaosFullReport",
    "ChaosSummary",
    "ChaosSurveySummary",
    "OrbitChaosDetector",
    "GridChaosDetector",
    "IntegrationCriteria",
    "MethodChaosStats",
    "ResonanceRadii",
    "CorotationSetup",
    "omega_for_corotation_ratio",
    "chaos_summary_row",
    "AgamaUnits",
    "tag_unit",
    "FamilyStats",
    "get_logger",
    "setup_logging",
    "print_banner",
    "print_dataframe_table",
    "print_kv_table",
    "HAS_RICH",
    "console",
    "set_output_dir",
    "set_publication_style",
]

from ._terminal_config import (
    get_logger,
    setup_logging,
    print_banner,
    print_dataframe_table,
    print_kv_table,
    HAS_RICH,
    console,
)
from ._resonance import ResonanceRadii, omega_for_corotation_ratio, CorotationSetup
from ._types import (
    ChaosAgreement,
    ChaosFullReport,
    ChaosSummary,
    ChaosSurveySummary,
    IntegrationCriteria,
    MethodChaosStats,
    chaos_summary_row,
)

from .visualisation import set_output_dir, set_publication_style

from ._units import AgamaUnits, tag_unit
from ._family_check import FamilyStats

from .grid_detector import GridChaosDetector
from .orbit_detector import OrbitChaosDetector
