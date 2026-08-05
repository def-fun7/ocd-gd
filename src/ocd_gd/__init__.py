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
    "chaos_summary_row",
    "setup_logging",
    "print_banner",
    "print_dataframe_table",
    "print_kv_table",
    "HAS_RICH",
    "console",
]

from ._terminal_config import (
    setup_logging,
    print_banner,
    print_dataframe_table,
    print_kv_table,
    HAS_RICH,
    console,
)
from ._resonance import ResonanceRadii
from ._types import (
    ChaosAgreement,
    ChaosFullReport,
    ChaosSummary,
    ChaosSurveySummary,
    IntegrationCriteria,
    MethodChaosStats,
    chaos_summary_row,
)
from .grid_detector import GridChaosDetector
from .orbit_detector import OrbitChaosDetector
