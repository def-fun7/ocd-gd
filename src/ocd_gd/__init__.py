"""
Chaos Analysis Package.

A package for evaluating orbit integration data, checking convergence,
and detecting chaotic vs. regular dynamical behavior using SALI/GALI metrics.
"""

from ._logging_config import setup_logging
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

__all__ = [
    "ChaosAgreement",
    "ChaosFullReport",
    "ChaosSummary",
    "ChaosSurveySummary",
    "GridChaosDetector",
    "IntegrationCriteria",
    "MethodChaosStats",
    "OrbitChaosDetector",
    "ResonanceRadii",
    "chaos_summary_row",
    "setup_logging",
]
