"""
Chaos Analysis Package.

A package for evaluating orbit integration data, checking convergence,
and detecting chaotic vs. regular dynamical behavior using SALI/GALI metrics.
"""

from .orbit_detector import OrbitChaosDetector
from .grid_detector import GridChaosDetector
from ._types import (
    IntegrationCriteria,
    ChaosSummary,
    ChaosFullReport,
    ChaosSurveySummary,
    MethodChaosStats,
    ChaosAgreement,
    chaos_summary_row,
)
from ._resonance import ResonanceRadii

__all__ = [
    "OrbitChaosDetector",
    "GridChaosDetector",
    "IntegrationCriteria",
    "ChaosSummary",
    "ChaosFullReport",
    "ChaosSurveySummary",
    "MethodChaosStats",
    "ChaosAgreement",
    "chaos_summary_row",
    "ResonanceRadii",
]
