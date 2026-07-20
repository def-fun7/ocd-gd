"""
Chaos Analysis Package.

A package for evaluating orbit integration data, checking convergence,
and detecting chaotic vs. regular dynamical behavior using SALI/GALI metrics.
"""

# 1. Import the core calculation function
from .evaluate_chaos import evaluate_chaos

# 2. Import the detector class and its output data structures
from .orbit_detector import (
    ChaosFullReport,
    ChaosSummary,
    OrbitChaosDetector,
)

# 3. Explicitly define the public API exposed to users
__all__ = [
    "evaluate_chaos",
    "OrbitChaosDetector",
    "ChaosSummary",
    "ChaosFullReport",
]
