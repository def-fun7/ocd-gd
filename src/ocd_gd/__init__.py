"""
Chaos Analysis Package.

A package for evaluating orbit integration data, checking convergence,
and detecting chaotic vs. regular dynamical behavior using SALI/GALI metrics.
"""

# 2. Import the detector class and its output data structures
from .orbit_detector import OrbitChaosDetector
from .grid_detector import GridChaosDetector

# 3. Explicitly define the public API exposed to users
__all__ = ["OrbitChaosDetector", "GridChaosDetector"]
