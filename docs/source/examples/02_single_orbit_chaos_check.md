# 02. Single Orbit Chaos Check

This example demonstrates how to perform chaos detection on an integrated orbit using `OrbitChaosDetector.detect_chaos()`. It highlights the distinction between a quick convergence summary (`check_only=True`) and a detailed diagnostic report (`check_only=False`).

## Overview

Chaos in dynamical systems is evaluated by measuring how fast nearby trajectories diverge:

- **SALI (Small Alignment Index):** Tracks parallel and antiparallel deviation vectors. Drops exponentially to $0$ for chaotic orbits and fluctuates around non-zero values for regular orbits.
- **GALI (Generalized Alignment Index):** Generalizes SALI using $k$-deviation vectors to detect chaos across higher-dimensional subspaces.
- **Lyapunov Exponents:** Evaluates long-term exponential divergence rates of trajectories.

```{literalinclude} ../../../examples/02_single_orbit_chaos_check.py
:language: python
:linenos:
```
