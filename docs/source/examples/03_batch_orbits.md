# 03. Batch Orbits & Chaos Survey

This example demonstrates how to integrate a batch of initial conditions simultaneously using `OrbitChaosDetector` and evaluate population-level statistics using `chaos_summary()`.

## Overview

When analyzing galaxy models or stellar halos, you often need to classify hundreds or thousands of orbits at once. `OrbitChaosDetector` handles 2D initial condition arrays seamlessly and provides vectorised metrics across the population.

`chaos_summary()` returns a `ChaosSurveySummary` instance containing:

- Per-indicator classification counts, fractions, and indices (`.sali`, `.gali`, `.lyapunov`).
- Pairwise and total agreement metrics across indicators (`.agreement`).

## Code Example

```{literalinclude} ../../../examples/03_batch_orbits.py
:language: python
:linenos:
```
