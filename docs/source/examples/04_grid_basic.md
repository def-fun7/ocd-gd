# 04. Basic Grid Chaos Detector

This example demonstrates how `GridChaosDetector` sets up a 2D $(x, v_x)$ initial condition grid at fixed total energy, filters out energy-forbidden unphysical cells, lazy-loads 2D chaos maps, and provides spatial lookups between grid positions and orbit indices.

## Overview

Instead of specifying manual 6D phase space vectors, `GridChaosDetector` scans a phase-space surface at fixed total energy $E_0$:

- Unphysical cells (where kinetic energy would be negative) are automatically masked as `NaN` and skipped during orbit integration.
- Physical orbits are integrated and mapped back to 2D numpy arrays accessible via `.chaos_grids`.
- Spatial methods (`orbit_idx_at`, `grid_position_of`, `grid_coordinates_of`) allow rapid mapping between 2D map locations $(row, col)$ and integrated orbit indices.

## Code Example

```{literalinclude} ../../../examples/04_grid_basic.py
:language: python
:linenos:
```
