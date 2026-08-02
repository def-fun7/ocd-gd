# 05. Grid Chaos Map Visualization

This example demonstrates how to visualize spatial chaos across initial condition grids using `GridChaosDetector` plotting mixin methods (`plot_chaos_map`, `plot_composite_chaos_map`, and `save_chaos_maps`).

## Overview

Visualizing chaos across a 2D grid of initial conditions $(x, v_x)$ exposes resonance boundaries and chaotic diffusion regions:

- **Side-by-Side Maps (`plot_chaos_map`):** Renders SALI, GALI, and Lyapunov exponent maps across three panels on a shared scale.
- **RGB Composite Overlay (`plot_composite_chaos_map`):** Blends indicators into a single composite map for rapid multi-indicator comparison.
- **Batch Export (`save_chaos_maps`):** Generates and exports both figures to disk in a single call without opening interactive GUI windows.
- **Resonance Overlays:** Computes and draws vertical lines for Inner Lindblad, Corotation, and Outer Lindblad resonance locations when `show_resonances=True`.

## Code Example

```{literalinclude} ../../../examples/05_grid_chaos_map_plot.py
:language: python
:linenos:
```
