# Examples CHANGELOG

Chronological log of additions, updates, API shifts, and structural reorganizations within the `examples/` directory.

---

## [Unreleased]

### 2026-08-03 Updates

- **Fixed** remove the `_to_scalar` call in `./02_single_orbit_chaos_check.py` from `lyapunov`, after typings update to `py310+`.
- **Updated `README.md` for examples:** added the date column for examples, added the section for data folder contents

### 2026-08-02 — Writing Basic example suite (with rich-formated-cli) along with README and CHANGELOG

- **Updated `05_grid_chaos_map_plot.py`:** Refactored to use `_GridChaosPlottingMixin.save_chaos_maps()` and `plot_composite_chaos_map()` matching the updated mixin signature (`cmap_colors`, `masked_color`, `show_resonances`).
- **Updated `02_single_orbit_chaos_check.py`:** Added `_to_scalar()` helper function to safely format 0D/1D NumPy array outputs returned by `detect_chaos()`.
- **Added `__main__.py`:** Suite discovery and execution runner.
- **Added Scripts 01–05:**
  - `01_single_orbit_setup.py`: Agama potential initialization & basic integration.
  - `02_single_orbit_chaos_check.py`: `check_only` vs full report diagnostics.
  - `03_batch_orbits.py`: Multi-orbit integration and `chaos_summary()` survey.
  - `04_grid_basic.py`: Energy-conserving surface initialization and index mapping.
  - `05_grid_chaos_map_plot.py`: 2D spatial chaos visualization and file export.
- **Target Package Version:** `ocd-gd == 0.1.0`
