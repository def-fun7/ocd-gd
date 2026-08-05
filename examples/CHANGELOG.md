# Examples Changelog

Chronological log of additions, updates, API shifts, and structural reorganizations within the `examples/` directory.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Automated suite runner (`__main__.py`) for example execution and reporting.
- Example scripts 01 through 05:
  - `01_single_orbit_setup.py`: AGAMA potential initialization and basic integration.
  - `02_single_orbit_chaos_check.py`: `check_only` vs. full report diagnostics.
  - `03_batch_orbits.py`: Multi-orbit integration and `chaos_summary()` survey.
  - `04_grid_basic.py`: Energy-conserving surface initialization and index mapping.
  - `05_grid_chaos_map_plot.py`: 2D spatial chaos visualization and file export.

### Changed

- Updated examples 01-05 with updated import from ocd_gd for print functions, instead of from `ocd_gd._logging_config`
- Updated `05_grid_chaos_map_plot.py` to use `_GridChaosPlottingMixin.save_chaos_maps()` and `plot_composite_chaos_map()` matching updated mixin signature (`cmap_colors`, `masked_color`, `show_resonances`).
- Updated `examples/README.md` to include a date column for examples and documentation for the `data/` folder contents.

### Fixed

- Removed obsolete `_to_scalar` call in `02_single_orbit_chaos_check.py` following the Python 3.10+ typing modernization.
