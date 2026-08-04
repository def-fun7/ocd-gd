# `ocd-gd` Example Suite

Runnable, self-contained scripts demonstrating `ocd_gd`. Each one can be run directly from the repository root:

```bash
python examples/01_single_orbit_setup.py
```

or all examples can be run by one command,

```bash
python -m examples
```

> **Changelog Notice:** Any modification or addition to scripts in this folder **must** be logged in [CHANGELOG.md](CHANGELOG.md) alongside updates to this README file.

Every script prints a single `✓ Example completed successfully` line on success, or `✗ Example failed: <reason>` on failure (exit code 1) — e.g. a malformed potential or an initial-conditions array of the wrong shape. Set `OCD_GD_DEBUG=1` in the environment to get the full traceback instead of the clean one-line message.

Scripts are numbered `0N_name.py` in increasing order of complexity; each corresponding `docs/source/examples/0N_name.md` page pulls the script in verbatim via MyST's `literalinclude`, so the docs never drift from the code.

## Index

| #   | Date         | Script                           | Covers                                                                                                                    |
| :-- | :----------- | :------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| 01  | 03 Aug, 2026 | `01_single_orbit_setup.py`       | Build a potential, one orbit's ICs, construct `OrbitChaosDetector`, inspect criteria/basic state (no chaos detection yet) |
| 02  | 03 Aug, 2026 | `02_single_orbit_chaos_check.py` | Same setup, then call `detect_chaos()` on the one orbit and interpret the summary                                         |
| 03  | 03 Aug, 2026 | `03_batch_orbits.py`             | Multiple initial conditions in one `OrbitChaosDetector`, `chaos_summary()` across the batch                               |
| 04  | 03 Aug, 2026 | `04_grid_basic.py`               | `GridChaosDetector` grid generation, `chaos_grids`, `orbit_idx_at`/`grid_position_of` lookups — no plotting               |
| 05  | 03 Aug, 2026 | `05_grid_chaos_map_plot.py`      | `GridChaosDetector.plot_chaos_map()` / `plot_composite_chaos_map()`                                                       |

## Requirements

Examples assume `ocd_gd` is available as `src.ocd_gd` from the repository root (see `src/__init__.py`) and that `agama` is installed in the active environment.

## Data

All the required data files, `.ini` files for custom potentials and `.npz` files for initial conditions arrays, are added in separate folders in the data. Following table list the sources of these files.

### Potentials

| #   | Date         | File                           | Source                                                                                                                                                       | About                                    |
| :-- | :----------- | :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 01  | 20 July 2026 | `MWPotentialHunter24_full.ini` | generated from [agama/py/example_mw_potential_hunter24.py](https://github.com/GalacticDynamics-Oxford/Agama/blob/master/py/example_mw_potential_hunter24.py) | full potential with a bar but no spirals |
