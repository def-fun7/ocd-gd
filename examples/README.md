# `ocd-gd` Example Suite

Runnable, self-contained scripts demonstrating `ocd_gd`. Each one can be run directly from the repository root:

```bash
python examples/01_single_orbit_setup.py
```

or all examples can be run by one command,

```bash
python examples
```

> **Changelog Notice:** Any modification or addition to scripts in this folder **must** be logged in [CHANGELOG.md](CHANGELOG.md) alongside updates to this README file.

Every script prints a single `✓ Example completed successfully` line on success, or `✗ Example failed: <reason>` on failure (exit code 1) — e.g. a malformed potential or an initial-conditions array of the wrong shape. Set `OCD_GD_DEBUG=1` in the environment to get the full traceback instead of the clean one-line message.

Scripts are numbered `0N_name.py` in increasing order of complexity; each corresponding `docs/source/examples/0N_name.md` page pulls the script in verbatim via MyST's `literalinclude`, so the docs never drift from the code.

## Index

| #   | Script                                       | Covers                                                                                                                    |
| --- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 01  | `01_single_orbit_setup.py`                   | Build a potential, one orbit's ICs, construct `OrbitChaosDetector`, inspect criteria/basic state (no chaos detection yet) |
| 02  | `02_single_orbit_chaos_check.py` _(planned)_ | Same setup, then call `detect_chaos()` on the one orbit and interpret the summary                                         |
| 03  | `03_batch_orbits.py` _(planned)_             | Multiple initial conditions in one `OrbitChaosDetector`, `chaos_summary()` across the batch                               |
| 04  | `04_grid_basic.py` _(planned)_               | `GridChaosDetector` grid generation, `chaos_grids`, `orbit_idx_at`/`grid_position_of` lookups — no plotting               |
| 05  | `05_grid_chaos_map_plot.py` _(planned)_      | `GridChaosDetector.plot_chaos_map()` / `plot_composite_chaos_map()`                                                       |

## Requirements

Examples assume `ocd_gd` is available as `src.ocd_gd` from the repository root (see `src/__init__.py`) and that `agama` is installed in the active environment.
