# Changelog

All notable changes to the `ocd-gd` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** For changes specific to example scripts, test outputs, and documentation walkthroughs, see [examples/CHANGELOG.md](examples/CHANGELOG.md).

## [Unreleased]

### Added

- Added `_decay_classifier.py` (`classify_decay`, `crossover_onset`, `is_chaotic`, `is_regular`, presets, label definitions) implementing the continuous decay-model classification engine.
- Added `DecayClassification` NamedTuple in `_types.py` holding `sali`, `gali`, `lyapunov_check`, and `lyapunov_time` alongside convenience properties `sali_check`, `gali_check`, `sali_time`, `gali_time`.
- Exported `DecayClassification`, `classify_decay`, and `LABELS` in top-level package namespace (`ocd_gd.__init__`).
- Added `n_periods` parameter to `OrbitChaosDetector.__init__`, mutually exclusive with `iter_time`, dynamically deriving integration duration via `pot.Tcirc(ic)`.
- Added `separate_sali: bool = False` option to `OrbitChaosDetector.detect_chaos()` to evaluate deviation vector pairs individually or as an envelope minimum.
- Added the functions by `ocd_gd._logging_config` to `__init__.py` and renamed it to `_terminal_config.py`.
- Added `docs/source/scripts` to contain documentation of scripts.
- Script `scripts/_cli_common.py` added for common argparse functionality for `scripts/build_potential.py` and `scripts/run_sweep.py`.
- Script `scripts/run_sweep.py`, `scripts/analyse_sweep.py` added to do parameter sweeps using potentials from `scripts/build_potential.py` and analyze them.
- Script `scripts/make_composite_potentials.py` to generate a composite AGAMA potential (disk, bar, and central black hole) driven by two parameters (`Q_b` for bar strength and `frac_M_bh` for central black hole mass fraction), including JSON save/load functionality for sweep runs.
- Modular display module (`src/ocd_gd/_logging_config_.py`) supporting graceful fallbacks when `rich` is not installed, alongside a basic `logging` setup for main classes.
- Automated test runner (`examples/__main__.py`) for suite execution and reporting.
- Examples 01 through 05 covering single orbit integration, batch analysis, grid setup, and chaos plots.

### Changed

- Replaced binary thresholding in `OrbitChaosDetector.detect_chaos()` with `classify_decay()` returning `DecayClassification`.
- Transposed `OrbitChaosDetector.sali_array` from `(num_orbits, 15, t)` to `(15, num_orbits, t)` to align multi-orbit traces consistently with GALI.
- Updated `OrbitChaosDetector.chaos_summary()` to derive classification statistics and indicator agreement from `is_chaotic()` and `is_regular()` decay labels.
- Restored `plotting_backend` parameter in `OrbitChaosDetector.__init__` with validation for `matplotlib` and `plotly`.
- Updated `GridChaosDetector` to integrate with `n_periods=None, iter_time=iter_time` and read binary indicators from `DecayClassification`.
- Updated matplotlib visual backend (`visualisation/mpl_backend.py`, `_plotting.py`, `visualisation/batch.py`) to support per-orbit differing timestamps (2D time arrays) and multi-line SALI/GALI overlays.
- Rewrote `tests/test_detect_chaos.py` to validate `detect_chaos()` with `DecayClassification`, caching, overrides, and slicing.
- Updated `tests/test_indexing.py` to assert the updated `(15, num_orbits, t)` SALI array shape.
- Renamed and moved `scripts/setup-env.sh` from scripts to root with new name `setup.sh` for better accessibility; updated `README.md` and `docs/source/quickstart.md`.
- Renamed script `scripts/make_composite_potentials.py` to `scripts/build_potential.py`, updated it to use `_cli_common.py` for argparse.
- Modernized type hints across the codebase to Python 3.10+ standards (replacing `Tuple`, `Union`, `List`, and `np.ndarray`), using `ruff` (`uvx ruff check --unsafe-fixes --fix`).
- Updated `examples/README.md` with documentation for the `examples/data` directory.

### Fixed

- Fixed dimension mismatch errors when plotting orbits with differing timestamps (`(num_orbits, t)` vs `(t,)`).
- Fixed broken links in `README.md`.
- Removed duplicate bullet entries in `CHANGELOG.md`.

### Removed

- Removed `/scripts` and `/runs` directories (`build_potential.py`, `run_sweep.py`, `analyse_sweep.py`, `summarise_sweep.py`, `_cli_common.py`, and `RUN_LOG.md`) as parameter sweeps, CLI scripts, and run execution setups will be maintained outside this core library repository from here on.
