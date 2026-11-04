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
- Added `get_preset()` to `_decay_classifier.py`; `OrbitChaosDetector.detect_chaos()` now resolves `sali_preset` / `gali_preset` through it.
- Added `n_periods` / `iter_time` (mutually exclusive, default 30 orbital periods) to `GridChaosDetector.__init__`, forwarded to `OrbitChaosDetector`.
- Added `map_values` (`"checks"`, `"score"` or `"rate_norm"`) to `GridChaosDetector.__init__` to choose what the SALI/GALI/Lyapunov chaos maps show, an optional per-call `map_values` override on `plot_chaos_map()`, and a `check_grids` property that always returns the binary check maps. `score` and `rate_norm` maps are NaN for undetermined orbits.
- Added a continuous colormap, shared colorbar and value type in the panel titles (e.g. `SALI Chaos Map (10x10, score)`) to the matplotlib side-by-side chaos map for the `score` and `rate_norm` modes.
- Added `tests/test_decay_classifier.py` (synthetic decay laws, label convention, floor censoring, presets, output contract, invariances, `rate_norm`, `crossover_onset`).
- Added `tests/test_decay_classification.py` for the `DecayClassification` properties (`*_check`, `*_time`, the `(15, N)` per-pair layout).
- Added `tests/test_integration.py` (real AGAMA: array shapes with per-orbit time axes, `n_periods * Tcirc` integration time, energy conservation, reproducibility, batch-versus-single consistency).
- Added `tests/test_grid_detector.py` (grid bookkeeping, `orbit_idx_at` / `grid_position_of` round trip, chaos grids per `map_values` mode, `n_periods` handling, map plot smoke tests).
- Added `tests/test_api_and_plots.py` (every name in `__all__` resolves; single-orbit and batch plot smoke tests on per-orbit time axes).
- Added a root `conftest.py` that forces the `Agg` matplotlib backend and closes figures after every test.
- Enabled the previously comment-only plotting-mixin examples in `_plotting.py` and `_grid_plotting.py` as executable doctests.

### Changed

- Replaced binary thresholding in `OrbitChaosDetector.detect_chaos()` with `classify_decay()` returning `DecayClassification`.
- Transposed `OrbitChaosDetector.sali_array` from `(num_orbits, 15, t)` to `(15, num_orbits, t)` to align multi-orbit traces consistently with GALI.
- Updated `OrbitChaosDetector.chaos_summary()` to derive classification statistics and indicator agreement from `is_chaotic()` and `is_regular()` decay labels.
- Restored `plotting_backend` parameter in `OrbitChaosDetector.__init__` with validation for `matplotlib` and `plotly`.
- Updated `GridChaosDetector` to read its chaos indicators from `DecayClassification`.
- Updated matplotlib visual backend (`visualisation/mpl_backend.py`, `_plotting.py`, `visualisation/batch.py`) to support per-orbit differing timestamps (2D time arrays) and multi-line SALI/GALI overlays.
- Rewrote `tests/test_detect_chaos.py` to validate `detect_chaos()` with `DecayClassification`, caching, overrides, and slicing.
- Updated `tests/test_indexing.py` to assert the updated `(15, num_orbits, t)` SALI array shape.
- Renamed and moved `scripts/setup-env.sh` from scripts to root with new name `setup.sh` for better accessibility; updated `README.md` and `docs/source/quickstart.md`.
- Renamed script `scripts/make_composite_potentials.py` to `scripts/build_potential.py`, updated it to use `_cli_common.py` for argparse.
- Modernized type hints across the codebase to Python 3.10+ standards (replacing `Tuple`, `Union`, `List`, and `np.ndarray`), using `ruff` (`uvx ruff check --unsafe-fixes --fix`).
- Updated `examples/README.md` with documentation for the `examples/data` directory.
- Replaced `iter_time: float = 10.0` in `GridChaosDetector` with `n_periods` / `iter_time`; the default integration length is now 30 orbital periods instead of 10 time units (examples that pass `iter_time` explicitly are unaffected).
- `GridChaosDetector.chaos_grids` now follows `map_values` (default `"checks"`, the previous behaviour); the composite and consensus maps always use the binary `check_grids`.
- `classify_decay` no longer sets `t_onset` for regular orbits (only for label 4, plateau -> exponential), so `sali_time` / `gali_time` are `inf` for regular orbits.
- `classify_decay` now reports a series that is already at the numerical floor at its first valid sample as undetermined (label 0) instead of floor-hit chaotic (label 6).
- Restructured the unit tests: parametrized index validation and getter checks, one parametrized lazy-property test that also asserts the `(15, N, T)`, `(N, T)` and `(N, 6, T, 6)` array layouts, `n_periods` / `iter_time` tests in `tests/test_init.py`, and a tightened SALI upper bound (`sqrt(2)`) in `tests/test_private_methods.py`.

### Fixed

- Fixed dimension mismatch errors when plotting orbits with differing timestamps (`(num_orbits, t)` vs `(t,)`).
- Fixed broken links in `README.md`.
- Removed duplicate bullet entries in `CHANGELOG.md`.
- Fixed unknown `sali_preset` / `gali_preset` names being silently ignored; they now raise `ValueError`.
- Fixed the matplotlib chaos maps discarding the result of `cmap.with_extremes(bad=...)`, so masked (NaN) cells now use the theme's masked colour.
- Fixed `sali_time` / `gali_time` returning a finite detection time for regular plateau -> power-law orbits.

### Removed

- Removed `/src/ocd_gd/_evaluate_chaos.py` which was previously used inside the `orbit_detector.py` inside detect_chaos, it now uses new `_decay_classifier.py` instead
- Removed `/scripts` and `/runs` directories (`build_potential.py`, `run_sweep.py`, `analyse_sweep.py`, `summarise_sweep.py`, `_cli_common.py`, and `RUN_LOG.md`) as parameter sweeps, CLI scripts, and run execution setups will be maintained outside this core library repository from here on.
- Removed `tests/test_chaos_sensitivity.py` and `tests/test_regular_sensitivity.py` (one asserted nothing, the other computed its rate over all orbits instead of the regular ones); sensitivity and detection-performance checks move to a separate validation suite.
- Removed the unused binary-era `ChaosSummary` and `ChaosFullReport` types and their top-level exports.

### Known issues

- SALI classification can depend on integration length and output density: in a spherical test potential a regular orbit was labelled chaotic with weak evidence (`delta_bic` ~ 1) for `n_periods <= 15`, and at 30 periods when `traj_size` was above about 750. `tests/test_integration.py` uses `n_periods=30, traj_size=600` until this is characterised by the validation suite.
- In `map_values="checks"` mode, undetermined orbits (label 0) appear as regular (0) on the chaos maps; the `score` and `rate_norm` modes show them as masked (NaN).
- The Lyapunov panel has no score: in `score` mode it shows the 0/1 Lyapunov flag, and its flag threshold (`> 0.1`) is hard-coded.
