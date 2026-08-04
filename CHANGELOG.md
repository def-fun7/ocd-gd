# Changelog

All notable changes to the `ocd-gd` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** For changes specific to example scripts, test outputs, and documentation walkthroughs, see [examples/CHANGELOG.md](examples/CHANGELOG.md).

## [Unreleased]

### Added

- Script `scripts/make_composite_potentials.py` to generate a composite AGAMA potential (disk, bar, and central black hole) driven by two parameters (`Q_b` for bar strength and `frac_M_bh` for central black hole mass fraction), including JSON save/load functionality for sweep runs.
- Modular display module (`src/ocd_gd/_logging_config_.py`) supporting graceful fallbacks when `rich` is not installed, alongside a basic `logging` setup for main classes.
- Automated test runner (`examples/__main__.py`) for suite execution and reporting.
- Examples 01 through 05 covering single orbit integration, batch analysis, grid setup, and chaos plots.

### Changed

- Modernized type hints across the codebase to Python 3.10+ standards (replacing `Tuple`, `Union`, `List`, and `np.ndarray`), using `ruff` (`uvx ruff check --unsafe-fixes --fix`).
- Updated `examples/README.md` with documentation for the `examples/data` directory.

### Fixed

- Fixed broken links in `README.md`.
