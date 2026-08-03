# CHANGELOG

All notable changes to the `ocd-gd` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** For changes specific to example scripts, test outputs, and documentation walkthroughs, see [examples/CHANGELOG.md](examples/CHANGELOG.md).

## [0.1.0] - 2026-08-03 [Unreleased]

- **Modernize** Updated from old generic types `Tuple, Union, List` and `nd.narray` to `py310+` newer ones, using ruff through `uvx ruff check --unsafe-fixes --fix`, rerun examples and test to confirm it still works (which it does)
- **Updated** Some Broken Links in the `README.md`.
- **Added** Script `scripts/make_composite_potentials.py`, that makes a composite agama potential of DISK, BAR and CENTRAL BLACK HOLE, with only two varying parameters, `Q_b` (Bar strength) and `frac_M_bh` (Mass of Central Black Hole, in terms of fraction of total mass) and saves them in JSON and also provide a function to load them again, this can then later on be used to sweep runs.
- **Updated** `examples/README.md` to include info of `examples/data` folder.

## [0.1.0] - 2026-08-02 [Unreleased]

- **Added** Modular display module (`src/ocd_gd/_logging_config_.py`) supporting graceful fallbacks when `rich` is not installed with basic `logging` setup for main classes.
- **Added** Automated test runner (`examples/__main__.py`) for suite execution and reporting.
- **Added** Examples 01 through 05 covering single orbit integration, batch analysis, grid setup, and chaos plots.

---
