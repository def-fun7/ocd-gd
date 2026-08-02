# Changelog

All notable changes to the `ocd-gd` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** For changes specific to example scripts, test outputs, and documentation walkthroughs, see [examples/CHANGELOG.md](examples/CHANGELOG.md).

## [0.1.0] - 2026-08-02 [Unreleased]

### Added

- Modular display module (`src/ocd_gd/_logging_config_.py`) supporting graceful fallbacks when `rich` is not installed with basic `logging` setup for main classes.
- Automated test runner (`examples/__main__.py`) for suite execution and reporting.
- Examples 01 through 05 covering single orbit integration, batch analysis, grid setup, and chaos plots.

---
