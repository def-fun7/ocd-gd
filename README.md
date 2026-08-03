# Orbital Chaos Detector - Galactic Dynamics (ocd-gd)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE) [![Python Version](https://img.shields.io/badge/03.13+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![AGAMA Core](https://img.shields.io/badge/C%2B%2B%20Core-AGAMA-orange.svg)](https://github.com/gaspicker/agama) [![Build Tool](https://img.shields.io/badge/package%20manager-uv-de5b44.svg?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Nix Support](https://img.shields.io/badge/Nix-Flake-5277C3.svg?logo=nixos&logoColor=white)](flake.nix) [![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#setup--installation)

`ocd-gd` is a toolkit for analyzing chaotic dynamics in galactic potential models. It builds on top of the **AGAMA** library for galactic dynamics, providing automated tools for orbit integration, chaos detection, and phase-space classification.

---

## Prerequisites

This project uses AGAMA, which requires a standard **C/C++ toolchain** (`gcc` or `clang`, `make`) along with core scientific C-libraries:

- **GSL** (GNU Scientific Library)
- **OpenBLAS** or **LAPACK**
- **Eigen3**
- **GMP**

> **Note for Nix users:** You do **not** need to manually install any compilers or libraries. Nix automatically builds and isolates all dependencies inside a sandbox environment.

---

## Setup & Installation

### Option A: Automated Setup (Recommended)

The repository includes a cross-platform shell script that detects your operating system, automatically installs missing native C libraries via your local package manager (`brew`, `apt`, `dnf`, `pacman`, `winget`), sets required compiler search paths (including nested Eigen headers), and builds the environment non-interactively.

```bash
# Make the setup script executable (first time only)
chmod +x scripts/setup-env.sh

# Run the automated setup
./scripts/setup-env.sh
```

### Option B: Manual Setup by Package Manager

If you prefer to manage your environment manually, ensure your system already has the prerequisite C libraries installed before invoking your package manager.

#### 1. Nix (`nix develop`)

Nix provides a completely isolated, reproducible developer environment. It supplies pre-configured compilers and C headers directly from the Nix store without altering your host system.

```bash
# Spawns a dev shell pre-loaded with GSL, Eigen, OpenBLAS, GCC, and uv
nix develop
```

### 2. `uv`

`uv` handles virtual environments and dependency resolution fast. AGAMA’s `setup.py` presents an interactive build prompt by default; passing `--config-settings="--build-option=--yes"` forces a non-interactive build.

```bash
uv sync --config-settings="--build-option=--yes" --all-groups
```

### 3. `pip`

To build and install the package in editable mode using standard `pip`:

```bash
pip install --config-settings="--build-option=--yes" -e .
```

### 4. `poetry`

If managing dependencies via Poetry, export the build setting prior to installation:

```bash
poetry install --config-settings="--build-option=--yes"
```

### How `uv` and `Nix` Work for This Repository

Because `ocd-gd` relies on AGAMA's underlying C++ extensions, both `uv` and `Nix` are configured to handle complex build requirements reproducible:

- **`uv` Integration:** `uv` uses PEP 517 build setting forwarding. Injecting `--config-settings="--build-option=--yes"` instructs `uv` to pass `--yes` directly to AGAMA's `setup.py` underlying C++ compiler setup. This prevents interactive terminal prompts from hanging unattended builds, CI pipelines, or headless installations while ensuring all header include paths match your target OS.
- **`Nix` Integration:** Controlled via `flake.nix`, Nix intercepts the build phase to patch AGAMA's setup process dynamically. It links AGAMA directly against fixed derivations of `gsl`, `eigen`, and `openblas` inside the Nix store, preventing network calls or host-system header mismatches during compilation.

---

### Verification

After completing setup, verify your installation by running the test suite:

```bash
uv run pytest
```

---

## 📚 Documentation & Resources

Comprehensive guides, API references, and interactive tutorials are available on **[ocd-gd.readthedocs.io](https://ocd-gd.readthedocs.io/en/latest/)**.

[![Documentation Status](https://readthedocs.org/projects/ocd-gd/badge/?version=latest)](https://ocd-gd.readthedocs.io/en/latest/?badge=latest) [![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)

### 🚀 Quick Links

- **[Getting Started Guide](https://ocd-gd.readthedocs.io/en/latest/quickstart.html)** — Installation, Agama potential setup, and first orbit integration.
- **[Example Gallery](https://ocd-gd.readthedocs.io/en/latest/examples/index.html)** — Executable recipes ranging from single orbit SALI/GALI checks to 2D grid chaos mapping.
- **[API Reference](https://ocd-gd.readthedocs.io/en/latest/modules.html)** — Detailed docstrings and signatures for `OrbitChaosDetector`, `GridChaosDetector`, and visualization mixins.

### 💻 Running Examples Locally

You can explore and run the full executable example suite locally:

```bash
# Run the automated test runner across all examples
python examples

# Or execute a specific topic script
python examples/05_grid_chaos_map_plot.py
```
