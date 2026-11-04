# Orbital Chaos Detector - Galactic Dynamics (ocd-gd)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE) [![Python Version](https://img.shields.io/badge/3.13+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![AGAMA Core](https://img.shields.io/badge/C%2B%2B%20Core-AGAMA-orange.svg)](https://github.com/GalacticDynamics-Oxford/AGAMA) [![Build Tool](https://img.shields.io/badge/package%20manager-uv-de5b44.svg?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
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
chmod +x setup.sh
# Run the automated setup
./setup.sh
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

## Quick example

```python
import agama
from ocd_gd import OrbitChaosDetector

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0, gamma=1.0, beta=4.0)

detector = OrbitChaosDetector(
    ic=[8.0, 0.0, 0.05, 0.0, 180.0, 5.0], pot=pot, n_periods=30
)
report = detector.detect_chaos()

report.gali_check  # 1.0 if GALI_6 classifies the orbit as chaotic, else 0.0
report.gali["score"]  # chaos score in [0, 1]
report.gali["rate_norm"]  # normalised decay rate (decay rate x orbital period)
```

Each orbit is classified by fitting decay laws to the SALI and GALI_6 time series (plateau, power
law, exponential, plateau -> exponential, plateau -> power law) and selecting among them with the
BIC. See the [Getting Started Guide](https://ocd-gd.readthedocs.io/en/latest/quickstart.html)
for the result fields, the integration-length options, and 2D chaos maps.

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
python -m examples

# Or execute a specific topic script
python examples/05_grid_chaos_map_plot.py
```

---

> **Changelog Reference:**
>
> - Core package updates: [`CHANGELOG.md`](CHANGELOG.md)
> - Example suite updates: [`examples/CHANGELOG.md`](examples/CHANGELOG.md)
