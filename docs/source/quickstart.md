# Quickstart Guide

This guide will walk you through setting up your environment, initializing a galactic potential with **AGAMA**, and running your first orbit chaos analysis using **`ocd-gd`**.

---

## 1. Environment Setup

`ocd-gd` relies on native C++ libraries (AGAMA, GSL, Eigen3) linked to Python. We recommend using either **Nix** (for reproducible system toolchains) or **`uv`** (for lightning-fast Python virtual environment management).

### Option A: Using Nix (Recommended)

If you have Nix installed with Flakes enabled:

```bash
# Clone the repository
git clone [https://github.com/your-username/ocd-gd.git](https://github.com/your-username/ocd-gd.git)
cd ocd-gd

# Enter the isolated development shell (automatically sets environment paths)
nix develop
```

### Option B: Using uv & System Dependencies

If you prefer uv on macOS (via Homebrew) or Linux:

```bash
# 1. Run the automated setup script to configure C++ dependencies & paths

./setup.sh

# 2. Sync virtual environment dependencies and install ocd-gd in editable mode

uv sync
```

## 2. Basic Concepts

ocd-gd operates around two core abstractions:
Galactic Potentials: Potential models constructed via AGAMA (e.g., NFW halo, Miyamoto-Nagai disk, Hernquist bulge).
OrbitChaosDetector: The main engine that integrates initial phase-space conditions ($x, y, z, v_x, v_y, v_z$) and computes chaos indicators (such as Lyapunov exponents or frequency map analysis metrics). 3. Running Included Examples
The repository comes with pre-configured scripts in examples/:

````Bash

# Run single orbit setup

uv run python examples/01_single_orbit_setup.py
``` bash
# Run unit test suite

uv run pytest
````

## Next Steps

Check out the User Guide & Examples for grid sweeps and phase-space mapping.
Refer to the API Reference for detailed class signatures and algorithm parameters.
