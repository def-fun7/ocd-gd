# Quickstart Guide

This guide will walk you through setting up your environment, initializing a galactic potential with **AGAMA**, and running your first orbit chaos analysis using **`ocd-gd`**.

---

## 1. Environment Setup

`ocd-gd` relies on native C++ libraries (AGAMA, GSL, Eigen3) linked to Python. We recommend using either **Nix** (for reproducible system toolchains) or **`uv`** (for lightning-fast Python virtual environment management).

### Option A: Using Nix (Recommended)

If you have Nix installed with Flakes enabled:

```bash
# Clone the repository
git clone https://github.com/def-fun7/ocd-gd.git
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

`ocd-gd` has two main building blocks:

- **Galactic potentials** built with AGAMA (e.g. NFW halo, Miyamoto-Nagai disc, Hernquist bulge).
- **Detectors.** `OrbitChaosDetector` integrates one or many orbits together with their deviation
  vectors and computes the SALI (15 deviation-vector pairs, reduced to their minimum envelope)
  and GALI_6 time series. It then classifies every orbit as regular or chaotic by fitting decay
  laws to the log-indicator and choosing between them with the BIC. AGAMA's own Lyapunov
  exponent is reported alongside. `GridChaosDetector` does the same for a 2D grid of initial
  conditions at fixed energy and turns the results into chaos maps.

## 3. Your First Orbit

```python
import agama
from ocd_gd import OrbitChaosDetector

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
pot = agama.Potential(type="Spheroid", mass=1e10, scaleRadius=5.0, gamma=1.0, beta=4.0)

detector = OrbitChaosDetector(
    ic=[8.0, 0.0, 0.05, 0.0, 180.0, 5.0], pot=pot, n_periods=30
)
report = detector.detect_chaos()  # a DecayClassification
```

## 4. Reading the Result

`detect_chaos()` returns a `DecayClassification` with one dictionary per indicator
(`report.sali`, `report.gali`), plus convenience arrays:

| Field                                             | Meaning                                                                                                                                         |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.sali_check`, `report.gali_check`          | `1.0` chaotic, `0.0` otherwise (undetermined orbits count as `0.0`). With `separate_sali=True` the SALI result holds the 15 pairs individually. |
| `report.sali_time`, `report.gali_time`            | Onset time (plateau -> exponential) or floor-hit time; `inf` if neither exists.                                                                 |
| `report.sali["score"]`, `report.gali["score"]`    | Classifier chaos score in [0, 1], the weight of the chaotic decay models.                                                                       |
| `report.gali["rate"]`, `report.gali["rate_norm"]` | Exponential decay rate, and the same rate times the orbital period (`0` for regular orbits).                                                    |
| `report.gali["label"]`                            | Decay-law code, see below.                                                                                                                      |
| `report.lyapunov_check`, `report.lyapunov_time`   | AGAMA's Lyapunov-based flag and time, as a cross-check.                                                                                         |

Label codes follow one rule: **odd means regular, even means chaotic, 0 means undetermined**.

| Code | Meaning                          |
| ---- | -------------------------------- |
| 0    | undetermined                     |
| 1    | regular (plateau)                |
| 2    | chaotic (exponential)            |
| 3    | regular (power law)              |
| 4    | chaotic (plateau -> exponential) |
| 5    | regular (plateau -> power law)   |
| 6    | chaotic (floor hit, censored)    |

`ocd_gd.LABELS` holds the same table in code.

## 5. Choosing the Integration Length

Give either `n_periods` or `iter_time`, never both (a `ValueError` is raised):

- `n_periods` (default 30 when neither is given): the integration time of each orbit is
  `n_periods * pot.Tcirc(ic)`, so orbits at different radii get comparable lengths.
- `iter_time`: a fixed integration time in AGAMA time units.
- `traj_size` (default 5000) is the number of stored output points per orbit.

The classification needs enough time to tell a decay law from a plateau. Very short integrations
can give marginal, low-confidence calls (a low `score` or a small `delta_bic`), so check these
fields before trusting a single orbit. Both `n_periods` and `traj_size` can change a marginal
orbit's label, and the validation work on this is ongoing.

## 6. Chaos Maps on a Grid

`GridChaosDetector` takes the same `n_periods` / `iter_time` / `traj_size` arguments and adds
`map_values`, which chooses what the SALI, GALI and Lyapunov panels of `plot_chaos_map()` show:

| `map_values`         | SALI / GALI panels    | Lyapunov panel             | Colour scale                                                   |
| -------------------- | --------------------- | -------------------------- | -------------------------------------------------------------- |
| `"checks"` (default) | 0/1 chaotic flag      | 0/1 flag                   | two colours                                                    |
| `"score"`            | chaos score in [0, 1] | 0/1 flag (it has no score) | gradient, 0-1                                                  |
| `"rate_norm"`        | normalised decay rate | AGAMA's Lyapunov value     | gradient from 0 to the 99th percentile (override with `vmax=`) |

Cells outside the physical region are masked. In `"score"` and `"rate_norm"` modes undetermined
orbits are masked as well, while in `"checks"` mode they appear as regular. The titles show the
chosen type, e.g. `SALI Chaos Map (10x10, score)`. The composite and consensus maps always use
the binary checks.

```python
from ocd_gd import GridChaosDetector

detector = GridChaosDetector(potential=pot, R_0=8.0, grid_size=20, map_values="score")
detector.plot_chaos_map()  # SALI / GALI / Lyapunov score maps
detector.plot_composite_chaos_map()  # always binary
```

## 7. Running Included Examples

(unchanged: keep the existing block)## 3. Running Included Examples

The repository comes with pre-configured scripts in `examples/`:

```bash
# Run single orbit setup
uv run python examples/01_single_orbit_setup.py

# Run unit test suite
uv run pytest
```

## Next Steps

Check out the User Guide & Examples for grid sweeps and phase-space mapping.
Refer to the API Reference for detailed class signatures and algorithm parameters.
