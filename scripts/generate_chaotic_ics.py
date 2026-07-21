"""Generate, filter, and export a benchmark set of chaotic Initial Conditions (ICs)
using AGAMA with strict float64 precision and reproducible seeds.
"""

import csv
from pathlib import Path
import agama
import numpy as np

# Agama global settings
agama.setUnits(length=1, mass=1, velocity=1)

# Configuration
INI_FILE = Path("data/potentials/MWPotentialHunter24_full.ini")
BASE_PATH = Path("data/initial_conditions/chaotic_ics_benchmark.npz")

NUM_CANDIDATES = 1000
TIME_SPAN = 10
NUM_STEPS = 1000

MIN_MLE_THRESHOLD = 1e-1
MAX_MLE_THRESHOLD = 1.0


def generate_bound_phase_space_points(
    potential: agama.Potential, n_samples: int
) -> np.ndarray:
    """Generate 6D phase space points forced to float64 precision and bound energy."""
    np.random.seed(42)
    bound_ics = []

    while len(bound_ics) < n_samples:
        r = np.random.uniform(1.0, 20.0, 200)
        theta = np.random.uniform(0, np.pi, 200)
        phi = np.random.uniform(0, 2 * np.pi, 200)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        vx = np.random.uniform(-250.0, 250.0, 200)
        vy = np.random.uniform(-250.0, 250.0, 200)
        vz = np.random.uniform(-250.0, 250.0, 200)

        pts = np.column_stack([x, y, z, vx, vy, vz]).astype(np.float64)

        # Filter bound orbits (E = K + V < 0)
        pot_vals = potential.potential(pts[:, :3])
        kinetic = 0.5 * np.sum(pts[:, 3:] ** 2, axis=1)
        total_energy = kinetic + pot_vals

        valid = pts[total_energy < 0.0]
        bound_ics.extend(valid)

    return np.array(bound_ics[:n_samples], dtype=np.float64)


def main():
    if not INI_FILE.exists():
        raise FileNotFoundError(f"Potential file '{INI_FILE}' not found!")

    print(f"Loading potential from '{INI_FILE}'...")
    potential = agama.Potential(file=str(INI_FILE))

    print(f"Generating {NUM_CANDIDATES} candidate initial conditions (float64)...")
    candidate_ics = generate_bound_phase_space_points(potential, NUM_CANDIDATES)

    print(f"Integrating orbits with lyapunov=True over t={TIME_SPAN}...")
    # Omit separateTime=True to get summary array at t=TIME_SPAN directly
    _, _, lyap_arr = agama.orbit(
        ic=candidate_ics,
        potential=potential,
        time=TIME_SPAN,
        trajSize=NUM_STEPS,
        lyapunov=True,
        separateTime=True,
    )

    # Column 0 holds the finite-time MLE at final time t=TIME_SPAN
    mle_values = lyap_arr[:, 0].astype(np.float64)

    # Filter out regular or unphysical orbits
    valid_mask = (mle_values > MIN_MLE_THRESHOLD) & (mle_values < MAX_MLE_THRESHOLD)

    filtered_ics = candidate_ics[valid_mask]
    filtered_mle = mle_values[valid_mask]

    print(
        f"Retained {len(filtered_mle)} / {NUM_CANDIDATES} chaotic initial conditions."
    )

    # Sort descending by MLE value
    sort_idx = np.argsort(filtered_mle)[::-1]
    sorted_ics = np.ascontiguousarray(filtered_ics[sort_idx], dtype=np.float64)
    sorted_mle = np.ascontiguousarray(filtered_mle[sort_idx], dtype=np.float64)

    # 1. Export to NPZ (Guarantees zero-loss exact binary float64 retrieval)
    OUTPUT_NPZ = BASE_PATH.with_stem(
        f"{BASE_PATH.stem}_{len(filtered_mle)}_{MIN_MLE_THRESHOLD}"
    )
    np.savez(OUTPUT_NPZ, ics=sorted_ics, mles=sorted_mle)
    print(f"Exported exact binary float64 array to '{OUTPUT_NPZ}'.")


if __name__ == "__main__":
    main()
