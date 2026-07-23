"""Generate, filter, and export a balanced benchmark set of chaotic and regular Initial Conditions (ICs)
using OrbitChaosDetector with strict float64 precision, reproducible seeds, and a validation step.
"""

from pathlib import Path
import sys
import agama
import numpy as np

# Adjust import path to include local src module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocd_gd.orbit_detector import OrbitChaosDetector

# Agama global settings
agama.setUnits(length=1, mass=1, velocity=1)

# Configuration
INI_FILE = Path("data/potentials/MWPotentialHunter24_full.ini")
BASE_PATH = Path("data/initial_conditions/labeled_ics_benchmark.npz")

NUM_CANDIDATES = 1000

# Thresholds
CHAOTIC_MIN_MLE = 1e-1
CHAOTIC_MAX_MLE = 1.0
REGULAR_MAX_MLE = 0.0  # Near-zero threshold for regular orbits


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

    # 1. Run OrbitChaosDetector on candidate ICs
    print("Initializing OrbitChaosDetector and integrating candidates...")
    detector = OrbitChaosDetector(ic=candidate_ics, pot=potential)

    # 2. Extract Maximum Lyapunov Exponents (MLE)
    lyap_arr = np.asarray(detector.lyapunov_exponents, dtype=np.float64)
    mle_values = lyap_arr[:, 0] if lyap_arr.ndim > 1 else lyap_arr

    # 3. Mask chaotic and regular orbits
    chaotic_mask = (mle_values > CHAOTIC_MIN_MLE) & (mle_values < CHAOTIC_MAX_MLE)
    regular_mask = mle_values == REGULAR_MAX_MLE

    chaotic_ics = candidate_ics[chaotic_mask]
    chaotic_mles = mle_values[chaotic_mask]

    regular_ics = candidate_ics[regular_mask]
    regular_mles = mle_values[regular_mask]

    num_chaotic = len(chaotic_ics)
    print(f"Found {num_chaotic} chaotic and {len(regular_ics)} regular candidate ICs.")

    if num_chaotic == 0:
        raise ValueError(
            "No chaotic orbits found with current thresholds. Adjust candidate pool or thresholds."
        )

    if len(regular_ics) < num_chaotic:
        print(
            f"Warning: Fewer regular ICs ({len(regular_ics)}) than chaotic ({num_chaotic}). Using all regular ICs."
        )
        selected_regular_ics = regular_ics
        selected_regular_mles = regular_mles
    else:
        # Sample regular ICs to match the exact size of chaotic ICs
        selected_regular_ics = regular_ics[:num_chaotic]
        selected_regular_mles = regular_mles[:num_chaotic]

    # Combine datasets
    combined_ics = np.vstack([chaotic_ics, selected_regular_ics])
    combined_mles = np.concatenate([chaotic_mles, selected_regular_mles])
    combined_labels = np.concatenate(
        [
            np.ones(len(chaotic_ics), dtype=np.int32),  # 1 = Chaotic
            np.zeros(len(selected_regular_ics), dtype=np.int32),  # 0 = Regular
        ]
    )

    # Shuffle dataset so 1s and 0s are interspersed
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(combined_labels))

    final_ics = combined_ics[shuffle_idx]
    final_mles = combined_mles[shuffle_idx]
    final_labels = combined_labels[shuffle_idx]
    indices = np.arange(len(final_labels), dtype=np.int32)

    # 4. Save dataset to NPZ
    OUTPUT_NPZ = BASE_PATH.with_stem(f"{BASE_PATH.stem}_size_{len(final_labels)}")
    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_NPZ,
        indices=indices,
        ics=final_ics,
        mles=final_mles,
        labels=final_labels,
    )
    print(f"Dataset successfully saved to '{OUTPUT_NPZ}'.")

    # -------------------------------------------------------------
    # 5. Validation Step: Reload and Rerun
    # -------------------------------------------------------------
    print("\n--- Running Validation Step ---")
    data = np.load(OUTPUT_NPZ)
    val_ics = data["ics"]
    val_mles = data["mles"]
    val_labels = data["labels"]

    print("Re-evaluating saved ICs through OrbitChaosDetector...")
    val_detector = OrbitChaosDetector(ic=val_ics, pot=potential)

    re_lyap_arr = np.asarray(val_detector.lyapunov_exponents, dtype=np.float64)
    re_mles = re_lyap_arr[:, 0] if re_lyap_arr.ndim > 1 else re_lyap_arr

    # Compute Absolute Error
    mle_diff = np.abs(val_mles - re_mles)
    max_err = np.max(mle_diff)
    mean_err = np.mean(mle_diff)

    # Re-calculate predicted labels based on thresholds
    re_labels = np.zeros_like(val_labels)
    re_labels[(re_mles > CHAOTIC_MIN_MLE) & (re_mles < CHAOTIC_MAX_MLE)] = 1

    class_match_rate = np.mean(re_labels == val_labels) * 100

    print(f"Validation Results:")
    print(f" - Max MLE Difference:  {max_err:.2e}")
    print(f" - Mean MLE Difference: {mean_err:.2e}")
    print(f" - Label Match Rate:    {class_match_rate:.2f}%")

    if max_err < 1e-5:
        print(" SUCCESS: Loaded data accurately matches re-calculated orbits!")
    else:
        print(
            " WARNING: Significant deviation detected between original and re-calculated MLEs."
        )


if __name__ == "__main__":
    main()
