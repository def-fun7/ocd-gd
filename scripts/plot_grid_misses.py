import ast
import re
from pathlib import Path
from typing import List, Set
import numpy as np
import agama

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocd_gd.orbit_detector import OrbitChaosDetector

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION
# ==============================================================================
NUM = 962
SUBSET = 100
MW_POTENTIAL_PATH = "data/potentials/MWPotentialHunter24_full.ini"
MISCLASSIFIED_IDX = Path(
    f"./outputs/misclassified_datasets/misclassified_indices_{SUBSET}.npz"
)
DATASET_PATH = Path(f"data/initial_conditions/labeled_ics_benchmark_size_{NUM}.npz")


OUTPUT_DIR = Path(f"./outputs/misclassified_dashboard/ds_{NUM}_ss_{SUBSET}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline Parameters for Diagnostics
SALI_THRESHOLD = 1e-3
GALI_THRESHOLD = 1e-20
SALI_WINDOW_SIZE = 25
GALI_WINDOW_SIZE = 50


def load_misclassified_indices(misclassified_idx):
    data = np.load(misclassified_idx)
    arrays = {
        "sali_fp": data["sali_fp"],
        "sali_fn": data["sali_fn"],
        "gali_fp": data["gali_fp"],
        "gali_fn": data["gali_fn"],
    }

    # 2. Extract sorted unique indices
    unique_indices = np.unique(np.hstack(list(arrays.values())))
    presence_map = []
    for idx in unique_indices:
        sources = [name for name, arr in arrays.items() if idx in arr]
        presence_map.append(sources)

    return unique_indices, presence_map


# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================
def main():
    mw_potential = agama.Potential(MW_POTENTIAL_PATH)
    idx, src = load_misclassified_indices(misclassified_idx=MISCLASSIFIED_IDX)
    data = np.load(DATASET_PATH)
    selected_ics = data["ics"][idx]

    # # 3. Plot Chaotic Dashboards (if any exist)
    if len(selected_ics) > 0:
        print(
            f"\nInitializing detector and running orbits for {len(selected_ics)} MISSES..."
        )
        detector = OrbitChaosDetector(
            ic=selected_ics,
            pot=mw_potential,
            sali_threshold=SALI_THRESHOLD,
            gali_threshold=GALI_THRESHOLD,
            sali_window_size=SALI_WINDOW_SIZE,
            gali_window_size=GALI_WINDOW_SIZE,
        )
        for i in range(len(idx)):
            missed_in = " | ".join(src[i])
            save_file = OUTPUT_DIR / f"orbit_{idx[i]}_{missed_in}.png"
            detector.plot_dashboard(
                orbit_idx=i,
                show=False,
                save_path=str(save_file),
                title=f"Orbit #{idx[i]} in {missed_in}",
            )

    print("DONEZOO")


if __name__ == "__main__":
    main()
