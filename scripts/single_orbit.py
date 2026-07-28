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
SUBSET = 900
idx = 441
MW_POTENTIAL_PATH = "data/potentials/MWPotentialHunter24_full.ini"
DATASET_PATH = Path(f"data/initial_conditions/labeled_ics_benchmark_size_{NUM}.npz")

# Baseline Parameters for Diagnostics
SALI_THRESHOLD = 1e-3
GALI_THRESHOLD = 1e-20
SALI_WINDOW_SIZE = 25
GALI_WINDOW_SIZE = 50


# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================
def main():
    mw_potential = agama.Potential(MW_POTENTIAL_PATH)
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
        summary = detector.detect_chaos()
        detector.plot_dashboard()
        print(summary)

    print("DONEZOO")


if __name__ == "__main__":
    main()
