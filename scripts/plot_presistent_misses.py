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
MW_POTENTIAL_PATH = "data/potentials/MWPotentialHunter24_full.ini"
CHAOTIC_BENCHMARK_NPZ = Path("data/initial_conditions/chaotic_ics_benchmark_185.npz")
REGULAR_BENCHMARK_NPZ = Path("data/initial_conditions/regular_ics_benchmark_226.npz")

ANALYSIS_FILE = Path("./outputs/sensitivity_reports/FINAL_SENSITIVITY_ANALYSIS.txt")
OUTPUT_DIR = Path("./outputs/persistent_misses_dashboards")

# Baseline Parameters for Diagnostics
SALI_THRESHOLD = 1e-3
GALI_THRESHOLD = 1e-20
SALI_WINDOW_SIZE = 10
GALI_WINDOW_SIZE = 100


def parse_persistent_misses_from_file(filepath: Path):
    """
    Reads the FINAL_SENSITIVITY_ANALYSIS.txt file and dynamically extracts
    the persistent miss index lists using regex.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Analysis report file not found at '{filepath}'. "
            "Please run 'analyze_sensitivity_reports.py' first!"
        )

    text = filepath.read_text(encoding="utf-8")

    def extract_list(pattern: str) -> List[int]:
        match = re.search(pattern, text)
        if match:
            return ast.literal_eval(match.group(1))
        return []

    sali_c_misses = extract_list(
        r"SALI_chaotic_persistent_misses\s*\(>4 runs\)\s*:\s*(\[.*?\])"
    )
    gali_c_misses = extract_list(
        r"GALI_chaotic_persistent_misses\s*\(>4 runs\)\s*:\s*(\[.*?\])"
    )
    sali_r_misses = extract_list(
        r"SALI_regular_persistent_misses\s*\(>4 runs\)\s*:\s*(\[.*?\])"
    )
    gali_r_misses = extract_list(
        r"GALI_regular_persistent_misses\s*\(>4 runs\)\s*:\s*(\[.*?\])"
    )

    return sali_c_misses, gali_c_misses, sali_r_misses, gali_r_misses


def load_benchmark_ics(filepath: Path) -> np.ndarray:
    """Loads initial conditions array from benchmark file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Benchmark file not found: {filepath}")
    data = np.load(filepath)
    return data["ics"]


def get_unique_indices(*index_lists: List[int]) -> List[int]:
    """Combines multiple index lists into a sorted list of unique indices."""
    unique_set: Set[int] = set()
    for lst in index_lists:
        unique_set.update(lst)
    return sorted(list(unique_set))


# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================
def main():
    print(f"Parsing persistent miss indices from {ANALYSIS_FILE}...")
    (
        sali_c_misses,
        gali_c_misses,
        sali_r_misses,
        gali_r_misses,
    ) = parse_persistent_misses_from_file(ANALYSIS_FILE)

    # 1. Get unique persistent miss indices dynamically from parsed files
    chaotic_indices_to_plot = get_unique_indices(sali_c_misses, gali_c_misses)
    regular_indices_to_plot = get_unique_indices(sali_r_misses, gali_r_misses)

    print(
        f"Chaotic persistent misses found ({len(chaotic_indices_to_plot)}): {chaotic_indices_to_plot}"
    )
    print(
        f"Regular persistent misses found ({len(regular_indices_to_plot)}): {regular_indices_to_plot}"
    )

    if not chaotic_indices_to_plot and not regular_indices_to_plot:
        print("No persistent misses found in the file. Exiting.")
        return

    # 2. Extract ICs for these indices from full benchmark sets
    print("\nLoading potential and benchmark initial conditions...")
    mw_potential = agama.Potential(MW_POTENTIAL_PATH)
    all_chaotic_ics = load_benchmark_ics(CHAOTIC_BENCHMARK_NPZ)
    all_regular_ics = load_benchmark_ics(REGULAR_BENCHMARK_NPZ)

    selected_chaotic_ics = (
        all_chaotic_ics[chaotic_indices_to_plot]
        if chaotic_indices_to_plot
        else np.array([])
    )
    selected_regular_ics = (
        all_regular_ics[regular_indices_to_plot]
        if regular_indices_to_plot
        else np.array([])
    )

    chaotic_output_dir = OUTPUT_DIR / "chaotic"
    regular_output_dir = OUTPUT_DIR / "regular"
    chaotic_output_dir.mkdir(parents=True, exist_ok=True)
    regular_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Plot Chaotic Dashboards (if any exist)
    if len(selected_chaotic_ics) > 0:
        print(
            "\nInitializing detector and running orbits for Chaotic Persistent Misses..."
        )
        chaotic_detector = OrbitChaosDetector(
            ic=selected_chaotic_ics,
            pot=mw_potential,
            sali_threshold=SALI_THRESHOLD,
            gali_threshold=GALI_THRESHOLD,
            sali_window_size=SALI_WINDOW_SIZE,
            gali_window_size=GALI_WINDOW_SIZE,
        )

        for internal_idx, original_orbit_idx in enumerate(chaotic_indices_to_plot):
            save_file = chaotic_output_dir / f"orbit_{original_orbit_idx:03d}.png"
            print(
                f"  [Chaotic] Plotting Orbit #{original_orbit_idx:03d} -> {save_file}"
            )
            chaotic_detector.plot_dashboard(
                orbit_idx=internal_idx,
                show=False,
                save_path=str(save_file),
            )

    # 4. Plot Regular Dashboards (if any exist)
    if len(selected_regular_ics) > 0:
        print(
            "\nInitializing detector and running orbits for Regular Persistent Misses..."
        )
        regular_detector = OrbitChaosDetector(
            ic=selected_regular_ics,
            pot=mw_potential,
            sali_threshold=SALI_THRESHOLD,
            gali_threshold=GALI_THRESHOLD,
            sali_window_size=SALI_WINDOW_SIZE,
            gali_window_size=GALI_WINDOW_SIZE,
        )

        for internal_idx, original_orbit_idx in enumerate(regular_indices_to_plot):
            save_file = regular_output_dir / f"orbit_{original_orbit_idx:03d}.png"
            print(
                f"  [Regular] Plotting Orbit #{original_orbit_idx:03d} -> {save_file}"
            )
            regular_detector.plot_dashboard(
                orbit_idx=internal_idx,
                show=False,
                save_path=str(save_file),
            )

    print(f"\nAll dashboards successfully generated in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
