"""Grid Search for SALI and GALI Optimal Thresholds and Window Sizes (Pure NumPy).

Evaluates 25 combined parameter pairs on a pre-generated benchmark dataset and exports
a summary metrics table to a text file without external ML library dependencies.
"""

from pathlib import Path
import sys
import agama
import numpy as np

# Adjust path to include src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocd_gd.orbit_detector import OrbitChaosDetector

# Agama global settings
agama.setUnits(length=1, mass=1, velocity=1)

# Paths
NUM = 962
SUBSET = int(962 / 2)

INI_FILE = Path("data/potentials/MWPotentialHunter24_full.ini")
DATASET_PATH = Path(f"data/initial_conditions/labeled_ics_benchmark_size_{NUM}.npz")
OUTPUT_TXT = Path(f"./outputs/reports/grid_search_results_{SUBSET}.txt")

# Define parameter grids
SALI_THRESHOLDS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
SALI_WINDOWS = [5, 10, 15, 20, 25]

GALI_THRESHOLDS = [1e-14, 1e-16, 1e-18, 1e-20, 1e-22]
GALI_WINDOWS = [50, 100, 150, 200, 250]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Force boolean arrays
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    # Total number of samples
    n_samples = len(y_true)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    # Standard Accuracy (0.0 to 1.0)
    acc = (tp + tn) / n_samples if n_samples > 0 else 0.0

    # Recall / Specificity
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced Accuracy (0.0 to 1.0)
    bal_acc = 0.5 * (tpr + tnr)

    # Precision & F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {"acc": float(acc), "bal_acc": float(bal_acc), "f1": float(f1)}


def main():
    if not INI_FILE.exists():
        raise FileNotFoundError(f"Potential file '{INI_FILE}' not found!")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark dataset '{DATASET_PATH}' not found! Run IC generator first."
        )

    # 1. Load ground truth benchmark dataset
    print(f"Loading dataset from '{DATASET_PATH}'...")
    data = np.load(DATASET_PATH)
    ics = data["ics"][:SUBSET]
    y_true = data["labels"][:SUBSET]  # 1 = chaotic, 0 = regular

    print(
        f"Loaded {len(y_true)} labeled ICs (Chaotic: {np.sum(y_true == 1)}, Regular: {np.sum(y_true == 0)})"
    )

    # 2. Initialize potential and detector ONCE
    print("Loading potential and initializing OrbitChaosDetector...")
    potential = agama.Potential(file=str(INI_FILE))
    detector = OrbitChaosDetector(ic=ics, pot=potential)

    sali_results = []
    gali_results = []

    print("\nRunning 25 combined grid evaluations...")

    run_count = 0
    # Iterate through 5x5 combinations
    for s_thresh in SALI_THRESHOLDS:
        for s_win in SALI_WINDOWS:
            # Pair 1-to-1 with corresponding GALI index
            g_thresh = GALI_THRESHOLDS[run_count // 5]
            g_win = GALI_WINDOWS[run_count % 5]
            run_count += 1

            # 3. Execute detect_chaos ONCE per joint parameter pair
            summary = detector.detect_chaos(
                separate_sali=False,
                check_only=True,
                sali_threshold_override=s_thresh,
                sali_window_override=s_win,
                gali_threshold_override=g_thresh,
                gali_window_override=g_win,
            )

            # Process SALI predictions
            s_metrics = compute_metrics(y_true, summary.sali_check.flatten())
            sali_results.append(
                {
                    "method": "SALI",
                    "threshold": s_thresh,
                    "window": s_win,
                    **s_metrics,
                }
            )

            # Process GALI predictions
            g_metrics = compute_metrics(y_true, summary.gali_check.flatten())
            gali_results.append(
                {
                    "method": "GALI",
                    "threshold": g_thresh,
                    "window": g_win,
                    **g_metrics,
                }
            )

    # Sort results by Balanced Accuracy (descending)
    sali_sorted = sorted(sali_results, key=lambda x: x["bal_acc"], reverse=True)
    gali_sorted = sorted(gali_results, key=lambda x: x["bal_acc"], reverse=True)

    # 4. Write output to TXT summary file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(
            "========================================================================================\n"
        )
        f.write(
            "                         GRID SEARCH RESULTS (SALI & GALI)                              \n"
        )
        f.write(
            "========================================================================================\n\n"
        )
        f.write(
            f"Loaded {len(y_true)} labeled ICs (Chaotic: {np.sum(y_true == 1)}, Regular: {np.sum(y_true == 0)}) from {DATASET_PATH}"
        )
        f.write("\n--- BEST PARAMETER SETS ---\n")
        f.write(
            f"Best SALI: Threshold = {sali_sorted[0]['threshold']:.1e}, Window = {sali_sorted[0]['window']} "
            f"| Balanced Acc = {sali_sorted[0]['bal_acc']:.4f}\n"
        )
        f.write(
            f"Best GALI: Threshold = {gali_sorted[0]['threshold']:.1e}, Window = {gali_sorted[0]['window']} "
            f"| Balanced Acc = {gali_sorted[0]['bal_acc']:.4f}\n\n"
        )

        f.write("=" * 88 + "\n")
        f.write(
            f"{'Method':<8} | {'Threshold':<12} | {'Window':<8} | {'Bal. Acc':<10} | {'Accuracy':<10} | {'F1-Score':<10}\n"
        )
        f.write("=" * 88 + "\n")

        f.write("--- SALI RUNS ---\n")
        for res in sali_sorted:
            f.write(
                f"{res['method']:<8} | {res['threshold']:<12.1e} | {res['window']:<8} | "
                f"{res['bal_acc']:<10.4f} | {res['acc']:<10.4f} | {res['f1']:<10.4f}\n"
            )

        f.write("-" * 88 + "\n")

        f.write("--- GALI RUNS ---\n")
        for res in gali_sorted:
            f.write(
                f"{res['method']:<8} | {res['threshold']:<12.1e} | {res['window']:<8} | "
                f"{res['bal_acc']:<10.4f} | {res['acc']:<10.4f} | {res['f1']:<10.4f}\n"
            )

        f.write("=" * 88 + "\n")

    print(f"\nGrid search completed successfully! Report generated at '{OUTPUT_TXT}'.")


if __name__ == "__main__":
    main()
