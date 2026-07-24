"""Detailed Orbit Classification Diagnostic Script (Pure NumPy).

Evaluates orbit classification using locked optimal SALI and GALI parameters.
Outputs an in-depth metrics report and saves the misclassified orbit indices
to an NPZ file for downstream analysis.
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
NUM = 370
SUBSET = 370  # Adjust or set to NUM as needed

INI_FILE = Path("data/potentials/MWPotentialHunter24_full.ini")
DATASET_PATH = Path(f"data/initial_conditions/labeled_ics_benchmark_size_{NUM}.npz")

OUTPUT_TXT = Path(f"./outputs/reports/detailed_metrics_subset_{SUBSET}.txt")
OUTPUT_INDICES_NPZ = Path(
    f"./outputs/misclassified_datasets/misclassified_indices_{SUBSET}.npz"
)

# --- LOCKED OPTIMAL PARAMETERS ---
SALI_OPT_THRESH = 1.0e-03
SALI_OPT_WIN = 25

GALI_OPT_THRESH = 1.0e-20
GALI_OPT_WIN = 50


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes comprehensive classification metrics and extracts misclassified indices."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    n_samples = len(y_true)

    # Boolean masks for confusion components
    tp_mask = y_true & y_pred
    tn_mask = ~y_true & ~y_pred
    fp_mask = ~y_true & y_pred  # False Positive: Regular classified as Chaotic
    fn_mask = y_true & ~y_pred  # False Negative: Chaotic classified as Regular

    tp = int(np.sum(tp_mask))
    tn = int(np.sum(tn_mask))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))

    # Extract 0-based indices in the dataset
    indices_fp = np.where(fp_mask)[0]
    indices_fn = np.where(fn_mask)[0]

    # Calculate metrics safely
    acc = (tp + tn) / n_samples if n_samples > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    bal_acc = 0.5 * (tpr + tnr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "n_samples": n_samples,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": float(acc),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "bal_acc": float(bal_acc),
        "precision": float(precision),
        "f1": float(f1),
        "fp_indices": indices_fp,
        "fn_indices": indices_fn,
    }


def main():
    if not INI_FILE.exists():
        raise FileNotFoundError(f"Potential file '{INI_FILE}' not found!")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Benchmark dataset '{DATASET_PATH}' not found!")

    # Ensure output directories exist
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark dataset
    print(f"Loading dataset from '{DATASET_PATH}'...")
    data = np.load(DATASET_PATH)
    ics = data["ics"][:SUBSET]
    y_true = data["labels"][:SUBSET]  # 1 = chaotic, 0 = regular

    n_chaotic = int(np.sum(y_true == 1))
    n_regular = int(np.sum(y_true == 0))
    print(f"Loaded {len(y_true)} ICs (Chaotic: {n_chaotic}, Regular: {n_regular})")

    # 2. Initialize detector
    print("Initializing potential and detector...")
    potential = agama.Potential(file=str(INI_FILE))
    detector = OrbitChaosDetector(ic=ics, pot=potential)

    # 3. Run detection with optimal parameters
    print("Running chaos detection...")
    summary = detector.detect_chaos(
        separate_sali=False,
        check_only=True,
        sali_threshold_override=SALI_OPT_THRESH,
        sali_window_override=SALI_OPT_WIN,
        gali_threshold_override=GALI_OPT_THRESH,
        gali_window_override=GALI_OPT_WIN,
    )

    # 4. Compute metrics and indices
    s_pred = summary.sali_check.flatten()
    g_pred = summary.gali_check.flatten()

    sali_metrics = compute_detailed_metrics(y_true, s_pred)
    gali_metrics = compute_detailed_metrics(y_true, g_pred)

    # 5. Extract 4 misclassification arrays:
    # 1) SALI False Positives (Regular marked as Chaotic)
    # 2) SALI False Negatives (Chaotic marked as Regular)
    # 3) GALI False Positives (Regular marked as Chaotic)
    # 4) GALI False Negatives (Chaotic marked as Regular)
    sali_fp_indices = sali_metrics["fp_indices"]
    sali_fn_indices = sali_metrics["fn_indices"]
    gali_fp_indices = gali_metrics["fp_indices"]
    gali_fn_indices = gali_metrics["fn_indices"]

    # Save indices to NPZ for downstream analysis
    np.savez_compressed(
        OUTPUT_INDICES_NPZ,
        sali_fp=sali_fp_indices,
        sali_fn=sali_fn_indices,
        gali_fp=gali_fp_indices,
        gali_fn=gali_fn_indices,
    )
    print(f"Saved misclassified indices to '{OUTPUT_INDICES_NPZ}'")

    # 6. Generate detailed text report
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(
            "====================================================================\n"
        )
        f.write(
            "             DETAILED ORBIT CLASSIFICATION REPORT                   \n"
        )
        f.write(
            "====================================================================\n\n"
        )
        f.write(f"Dataset File: {DATASET_PATH.name}\n")
        f.write(f"Evaluated Subset Size: {len(y_true)}\n")
        f.write(f"Class Distribution   : Chaotic={n_chaotic}, Regular={n_regular}\n\n")

        f.write("--- PARAMETERS USED ---\n")
        f.write(
            f"SALI Optimal: Threshold = {SALI_OPT_THRESH:.1e}, Window = {SALI_OPT_WIN}\n"
        )
        f.write(
            f"GALI Optimal: Threshold = {GALI_OPT_THRESH:.1e}, Window = {GALI_OPT_WIN}\n\n"
        )

        f.write(
            "====================================================================\n"
        )
        f.write(
            "                      DETAILED METRIC SUMMARY                       \n"
        )
        f.write(
            "====================================================================\n"
        )
        f.write(f"{'Metric':<25} | {'SALI':<18} | {'GALI':<18}\n")
        f.write("-" * 68 + "\n")
        f.write(
            f"{'Balanced Accuracy':<25} | {sali_metrics['bal_acc']:<18.4f} | {gali_metrics['bal_acc']:<18.4f}\n"
        )
        f.write(
            f"{'Standard Accuracy':<25} | {sali_metrics['acc']:<18.4f} | {gali_metrics['acc']:<18.4f}\n"
        )
        f.write(
            f"{'F1-Score':<25} | {sali_metrics['f1']:<18.4f} | {gali_metrics['f1']:<18.4f}\n"
        )
        f.write(
            f"{'Precision':<25} | {sali_metrics['precision']:<18.4f} | {gali_metrics['precision']:<18.4f}\n"
        )
        f.write(
            f"{'Recall / Sensitivity (TPR)':<25} | {sali_metrics['tpr']:<18.4f} | {gali_metrics['tpr']:<18.4f}\n"
        )
        f.write(
            f"{'Specificity (TNR)':<25} | {sali_metrics['tnr']:<18.4f} | {gali_metrics['tnr']:<18.4f}\n"
        )
        f.write("-" * 68 + "\n")
        f.write(
            f"{'True Positives (TP)':<25} | {sali_metrics['tp']:<18} | {gali_metrics['tp']:<18}\n"
        )
        f.write(
            f"{'True Negatives (TN)':<25} | {sali_metrics['tn']:<18} | {gali_metrics['tn']:<18}\n"
        )
        f.write(
            f"{'False Positives (FP)':<25} | {sali_metrics['fp']:<18} | {gali_metrics['fp']:<18}\n"
        )
        f.write(
            f"{'False Negatives (FN)':<25} | {sali_metrics['fn']:<18} | {gali_metrics['fn']:<18}\n"
        )
        f.write(
            "====================================================================\n\n"
        )

        f.write("--- MISCLASSIFIED ORBIT INDEX SUMMARY ---\n")
        f.write(f"SALI FP Count (Regular -> Chaotic): {len(sali_fp_indices)}\n")
        f.write(f"SALI FN Count (Chaotic -> Regular): {len(sali_fn_indices)}\n")
        f.write(f"GALI FP Count (Regular -> Chaotic): {len(gali_fp_indices)}\n")
        f.write(f"GALI FN Count (Chaotic -> Regular): {len(gali_fn_indices)}\n\n")

        f.write("Index Lists:\n")
        f.write(f"sali_fp = {sali_fp_indices.tolist()}\n")
        f.write(f"sali_fn = {sali_fn_indices.tolist()}\n")
        f.write(f"gali_fp = {gali_fp_indices.tolist()}\n")
        f.write(f"gali_fn = {gali_fn_indices.tolist()}\n")

    print(f"Report successfully saved to '{OUTPUT_TXT}'.")


if __name__ == "__main__":
    main()
