from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import agama

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocd_gd.orbit_detector import OrbitChaosDetector

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION
# ==============================================================================
MW_POTENTIAL_PATH = "data/potentials/MWPotentialHunter24_full.ini"
CHAOTIC_BENCHMARK_NPZ = Path("data/initial_conditions/chaotic_ics_benchmark_185.npz")
REGULAR_BENCHMARK_NPZ = Path("data/initial_conditions/regular_ics_benchmark_226.npz")

OUTPUT_DIR = Path("./outputs/sensitivity_reports")

# Minimum runtime MLE threshold to qualify as chaotic
MIN_CHAOTIC_MLE_THRESHOLD = 0.1

# Target balanced class size per set
TARGET_CLASS_SIZE = 151

# Define the 6 specific parameter run configurations
RUN_CONFIGS: List[Dict[str, Any]] = [
    # Threshold Runs (Fixed Windows: SALI_W=10, GALI_W=100)
    {
        "name": "ST_2_GT_16_report.txt",
        "sali_th": 1e-2,
        "gali_th": 1e-16,
        "sali_w": 10,
        "gali_w": 100,
    },
    {
        "name": "ST_3_GT_18_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-18,
        "sali_w": 10,
        "gali_w": 100,
    },
    {
        "name": "ST_3_GT_20_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-20,
        "sali_w": 20,
        "gali_w": 100,
    },
    {
        "name": "ST_4_GT_20_report.txt",
        "sali_th": 1e-4,
        "gali_th": 1e-20,
        "sali_w": 10,
        "gali_w": 100,
    },
    {
        "name": "ST_4_SW_20_GT_20_report.txt",
        "sali_th": 1e-4,
        "gali_th": 1e-20,
        "sali_w": 20,
        "gali_w": 100,
    },
    # Window Runs (Fixed Thresholds: SALI_TH=1e-3, GALI_TH=1e-20)
    {
        "name": "SW_5_GW_10_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-20,
        "sali_w": 5,
        "gali_w": 10,
    },
    {
        "name": "SW_10_GW_100_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-20,
        "sali_w": 10,
        "gali_w": 100,
    },
    {
        "name": "SW_20_GW_100_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-20,
        "sali_w": 20,
        "gali_w": 100,
    },
    {
        "name": "SW_20_GW_200_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-20,
        "sali_w": 20,
        "gali_w": 200,
    },
    {
        "name": "SW_20_GW_500_report.txt",
        "sali_th": 1e-3,
        "gali_th": 1e-20,
        "sali_w": 20,
        "gali_w": 500,
    },
]


def load_benchmark_data(filepath: Path):
    """Loads initial conditions and ground truth MLEs from an NPZ benchmark file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Benchmark file not found: {filepath}")
    data = np.load(filepath)
    return data["ics"], data["mles"]


def format_missed_orbits_section(
    title: str,
    sali_passed: np.ndarray,
    gali_passed: np.ndarray,
    mles: np.ndarray,
    detector: OrbitChaosDetector,
) -> List[str]:
    """Generates detailed text lines for missed/misclassified orbits."""
    lines = []
    total_orbits = len(sali_passed)

    # Identify indices where either SALI or GALI failed the expected status
    missed_indices = [
        i for i in range(total_orbits) if not (sali_passed[i] and gali_passed[i])
    ]

    if not missed_indices:
        lines.append(f"\n{title}: None (100% accuracy)")
        return lines

    lines.append(f"\n{title}:")
    lines.append("-" * 75)

    # Safely retrieve Lyapunov exponents from detector if available
    lyap_exponents = getattr(detector, "lyapunov_exponents", None)

    for idx in missed_indices:
        sali_status = "PASS" if sali_passed[idx] else "MISS"
        gali_status = "PASS" if gali_passed[idx] else "MISS"

        # Get detector's calculated Lyapunov exponent safely
        if lyap_exponents is not None and len(lyap_exponents) > idx:
            val = lyap_exponents[idx]

            # Handle cases where value is a numpy array slice/vector
            if isinstance(val, np.ndarray):
                val = val.flat[0]  # grabs the first scalar element safely

            calc_lyap = f"{float(val):.5f}"
        else:
            calc_lyap = "N/A"

        lines.append(
            f"Orbit #{idx:03d} | Ground MLE: {mles[idx]:.5f} | Calc MLE: {calc_lyap:<8} | "
            f"SALI: {sali_status:<6} | GALI: {gali_status:<6}"
        )
    return lines


# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================
def main():
    print("Initializing potential and benchmark datasets...")
    mw_potential = agama.Potential(MW_POTENTIAL_PATH)

    # 1. Load Raw Initial Conditions and Ground Truth MLEs
    raw_chaotic_ics, raw_chaotic_mles = load_benchmark_data(CHAOTIC_BENCHMARK_NPZ)
    raw_regular_ics, raw_regular_mles = load_benchmark_data(REGULAR_BENCHMARK_NPZ)

    # 2. Run initial chaotic integration to retrieve calc_mles and filter false misses
    print("Initializing chaotic orbit detector and extracting runtime Calc MLEs...")
    temp_chaotic_detector = OrbitChaosDetector(ic=raw_chaotic_ics, pot=mw_potential)

    # Retrieve calc_mles directly from the detector and force to 1D MLE
    raw_lyap = np.squeeze(temp_chaotic_detector.lyapunov_exponents)

    if raw_lyap.ndim > 1:
        raw_calc_mles = raw_lyap[:, 0]  # Take column 0 (the Maximum Lyapunov Exponent)
    else:
        raw_calc_mles = raw_lyap

    # Mask out orbits where Calc MLE <= MIN_CHAOTIC_MLE_THRESHOLD (e.g. 0.1)
    valid_chaotic_mask = raw_calc_mles > MIN_CHAOTIC_MLE_THRESHOLD

    filtered_chaotic_ics = raw_chaotic_ics[valid_chaotic_mask]
    filtered_chaotic_mles = raw_calc_mles[valid_chaotic_mask]  # Use Calc MLE as truth

    print(f"  • Original Chaotic Candidates : {len(raw_chaotic_ics)}")
    print(
        f"  • Filtered Chaotic Orbits (Calc MLE > {MIN_CHAOTIC_MLE_THRESHOLD}) : {len(filtered_chaotic_ics)}"
    )
    print(f"  • Dropped False Misses        : {np.sum(~valid_chaotic_mask)}")

    # 3. Truncate both benchmark sets to exact balanced length (151 items each)
    chaotic_ics = filtered_chaotic_ics[:TARGET_CLASS_SIZE]
    chaotic_mles = filtered_chaotic_mles[:TARGET_CLASS_SIZE]

    regular_ics = raw_regular_ics[:TARGET_CLASS_SIZE]
    regular_mles = raw_regular_mles[:TARGET_CLASS_SIZE]

    print(
        f"\nFinal Balanced Benchmark Size: {len(chaotic_ics)} Chaotic vs {len(regular_ics)} Regular (Total: {len(chaotic_ics) + len(regular_ics)})"
    )

    # 4. Initialize Detectors on truncated, balanced sets
    print("\nInitializing chaotic orbit detector (1/2)...")
    chaotic_detector = OrbitChaosDetector(ic=chaotic_ics, pot=mw_potential)

    print("Initializing regular orbit detector (2/2)...")
    regular_detector = OrbitChaosDetector(ic=regular_ics, pot=mw_potential)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Iterate over the 6 configurations
    for cfg in RUN_CONFIGS:
        print(f"\nProcessing config: {cfg['name']}...")

        # Run detection using overrides
        c_summary = chaotic_detector.detect_chaos(
            check_only=True,
            sali_threshold_override=cfg["sali_th"],
            gali_threshold_override=cfg["gali_th"],
            sali_window_override=cfg["sali_w"],
            gali_window_override=cfg["gali_w"],
        )
        r_summary = regular_detector.detect_chaos(
            check_only=True,
            sali_threshold_override=cfg["sali_th"],
            gali_threshold_override=cfg["gali_th"],
            sali_window_override=cfg["sali_w"],
            gali_window_override=cfg["gali_w"],
        )

        # Evaluate Chaotic Orbits (Expecting True for Chaos)
        c_sali_passed = c_summary.sali_check.flatten().astype(bool)
        c_gali_passed = c_summary.gali_check.flatten().astype(bool)

        c_total = len(chaotic_ics)
        c_sali_hits = int(np.sum(c_sali_passed))
        c_gali_hits = int(np.sum(c_gali_passed))
        c_sali_miss_indices = [i for i in range(c_total) if not c_sali_passed[i]]
        c_gali_miss_indices = [i for i in range(c_total) if not c_gali_passed[i]]

        # Evaluate Regular Orbits (Expecting False for Chaos -> inversion)
        r_sali_passed = ~r_summary.sali_check.flatten().astype(bool)
        r_gali_passed = ~r_summary.gali_check.flatten().astype(bool)

        r_total = len(regular_ics)
        r_sali_hits = int(np.sum(r_sali_passed))
        r_gali_hits = int(np.sum(r_gali_passed))
        r_sali_miss_indices = [i for i in range(r_total) if not r_sali_passed[i]]
        r_gali_miss_indices = [i for i in range(r_total) if not r_gali_passed[i]]

        # Calculate Combined Metrics
        total_combined = c_total + r_total

        # Individual Accuracies
        sali_combined_correct = c_sali_hits + r_sali_hits
        gali_combined_correct = c_gali_hits + r_gali_hits
        sali_combined_acc = sali_combined_correct / total_combined
        gali_combined_acc = gali_combined_correct / total_combined

        # Joint / Consensus Accuracy (Both SALI & GALI Agree)
        joint_c_passed = c_sali_passed & c_gali_passed
        joint_r_passed = r_sali_passed & r_gali_passed

        joint_c_hits = int(np.sum(joint_c_passed))
        joint_r_hits = int(np.sum(joint_r_passed))

        joint_combined_correct = joint_c_hits + joint_r_hits
        joint_combined_acc = joint_combined_correct / total_combined

        # Build Text Report Lines
        report = [
            "=" * 75,
            f"SENSITIVITY REPORT: {cfg['name']}",
            "=" * 75,
            "CRITERIA & PARAMETERS:",
            f"  • SALI Threshold : {cfg['sali_th']}",
            f"  • GALI Threshold : {cfg['gali_th']}",
            f"  • SALI Window    : {cfg['sali_w']}",
            f"  • GALI Window    : {cfg['gali_w']}",
            "-" * 75,
            "1. CHAOTIC BENCHMARK SUMMARY (Sensitivity / True Positive Rate):",
            f"  • SALI Sensitivity : {c_sali_hits}/{c_total} ({c_sali_hits/c_total:.1%}) | Missed: {len(c_sali_miss_indices)}",
            f"  • GALI Sensitivity : {c_gali_hits}/{c_total} ({c_gali_hits/c_total:.1%}) | Missed: {len(c_gali_miss_indices)}",
            "-" * 75,
            "2. REGULAR BENCHMARK SUMMARY (Specificity / True Negative Rate):",
            f"  • SALI Specificity : {r_sali_hits}/{r_total} ({r_sali_hits/r_total:.1%}) | False Positives: {len(r_sali_miss_indices)}",
            f"  • GALI Specificity : {r_gali_hits}/{r_total} ({r_gali_hits/r_total:.1%}) | False Positives: {len(r_gali_miss_indices)}",
            "-" * 75,
            "3. OVERALL COMBINED ACCURACY (Chaotic + Regular):",
            f"  • SALI Overall Accuracy    : {sali_combined_correct}/{total_combined} ({sali_combined_acc:.1%}) | Total Errors: {total_combined - sali_combined_correct}",
            f"  • GALI Overall Accuracy    : {gali_combined_correct}/{total_combined} ({gali_combined_acc:.1%}) | Total Errors: {total_combined - gali_combined_correct}",
            f"  • JOINT (SALI & GALI) Acc  : {joint_combined_correct}/{total_combined} ({joint_combined_acc:.1%}) | Total Errors: {total_combined - joint_combined_correct}",
            "-" * 75,
            "4. CONFUSION MATRICES / TRUTH TABLES:",
            "",
            "  [ SALI DETECTOR ]",
            f"  Actual Chaotic  | TP (Hits): {c_sali_hits:<3} ({c_sali_hits/c_total:>6.1%})  | FN (Misses): {c_total - c_sali_hits:<3} ({(c_total - c_sali_hits)/c_total:>6.1%})",
            f"  Actual Regular  | FP (False): {r_total - r_sali_hits:<2} ({(r_total - r_sali_hits)/r_total:>6.1%})  | TN (Hits):   {r_sali_hits:<3} ({r_sali_hits/r_total:>6.1%})",
            "",
            "  [ GALI DETECTOR ]",
            f"  Actual Chaotic  | TP (Hits): {c_gali_hits:<3} ({c_gali_hits/c_total:>6.1%})  | FN (Misses): {c_total - c_gali_hits:<3} ({(c_total - c_gali_hits)/c_total:>6.1%})",
            f"  Actual Regular  | FP (False): {r_total - r_gali_hits:<2} ({(r_total - r_gali_hits)/r_total:>6.1%})  | TN (Hits):   {r_sali_hits:<3} ({r_sali_hits/r_total:>6.1%})",
            "-" * 75,
            "5. MISSED INDEXES SUMMARY:",
            f"  • SALI_chaotic_misses : {c_sali_miss_indices}",
            f"  • GALI_chaotic_misses : {c_gali_miss_indices}",
            f"  • SALI_regular_misses : {r_sali_miss_indices}",
            f"  • GALI_regular_misses : {r_gali_miss_indices}",
            "=" * 75,
            "MISSED ORBITS DETAILS:",
        ]

        # Append Detailed Missed Orbits at the very end
        report.extend(
            format_missed_orbits_section(
                "Chaotic Orbits Missed as Regular",
                c_sali_passed,
                c_gali_passed,
                chaotic_mles,
                chaotic_detector,
            )
        )
        report.extend(
            format_missed_orbits_section(
                "Regular Orbits Flagged as Chaotic",
                r_sali_passed,
                r_gali_passed,
                regular_mles,
                regular_detector,
            )
        )

        # Write file
        report_path = OUTPUT_DIR / cfg["name"]
        report_path.write_text("\n".join(report), encoding="utf-8")
        print(f"Report saved to: {report_path}")

    print("\nAll sensitivity reports successfully generated!")
    print(
        "\nRun 'analyze_sensitivity_reports.py' to generate finalised analysis report..."
    )


if __name__ == "__main__":
    main()
