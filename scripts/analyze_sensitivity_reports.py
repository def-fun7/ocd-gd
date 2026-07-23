import re
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional

REPORTS_DIR = Path("./outputs/sensitivity_reports")
OUTPUT_ANALYSIS_FILE = Path(
    "./outputs/sensitivity_reports/FINAL_SENSITIVITY_ANALYSIS.txt"
)

EXPECTED_RUNS = [
    "ST_2_GT_16_report.txt",
    "ST_3_GT_18_report.txt",
    "ST_3_GT_20_report.txt",
    "ST_4_GT_20_report.txt",
    "SW_5_GW_10_report.txt",
    "SW_10_GW_100_report.txt",
    "SW_20_GW_100_report.txt",
    "SW_20_GW_200_report.txt",
    "SW_20_GW_500_report.txt",
    "ST_4_SW_20_GT_20_report.txt",
]

# Orbits missed in more than this many runs are flagged as "persistent" failures.
PERSISTENT_MISS_THRESHOLD = 8


def parse_report_file(filepath: Path) -> Dict[str, Any]:
    text = filepath.read_text(encoding="utf-8")
    run_tag = filepath.name.replace("_report.txt", "")

    def search_val(pattern: str, cast_type=float):
        match = re.search(pattern, text)
        return cast_type(match.group(1)) if match else None

    def search_list(pattern: str) -> List[int]:
        match = re.search(pattern, text)
        return ast.literal_eval(match.group(1)) if match else []

    parsed = {
        "tag": run_tag,
        "sali_th": search_val(r"SALI Threshold\s*:\s*([\d\.e\-]+)"),
        "gali_th": search_val(r"GALI Threshold\s*:\s*([\d\.e\-]+)"),
        "sali_w": search_val(r"SALI Window\s*:\s*(\d+)", int),
        "gali_w": search_val(r"GALI Window\s*:\s*(\d+)", int),
        # Sensitivity (Chaotic) & Specificity (Regular)
        "c_sali_sens": search_val(r"SALI Sensitivity\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"),
        "c_gali_sens": search_val(r"GALI Sensitivity\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"),
        "r_sali_spec": search_val(r"SALI Specificity\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"),
        "r_gali_spec": search_val(r"GALI Specificity\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"),
        # Overall Accuracies
        "sali_acc": search_val(r"SALI Overall Accuracy\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"),
        "gali_acc": search_val(r"GALI Overall Accuracy\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"),
        "joint_acc": search_val(
            r"JOINT \(SALI & GALI\) Acc\s*:\s*\d+/\d+\s*\(([\d\.]+)%\)"
        ),
        # Miss Lists
        "c_sali_misses": search_list(r"SALI_chaotic_misses\s*:\s*(\[.*?\])"),
        "gali_misses": search_list(r"GALI_chaotic_misses\s*:\s*(\[.*?\])"),
        "r_sali_misses": search_list(r"SALI_regular_misses\s*:\s*(\[.*?\])"),
        "r_gali_misses": search_list(r"GALI_regular_misses\s*:\s*(\[.*?\])"),
    }

    # Warn loudly on missing fields instead of failing silently downstream.
    required_numeric = [
        "c_sali_sens",
        "c_gali_sens",
        "r_sali_spec",
        "r_gali_spec",
        "sali_acc",
        "gali_acc",
        "joint_acc",
    ]
    missing = [k for k in required_numeric if parsed[k] is None]
    if missing:
        print(
            f"  [WARNING] {filepath.name}: could not parse fields {missing} "
            f"— this run will be excluded from ranking to avoid crashes."
        )

    return parsed


def track_orbit_failure_frequencies(parsed_runs: List[Dict[str, Any]]):
    orbit_stats = {
        "chaotic_sali": defaultdict(list),
        "chaotic_gali": defaultdict(list),
        "regular_sali": defaultdict(list),
        "regular_gali": defaultdict(list),
    }

    for run in parsed_runs:
        tag = run["tag"]
        for idx in run["c_sali_misses"]:
            orbit_stats["chaotic_sali"][idx].append(tag)
        for idx in run["gali_misses"]:
            orbit_stats["chaotic_gali"][idx].append(tag)
        for idx in run["r_sali_misses"]:
            orbit_stats["regular_sali"][idx].append(tag)
        for idx in run["r_gali_misses"]:
            orbit_stats["regular_gali"][idx].append(tag)

    return orbit_stats


def format_frequency_table(
    category_name: str, stats_dict: Dict[int, List[str]]
) -> List[str]:
    lines = [f"\n--- {category_name} (Most Missed -> Least Missed) ---"]
    if not stats_dict:
        lines.append("  None! Perfect detection across all runs.")
        return lines

    sorted_orbits = sorted(
        stats_dict.items(), key=lambda item: len(item[1]), reverse=True
    )
    lines.append(f"  {'Orbit ID':<10} | {'Miss Count':<10} | {'Failed in Runs'}")
    lines.append("  " + "-" * 70)

    for orbit_id, failed_runs in sorted_orbits:
        lines.append(
            f"  Orbit #{orbit_id:<4} | {len(failed_runs)} / {len(EXPECTED_RUNS)} runs | [{', '.join(failed_runs)}]"
        )

    return lines


def build_metric_table(
    rows: List[Dict[str, Any]],
    col1_label: str,
    col1_key: str,
    col2_label: str,
    col2_key: str,
) -> List[str]:
    """Shared table-builder for both the threshold and window sections."""
    lines = [
        "-" * 105,
        f"{'Run Tag':<14} | {col1_label:<8} | {col2_label:<8} | {'Ch.SALI%':<8} | {'Ch.GALI%':<8} | "
        f"{'Reg.SALI%':<9} | {'Reg.GALI%':<9} | {'SALI Acc%':<9} | {'GALI Acc%'}",
        "-" * 105,
    ]
    for r in rows:
        lines.append(
            f"{r['tag']:<14} | {r[col1_key]:<8} | {r[col2_key]:<8} | "
            f"{r['c_sali_sens']:>7.1f}% | {r['c_gali_sens']:>7.1f}% | "
            f"{r['r_sali_spec']:>8.1f}% | {r['r_gali_spec']:>8.1f}% | "
            f"{r['sali_acc']:>8.1f}% | {r['gali_acc']:>8.1f}%"
        )
    return lines


def balanced_sali(run: Dict[str, Any]) -> float:
    """Worst of chaos-recall / regular-specificity for SALI — penalizes lopsided runs."""
    return min(run["c_sali_sens"], run["r_sali_spec"])


def balanced_gali(run: Dict[str, Any]) -> float:
    """Worst of chaos-recall / regular-specificity for GALI — penalizes lopsided runs."""
    return min(run["c_gali_sens"], run["r_gali_spec"])


def balanced_joint(run: Dict[str, Any]) -> float:
    """Worst of all four sub-metrics — the run with no weak spot in either indicator or direction."""
    return min(
        run["c_sali_sens"],
        run["r_sali_spec"],
        run["c_gali_sens"],
        run["r_gali_spec"],
    )


def safe_max(rows: List[Dict[str, Any]], key) -> Optional[Dict[str, Any]]:
    return max(rows, key=key) if rows else None


def main():
    all_parsed = [
        parse_report_file(REPORTS_DIR / f)
        for f in EXPECTED_RUNS
        if (REPORTS_DIR / f).exists()
    ]

    missing_files = [f for f in EXPECTED_RUNS if not (REPORTS_DIR / f).exists()]
    if missing_files:
        print(f"  [WARNING] Missing report files, skipped: {missing_files}")

    if not all_parsed:
        print("  [ERROR] No report files found or parsed — nothing to analyze.")
        OUTPUT_ANALYSIS_FILE.write_text(
            "No report files found or parsed. Check REPORTS_DIR and EXPECTED_RUNS.",
            encoding="utf-8",
        )
        return

    # Exclude runs with unparsed required fields from ranking (but keep them
    # for the raw tables below is not attempted here since printing also uses
    # these fields — safest is to drop incomplete runs entirely).
    required_numeric = [
        "c_sali_sens",
        "c_gali_sens",
        "r_sali_spec",
        "r_gali_spec",
        "sali_acc",
        "gali_acc",
        "joint_acc",
    ]
    parsed_runs = [
        r for r in all_parsed if all(r[k] is not None for k in required_numeric)
    ]
    dropped = [r["tag"] for r in all_parsed if r not in parsed_runs]
    if dropped:
        print(f"  [WARNING] Excluding incomplete runs from analysis: {dropped}")

    if not parsed_runs:
        print("  [ERROR] All runs had missing/unparseable fields — nothing to rank.")
        OUTPUT_ANALYSIS_FILE.write_text(
            "All parsed runs were missing required fields. Check report formats.",
            encoding="utf-8",
        )
        return

    threshold_runs = [r for r in parsed_runs if r["tag"].startswith("ST_")]
    window_runs = [r for r in parsed_runs if r["tag"].startswith("SW_")]
    freq_stats = track_orbit_failure_frequencies(parsed_runs)

    # --- Blended-accuracy "best" (original metric, kept for reference) ---
    best_overall_gali = safe_max(parsed_runs, key=lambda x: x["gali_acc"])
    best_overall_sali = safe_max(parsed_runs, key=lambda x: x["sali_acc"])
    best_overall_joint = safe_max(parsed_runs, key=lambda x: x["joint_acc"])

    best_threshold_run = safe_max(threshold_runs, key=lambda x: x["gali_acc"])
    best_window_run = safe_max(window_runs, key=lambda x: x["gali_acc"])

    # --- Balanced (Ch & Reg, both indicators) "best" — the real recommendation ---
    best_sali_balanced = safe_max(parsed_runs, key=balanced_sali)
    best_gali_balanced = safe_max(parsed_runs, key=balanced_gali)
    best_joint_balanced = safe_max(parsed_runs, key=balanced_joint)

    best_threshold_balanced = safe_max(threshold_runs, key=balanced_joint)
    best_window_balanced = safe_max(window_runs, key=balanced_joint)

    # Get orbits missed > PERSISTENT_MISS_THRESHOLD times across all runs
    persistent_misses = {
        "SALI_chaotic_persistent": sorted(
            [
                k
                for k, v in freq_stats["chaotic_sali"].items()
                if len(v) > PERSISTENT_MISS_THRESHOLD
            ]
        ),
        "GALI_chaotic_persistent": sorted(
            [
                k
                for k, v in freq_stats["chaotic_gali"].items()
                if len(v) > PERSISTENT_MISS_THRESHOLD
            ]
        ),
        "SALI_regular_persistent": sorted(
            [
                k
                for k, v in freq_stats["regular_sali"].items()
                if len(v) > PERSISTENT_MISS_THRESHOLD
            ]
        ),
        "GALI_regular_persistent": sorted(
            [
                k
                for k, v in freq_stats["regular_gali"].items()
                if len(v) > PERSISTENT_MISS_THRESHOLD
            ]
        ),
    }

    report = [
        "=" * 105,
        "                            COMPREHENSIVE SENSITIVITY & PERFORMANCE META-ANALYSIS",
        "=" * 105,
        "",
        "1. THRESHOLD EFFECTS ANALYSIS (Fixed Windows: SALI_W=10, GALI_W=100):",
    ]
    report.extend(
        build_metric_table(threshold_runs, "SALI Th", "sali_th", "GALI Th", "gali_th")
    )

    report.extend(
        [
            "",
            "2. WINDOW SIZE EFFECTS ANALYSIS (Fixed Thresholds: S3 [1e-3], G20 [1e-20]):",
        ]
    )
    report.extend(
        build_metric_table(window_runs, "SALI Win", "sali_w", "GALI Win", "gali_w")
    )

    report.extend(
        [
            "",
            "=" * 105,
            "3. BEST PERFORMING CONFIGURATIONS (blended accuracy — for reference only):",
            "=" * 105,
        ]
    )
    if best_threshold_run:
        report.append(
            f"  • Best Threshold Pair  : {best_threshold_run['tag']} "
            f"(GALI Acc: {best_threshold_run['gali_acc']}%, Joint Acc: {best_threshold_run['joint_acc']}%)"
        )
    if best_window_run:
        report.append(
            f"  • Best Window Pair     : {best_window_run['tag']} "
            f"(GALI Acc: {best_window_run['gali_acc']}%, Joint Acc: {best_window_run['joint_acc']}%)"
        )
    report.append(
        f"  • Absolute Best GALI   : {best_overall_gali['tag']} ({best_overall_gali['gali_acc']}% Overall Accuracy)"
    )
    report.append(
        f"  • Absolute Best SALI   : {best_overall_sali['tag']} ({best_overall_sali['sali_acc']}% Overall Accuracy)"
    )
    report.append(
        f"  • Absolute Best JOINT  : {best_overall_joint['tag']} ({best_overall_joint['joint_acc']}% Overall Joint Accuracy)"
    )

    report.extend(
        [
            "",
            "=" * 105,
            "3b. BALANCED BEST CONFIGURATIONS (no weak spot in Chaos OR Regular detection):",
            "    Score = worst of {Ch.SALI, Reg.SALI} for SALI; {Ch.GALI, Reg.GALI} for GALI;",
            "    all four for JOINT. A high blended accuracy can still hide a lopsided run",
            "    (e.g. great chaos recall but poor regular specificity) — this section can't.",
            "=" * 105,
        ]
    )
    if best_threshold_balanced:
        report.append(
            f"  • Best Threshold Pair (balanced) : {best_threshold_balanced['tag']} "
            f"(worst sub-metric: {balanced_joint(best_threshold_balanced):.1f}%)"
        )
    if best_window_balanced:
        report.append(
            f"  • Best Window Pair (balanced)    : {best_window_balanced['tag']} "
            f"(worst sub-metric: {balanced_joint(best_window_balanced):.1f}%)"
        )
    report.append(
        f"  • Best SALI (balanced)           : {best_sali_balanced['tag']} "
        f"(min(Ch.SALI, Reg.SALI) = {balanced_sali(best_sali_balanced):.1f}%)"
    )
    report.append(
        f"  • Best GALI (balanced)           : {best_gali_balanced['tag']} "
        f"(min(Ch.GALI, Reg.GALI) = {balanced_gali(best_gali_balanced):.1f}%)"
    )
    report.append(
        f"  • Best JOINT (balanced, overall)  : {best_joint_balanced['tag']} "
        f"(min of all 4 sub-metrics = {balanced_joint(best_joint_balanced):.1f}%)"
    )

    report.extend(
        [
            "",
            "=" * 105,
            f"4. PERSISTENT MISSED INDEXES SUMMARY (MISSED MORE THAN {PERSISTENT_MISS_THRESHOLD} "
            f"TIMES OUT OF {len(EXPECTED_RUNS)} RUNS):",
            "=" * 105,
            f"  • SALI_chaotic_persistent_misses (>{PERSISTENT_MISS_THRESHOLD} runs) : {persistent_misses['SALI_chaotic_persistent']}",
            f"  • GALI_chaotic_persistent_misses (>{PERSISTENT_MISS_THRESHOLD} runs) : {persistent_misses['GALI_chaotic_persistent']}",
            f"  • SALI_regular_persistent_misses (>{PERSISTENT_MISS_THRESHOLD} runs) : {persistent_misses['SALI_regular_persistent']}",
            f"  • GALI_regular_persistent_misses (>{PERSISTENT_MISS_THRESHOLD} runs) : {persistent_misses['GALI_regular_persistent']}",
            "",
            "=" * 105,
            "5. FULL MISSED ORBITS BREAKDOWN (FREQUENCY & RUN TAGS):",
            "=" * 105,
        ]
    )

    report.extend(
        format_frequency_table(
            "Chaotic Orbits Missed by GALI", freq_stats["chaotic_gali"]
        )
    )
    report.extend(
        format_frequency_table(
            "Chaotic Orbits Missed by SALI", freq_stats["chaotic_sali"]
        )
    )
    report.extend(
        format_frequency_table(
            "Regular Orbits Missed by GALI (False Positives)",
            freq_stats["regular_gali"],
        )
    )
    report.extend(
        format_frequency_table(
            "Regular Orbits Missed by SALI (False Positives)",
            freq_stats["regular_sali"],
        )
    )

    report.append("\n" + "=" * 105)

    OUTPUT_ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {OUTPUT_ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
