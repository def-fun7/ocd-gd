#!/usr/bin/env python
"""
Summarize results from a GridChaosDetector sweep table (.ecsv).

Reads the `.ecsv` table produced by `run_sweep.py` and prints 5 side-by-side
comparison tables comparing runs across columns (Run 0, Run 1, Run 2, etc.):
    1. Criteria (thresholds, window sizes, accuracy, grid size, etc.)
    2. Parameters (R_0, y_0, mass fractions, potential sources, etc.)
    3. Regular Orbit Summary (regular counts per indicator & consensus)
    4. Chaotic Orbit Summary (chaotic counts, fractions & consensus)
    5. Comparison Summary (indicator agreement pairs & wall clock performance)

Usage:
    python summarize_sweep.py
    python summarize_sweep.py --input outputs/sweep_results.ecsv
    python summarize_sweep.py --runs 0 1 2 3
    python summarize_sweep.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import QTable

from ocd_gd import (
    get_logger,
    print_banner,
    print_dataframe_table,
    setup_logging,
)

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent / "outputs"


def load_sweep(path: str | Path) -> QTable:
    """Load a sweep table written by `run_sweep.run_sweep`."""
    table = QTable.read(path, format="ascii.ecsv")
    log.info("Loaded %d-row sweep table from %s", len(table), path)
    return table


def _fmt(val: Any, spec: str = "") -> str:
    """Format table cell values cleanly, handling numpy/astropy types."""
    if hasattr(val, "value"):
        val = val.value
    if isinstance(val, (float, np.floating)):
        if spec:
            return f"{val:{spec}}"
        # Use clean scientific notation for masses/large energies
        if abs(val) >= 1e5 or (0 < abs(val) < 1e-3):
            return f"{val:.2e}"
        return f"{val:.4g}"
    if isinstance(val, (str, Path)):
        s = str(val)
        # Truncate long system paths to just the file name
        if "/" in s or "\\" in s:
            return Path(s).name
    return str(val)


def print_sweep_summaries(table: QTable, target_runs: list[int] | None = None) -> None:
    """Build and print side-by-side comparison tables across sweep runs."""
    if "run_index" in table.colnames and target_runs:
        mask = np.isin(table["run_index"], target_runs)
        runs = table[mask]
    else:
        runs = table

    if len(runs) == 0:
        log.warning("No runs found matching specified criteria.")
        return

    run_ids = [
        str(row["run_index"]) if "run_index" in row.colnames else str(i)
        for i, row in enumerate(runs)
    ]

    headers = ["Metric / Parameter"] + [f"Run {r}" for r in run_ids]

    def _render(title: str, row_dict: dict[str, list[str]]) -> None:
        rows = [[metric] + vals for metric, vals in row_dict.items()]
        print_dataframe_table(title=title, headers=headers, rows=rows)

    print_banner("GridChaosDetector Sweep Summary", f"Comparing {len(runs)} Run(s)")

    # 1. Criteria Summary
    criteria_rows: dict[str, list[str]] = {
        "Grid Size": [f"{_fmt(r['grid_size'])}x{_fmt(r['grid_size'])}" for r in runs],
        "Total Orbits": [_fmt(r["num_orbits"]) for r in runs],
        "Integration Time": [_fmt(r["iter_time"]) for r in runs],
        "Accuracy (atol/rtol)": [_fmt(r["accuracy"]) for r in runs],
        "Max Steps": [_fmt(r["max_num_steps"]) for r in runs],
        "SALI Threshold": [_fmt(r["sali_threshold"]) for r in runs],
        "SALI Window Size": [_fmt(r["sali_window_size"]) for r in runs],
        "GALI Threshold": [_fmt(r["gali_threshold"]) for r in runs],
        "GALI Window Size": [_fmt(r["gali_window_size"]) for r in runs],
    }
    _render("1. Detection & Integration Criteria", criteria_rows)

    # 2. Parameters Summary
    param_rows: dict[str, list[str]] = {
        "R_0 (Initial Radius)": [_fmt(r["R_0"]) for r in runs],
        "E_0 (Energy)": [_fmt(r["E_0"], ".2e") for r in runs],
        "y_0": [_fmt(r["y_0"]) for r in runs],
        "z_0": [_fmt(r["z_0"]) for r in runs],
        "v_y0 Fraction": [_fmt(r["v_y0_frac"]) for r in runs],
        "v_z0 Fraction": [_fmt(r["v_z0_frac"]) for r in runs],
        "Pattern Speed (omega)": [_fmt(r["omega"], ".2f") for r in runs],
        "Bar Mass Fraction (Qb)": [_fmt(r["Qb"]) for r in runs],
        "BH Mass Fraction (f_bh)": [_fmt(r["f_bh"]) for r in runs],
        "Disk Mass (M_disk)": [_fmt(r["M_disk"], ".2e") for r in runs],
        "Bar Mass (M_bar)": [_fmt(r["M_bar"], ".2e") for r in runs],
        "BH Mass (M_bh)": [_fmt(r["M_bh"], ".2e") for r in runs],
        "Total Mass (M_total)": [_fmt(r["M_total"], ".2e") for r in runs],
        "Potential Source": [Path(str(r["potential_source"])).name for r in runs],
    }
    _render("2. Physical & Model Parameters", param_rows)

    # 3. Regular Orbit Summary
    reg_rows: dict[str, list[str]] = {
        "SALI Regular Orbits": [_fmt(r["sali_n_regular"]) for r in runs],
        "GALI Regular Orbits": [_fmt(r["gali_n_regular"]) for r in runs],
        "Lyapunov Regular Orbits": [_fmt(r["lyapunov_n_regular"]) for r in runs],
        "All-Agree Regular Orbits": [_fmt(r["all_agree_regular"]) for r in runs],
    }
    _render("3. Regular Orbit Summary", reg_rows)

    # 4. Chaotic Orbit Summary
    ch_rows: dict[str, list[str]] = {
        "SALI Chaotic Count": [_fmt(r["sali_n_chaotic"]) for r in runs],
        "SALI Chaotic Fraction": [
            f"{float(r['sali_chaotic_fraction']) * 100:.2f}%" for r in runs
        ],
        "GALI Chaotic Count": [_fmt(r["gali_n_chaotic"]) for r in runs],
        "GALI Chaotic Fraction": [
            f"{float(r['gali_chaotic_fraction']) * 100:.2f}%" for r in runs
        ],
        "Lyapunov Chaotic Count": [_fmt(r["lyapunov_n_chaotic"]) for r in runs],
        "Lyapunov Chaotic Fraction": [
            f"{float(r['lyapunov_chaotic_fraction']) * 100:.2f}%" for r in runs
        ],
        "All-Agree Chaotic Orbits": [_fmt(r["all_agree_chaotic"]) for r in runs],
    }
    _render("4. Chaotic Orbit Summary", ch_rows)

    # 5. Comparison & Performance Summary
    comp_rows: dict[str, list[str]] = {
        "SALI vs GALI Agreement": [_fmt(r["sali_gali_agreement"]) for r in runs],
        "SALI vs Lyapunov Agreement": [
            _fmt(r["sali_lyapunov_agreement"]) for r in runs
        ],
        "GALI vs Lyapunov Agreement": [
            _fmt(r["gali_lyapunov_agreement"]) for r in runs
        ],
        "Total Disagreements": [_fmt(r["disagreement"]) for r in runs],
        "Total Undertermined": [_fmt(r["n_undetermined"]) for r in runs],
        "Wall Clock Time (s)": [f"{float(r['wall_time_s']):.2f} s" for r in runs],
        "Wall Clock Time (min)": [
            f"{float(r['wall_time_s']) / 60:.2f} min" for r in runs
        ],
    }
    _render("5. Indicator Comparison & Performance Summary", comp_rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print side-by-side comparison tables across runs from a GridChaosDetector sweep table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "sweep_results.ecsv",
        help="Sweep .ecsv file to read (default: outputs/sweep_results.ecsv).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Filter specific run_indices to compare (e.g. --runs 0 1 2). Default: compare all runs.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    setup_logging()

    table = load_sweep(args.input)

    print_sweep_summaries(table, target_runs=args.runs)
