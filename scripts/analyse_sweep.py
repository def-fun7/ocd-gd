#!/usr/bin/env python
"""
Analyze / plot results from a GridChaosDetector sweep (run_sweep.py).

Deliberately a separate script from run_sweep.py: this only ever reads the
`.ecsv` table run_sweep.py already wrote, so re-fitting or re-plotting
(different x/y columns, a different grouping, a different fit degree)
never means re-running the (expensive) orbit integrations -- just
re-running this script against the same file.

Usage:
    python analyze_sweep.py
    python analyze_sweep.py --input outputs/sweep_results.ecsv \
        --x f_bh --y sali_chaotic_fraction --group-by Qb
    python analyze_sweep.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

from ocd_gd import (
    get_logger,
    print_banner,
    print_kv_table,
    setup_logging,
)

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent / "outputs"


def load_sweep(path: str | Path) -> QTable:
    """Load a sweep table written by `run_sweep.run_sweep` (or its
    incremental checkpoint)."""
    table = QTable.read(path, format="ascii.ecsv")
    log.info("Loaded %d-row sweep table from %s", len(table), path)
    return table


def fit_and_plot(
    table: QTable,
    x_col: str,
    y_col: str,
    group_by: str | None,
    output: Path,
    degree: int = 1,
) -> dict[Any, tuple[np.ndarray, float]]:
    """Scatter `y_col` vs `x_col`, one series per unique value of
    `group_by` (or a single series if `group_by` is None), with a
    degree-`degree` polynomial fit overlaid per group.

    Returns `{group_value: (poly_coeffs, r_squared)}` per group that had
    enough finite points to fit (`poly_coeffs` in `numpy.polyfit`
    convention: highest-degree coefficient first).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fits: dict[Any, tuple[np.ndarray, float]] = {}

    groups = [None] if group_by is None else sorted(set(table[group_by]))
    for group in groups:
        sub = table if group is None else table[table[group_by] == group]
        x = np.asarray(sub[x_col], dtype=float)
        y = np.asarray(sub[y_col], dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]

        label = f"{group_by}={group:g}" if group is not None else y_col
        ax.scatter(x, y, label=label, zorder=3)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= degree + 1:
            coeffs = np.polyfit(x[finite], y[finite], deg=degree)
            fitted = np.polyval(coeffs, x[finite])
            ss_res = np.sum((y[finite] - fitted) ** 2)
            ss_tot = np.sum((y[finite] - np.mean(y[finite])) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fits[group] = (coeffs, r_squared)

            x_line = np.linspace(x[finite].min(), x[finite].max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), linestyle="--", alpha=0.7)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    title = f"{y_col} vs {x_col}"
    if group_by:
        title += f", grouped by {group_by}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    log.info("Saved plot to %s", output)
    return fits


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and plot a GridChaosDetector sweep table "
        "(e.g. chaotic fraction vs. a build parameter)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "sweep_results.ecsv",
        help="Sweep .ecsv to read (default: scripts/outputs/sweep_results.ecsv).",
    )
    parser.add_argument(
        "--x", default="f_bh", help="Column for the x-axis (default: f_bh)."
    )
    parser.add_argument(
        "--y",
        default="sali_chaotic_fraction",
        help="Column for the y-axis (default: sali_chaotic_fraction).",
    )
    parser.add_argument(
        "--group-by",
        default="Qb",
        help="Column to group/color series by, or 'none' (default: Qb).",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Polynomial fit degree (default: 1, linear).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the plot (default: outputs/<y>_vs_<x>.png).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    setup_logging()
    print_banner("GridChaosDetector sweep analysis", f"{args.y} vs {args.x}")

    table = load_sweep(args.input)
    group_by = None if args.group_by.lower() == "none" else args.group_by
    output = args.output or BASE_DIR / f"{args.y}_vs_{args.x}.png"

    fits = fit_and_plot(table, args.x, args.y, group_by, output, degree=args.degree)

    fit_summary: dict[str, str] = {}
    for group, (coeffs, r_squared) in fits.items():
        label = f"{group_by}={group:g}" if group is not None else args.y
        # index -2 is always the degree-1 (linear) term's coefficient,
        # regardless of the overall fit degree (polyfit orders highest-first).
        slope = coeffs[-2] if len(coeffs) >= 2 else float("nan")
        fit_summary[label] = f"slope={slope:.4g}, R^2={r_squared:.3f}"

    if fit_summary:
        print_kv_table(
            title=f"Fit: {args.y} ~ poly(deg={args.degree})({args.x})",
            data=fit_summary,
        )
