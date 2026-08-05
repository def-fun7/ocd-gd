"""
Shared argparse pieces for the ocd_gd scripts (build_potential.py,
run_sweep.py) -- kept in one place so the two scripts' command-line
interfaces stay consistent instead of drifting apart.
"""

from __future__ import annotations

import argparse


def add_qb_fbh_args(parser: argparse.ArgumentParser) -> None:
    """Add --qb/--fbh grid-axis arguments (the two knobs of the composite
    potential model in build_potential.py), with the same defaults as the
    example grid previously hardcoded in each script's __main__."""
    parser.add_argument(
        "--qb",
        type=float,
        nargs="+",
        default=[0.1],
        help="Bar-torque strength Q_b value(s) to sweep over (default: 0.1 0.2).",
    )
    parser.add_argument(
        "--fbh",
        type=float,
        nargs="+",
        default=[0.1],
        help="Central-mass fraction f_bh value(s) to sweep over (default: 0.0 0.005).",
    )


def add_clear_cache_arg(parser: argparse.ArgumentParser) -> None:
    """Add a --clear-cache flag. Both scripts share the same cache
    directory (scripts/outputs/), so this clears the same thing in either
    one: cached bar masses (.masscache) and exported .ini files, forcing
    the next build to recalibrate from scratch."""
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cached .ini/.masscache files under scripts/outputs/ before "
        "running, forcing every potential to be rebuilt/recalibrated from scratch.",
    )
