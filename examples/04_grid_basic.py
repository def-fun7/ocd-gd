"""
04_grid_basic.py

Demonstrates GridChaosDetector grid generation, physical cell filtering,
lazy-loaded chaos_grids arrays, and coordinate/index lookup methods without plotting.
"""

import logging
import sys

import agama
import numpy as np

from ocd_gd._logging_config import (
    print_banner,
    print_dataframe_table,
    print_kv_table,
    setup_logging,
)
from ocd_gd.grid_detector import GridChaosDetector


def build_potential() -> "agama.Potential":
    """Construct a basic Hernquist-like spheroid potential."""
    agama.setUnits(mass=1, length=1, velocity=1)
    return agama.Potential(
        type="Spheroid",
        mass=1e10,
        scaleRadius=5.0,
        gamma=1.0,
        beta=4.0,
    )


def main() -> None:
    # 1. Setup Logging
    setup_logging(level=logging.INFO)

    print_banner("ocd-gd", "Grid Chaos Detector (Basic Lookup & Grids)")

    # 2. Construct Potential & Grid Detector
    pot = build_potential()
    grid_size = 10

    detector = GridChaosDetector(
        potential=pot,
        R_0=8.0,
        y_0=1e-4,
        z_0=0.1,
        v_y0_frac=0.2,
        v_z0_frac=0.02,
        grid_size=grid_size,
        x_search_range=(-10.0, 10.0),
        iter_time=10.0,
    )

    # 3. Access Reshaped Chaos Grids
    sali_grid, _gali_grid, _lyap_grid = detector.chaos_grids

    # Count physical (valid float) vs unphysical (NaN) cells
    n_total = grid_size * grid_size
    n_physical = int(np.count_nonzero(~np.isnan(sali_grid)))
    n_unphysical = n_total - n_physical

    # ------------------------------------------------------------------
    # 4. Presentation Phase
    # ------------------------------------------------------------------

    # Grid Setup Overview
    print_kv_table(
        title="Grid Configuration & Energy State",
        data={
            "Grid Dimensions": f"{detector.grid_size} x {detector.grid_size} ({n_total} cells)",
            "Reference Radius (R_0)": f"{detector.R_0:.2f}",
            "Reference Energy (E_0)": f"{detector.E_0:.6f}",
            "Physical Integrated Cells": f"{n_physical} ({n_physical / n_total:.1%})",
            "Unphysical Skipped Cells": f"{n_unphysical} ({n_unphysical / n_total:.1%})",
            "SALI Grid Shape": str(sali_grid.shape),
        },
        header_style="bold magenta",
    )

    # Demonstrate Index and Coordinate Lookups across physical orbits
    lookup_rows = []
    # Pick sample physical orbit indices from integrated orbits
    sample_orbit_indices = [0, min(2, n_physical - 1), n_physical - 1]

    for orbit_idx in sample_orbit_indices:
        row, col = detector.grid_position_of(orbit_idx)
        x_val, vx_val = detector.grid_coordinates_of(orbit_idx)
        recovered_idx = detector.orbit_idx_at(row, col)
        sali_val = sali_grid[row, col]

        lookup_rows.append(
            [
                str(orbit_idx),
                f"({row}, {col})",
                f"{x_val:.3f}",
                f"{vx_val:.3f}",
                str(recovered_idx),
                f"{sali_val:.2f}" if not np.isnan(sali_val) else "NaN",
            ]
        )

    print_dataframe_table(
        title="Coordinate & Index Mapping Samples",
        headers=[
            "Orbit Index",
            "Grid (row, col)",
            "x",
            "v_x",
            "Recovered Index",
            "SALI Check",
        ],
        rows=lookup_rows,
        header_style="bold green",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\u2717 Example failed: {exc}", file=sys.stderr)
        sys.exit(1)
