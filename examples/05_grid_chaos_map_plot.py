"""
05_grid_chaos_map_plot.py

Demonstrates visualizing spatial chaos maps using GridChaosDetector's
plot_chaos_map(), plot_composite_chaos_map(), and save_chaos_maps() methods.
"""

import sys
from pathlib import Path
import logging
import agama

from ocd_gd.grid_detector import GridChaosDetector
from ocd_gd._logging_config import setup_logging
from ocd_gd._logging_config import print_banner, print_kv_table

BASE_OUTPUT_PATH = Path(__file__).parent / "outputs"


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

    print_banner("ocd-gd", "Grid Chaos Map Visualization Example")

    # 2. Construct Potential & Grid Detector
    pot = build_potential()
    grid_size = 12

    logging.info(f"Initializing GridChaosDetector ({grid_size}x{grid_size})...")

    detector = GridChaosDetector(
        potential=pot,
        R_0=8.0,
        y_0=1e-4,
        z_0=0.1,
        v_y0_frac=0.2,
        v_z0_frac=0.02,
        grid_size=grid_size,
        x_search_range=(-10.0, 10.0),
        omega=0.015,  # Rotating frame pattern speed for resonance overlays
        iter_time=10.0,
        plotting_backend="matplotlib",
    )

    s_by_s_file = str(BASE_OUTPUT_PATH / "grid_side_by_side_chaos_map.png")
    comp_file = str(BASE_OUTPUT_PATH / "grid_composite_chaos_map.png")

    # 3. Generate side-by-side & composite maps simultaneously without interactive popups
    detector.save_chaos_maps(
        side_by_side_path=s_by_s_file,
        composite_path=comp_file,
        show_resonances=True,
    )

    # ------------------------------------------------------------------
    # 4. Presentation Phase
    # ------------------------------------------------------------------

    # Output Summary Table
    print_kv_table(
        title="Generated Chaos Map Plots",
        data={
            "Grid Resolution": f"{grid_size} x {grid_size}",
            "Pattern Speed (omega)": f"{detector.omega}",
            "Resonance Radii Found": (
                f"ILR={detector.resonance_radii.ilr:.2f}, CR={detector.resonance_radii.corotation:.2f}"
                if detector.resonance_radii and detector.resonance_radii.corotation
                else "None / Axisymmetric"
            ),
            "Side-by-Side Map File": s_by_s_file,
            "Composite Map File": comp_file,
        },
        header_style="bold magenta",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\u2717 Example failed: {exc}", file=sys.stderr)
        sys.exit(1)
