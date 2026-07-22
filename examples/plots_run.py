import numpy as np
import agama

# Import your package

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocd_gd.orbit_detector import OrbitChaosDetector
from ocd_gd.visualisation import set_publication_style, set_output_dir

# Optional: Apply nice Matplotlib defaults globally
set_publication_style()
set_output_dir("./outputs/")


def main():
    print("1. Initializing AGAMA potential...")
    # Standard Plummer sphere potential (mass=1, scale radius=1)
    ic = [
        -3.524073026703971,
        -0.3075483741544035,
        0.6725163026327883,
        80.8593413219783,
        61.60518653400349,
        -80.74656028726356,
    ]
    potential = agama.Potential("data/potentials/MWPotentialHunter24_full.ini")
    print("2. Initializing OrbitChaosDetector & running orbit integration...")
    detector = OrbitChaosDetector(
        ic=ic,
        pot=potential,
        plotting_backend="matplotlib",  # Default backend
    )

    print(f"Integration complete! Timestamps shape: {detector.timestamps.shape}")

    # =========================================================================
    # TEST MATPLOTLIB BACKEND
    # =========================================================================
    print("\n--- Testing Matplotlib Backend ---")

    # 1. SALI vs Time
    print("  -> Plotting SALI vs Time (Matplotlib)...")
    detector.plot_sali(
        orbit_idx=0,
        backend="matplotlib",
        save_path="test_sali_mpl.png",
        show=False,
    )

    # 2. GALI vs Time
    print("  -> Plotting GALI vs Time (Matplotlib)...")
    detector.plot_gali(
        orbit_idx=0,
        backend="matplotlib",
        save_path="test_gali_mpl.png",
        show=False,
    )

    # 3. 2D Projections (Face-On & Edge-On)
    print("  -> Plotting 2D Trajectories (Matplotlib)...")
    detector.plot_trajectory_2d(
        orbit_idx=0,
        backend="matplotlib",
        save_path="test_traj2d_mpl.png",
        show=False,
    )

    # 4. 3D Trajectory
    print("  -> Plotting 3D Trajectory (Matplotlib)...")
    detector.plot_trajectory_3d(
        orbit_idx=0,
        backend="matplotlib",
        save_path="test_traj3d_mpl.png",
        show=False,
    )

    # 5. Phase Space (X vs Vx)
    print("  -> Plotting Phase Space (Matplotlib)...")
    detector.plot_phase_space(
        orbit_idx=0,
        plane="x",
        backend="matplotlib",
        save_path="test_phase_mpl.png",
        show=False,
    )

    # 6. Color-coded Trajectory (by SALI)
    print("  -> Plotting Color-coded Trajectory by SALI (Matplotlib)...")
    detector.plot_colored_trajectory(
        orbit_idx=0,
        color_by="sali",
        backend="matplotlib",
        save_path="test_colored_mpl.png",
        show=False,
    )

    # 7. Summary Dashboard
    print("  -> Plotting Summary Dashboard (Matplotlib)...")
    detector.plot_dashboard(
        orbit_idx=0,
        backend="matplotlib",
        save_path="test_dashboard_mpl.png",
        show=False,
    )

    # =========================================================================
    # TEST PLOTLY BACKEND
    # =========================================================================
    print("\n--- Testing Plotly Backend ---")

    # 1. SALI vs Time (HTML Interactive Export)
    print("  -> Plotting SALI vs Time (Plotly HTML)...")
    detector.plot_sali(
        orbit_idx=0,
        backend="plotly",
        save_path="test_sali_plotly.html",
        show=False,
    )

    # 2. 3D Interactive Orbit (HTML Export)
    print("  -> Plotting 3D Trajectory (Plotly HTML)...")
    detector.plot_trajectory_3d(
        orbit_idx=0,
        backend="plotly",
        save_path="test_traj3d_plotly.html",
        show=False,
    )

    print("\nAll test plots generated and saved successfully!")


if __name__ == "__main__":
    main()
