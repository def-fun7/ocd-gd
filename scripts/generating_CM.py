import agama

from pathlib import Path
import sys
import numpy as np

# Adjust path to include src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocd_gd.grid_detector import GridChaosDetector

# Set up Agama potential (e.g., standard logarithmic potential)
pot = agama.Potential("./data/potentials/MWPotentialHunter24_full.ini")

detector = GridChaosDetector(
    potential=pot,
    R_0=8,
    z_0=0.0,
    grid_size=10,
    v_z0_frac=0.0,
    plotting_backend="matplotlib",
)


# detector.plot_composite_chaos_map(save_path="composite_chaos_map.png")
# detector.plot_chaos_map()
