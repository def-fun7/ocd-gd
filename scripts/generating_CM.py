import agama

from pathlib import Path
import sys
import numpy as np

# Adjust path to include src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocd_gd.grid_detector import GridChaosDetector

# Set up Agama potential (e.g., standard logarithmic potential)
pot = agama.Potential("./data/potentials/MWPotentialHunter24_full.ini")

r_ref = 8.0
pos_ref = np.array([[r_ref, 10.0, 10.0]])

force = pot.force(pos_ref)[0]
v_circ = np.sqrt(r_ref * np.abs(force[0]))

# Total Energy E_0 = Phi(r) + 0.5 * v_circ^2
E_0 = pot.potential(pos_ref)[0] + 0.5 * v_circ**2

# Instantiate grid detector across E_0 = -0.5
detector = GridChaosDetector(
    potential=pot,
    E_0=E_0,
    grid_size=10,
    y_0=10.0,
    z_0=10.0,
    v_y0=0.6 * v_circ,
    v_z0=0.0,
    plotting_backend="matplotlib",
)

# # Render side-by-side chaos maps (SALI, GALI, Lyapunov)

# # Render composite RGB overlay map
detector.plot_composite_chaos_map(save_path="composite_chaos_map.png")

# detector.plot_dashboard(4)
# detector.plot_chaos_map(save_path="chaos_grid_maps.png")
