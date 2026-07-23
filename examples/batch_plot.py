import numpy as np
import agama


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocd_gd.orbit_detector import OrbitChaosDetector
from ocd_gd.visualisation import set_publication_style, set_output_dir

set_publication_style()
set_output_dir("./outputs/")

# 1. Setup AGAMA Potential (e.g., standard logarithmic potential)
potential = agama.Potential("data/potentials/MWPotentialHunter24_full.ini")

# 2. Generate initial conditions for 15 orbits in phase space
# Shape: (15, 6) -> [x, y, z, vx, vy, vz]
np.random.seed(42)
num_orbits = 15
ic_batch = np.zeros((num_orbits, 6))
ic_batch[:, 0] = np.linspace(2.0, 10.0, num_orbits)  # Vary X position
ic_batch[:, 4] = 150.0  # Initial Vy velocity

# 3. Instantiate Detector and run batch integrations
detector = OrbitChaosDetector(ic=ic_batch, pot=potential, iter_time=10)

# 4. Plot SALI Batch Grid (Max 10 per page -> Creates 2 pages)
# Page 1 will contain Orbits #0-#9, Page 2 will contain Orbits #10-#14
# sali_figs = detector.plot_sali_batch(
#     max_per_page=10,
#     save_path="batch_sali.png",  # Saves as 'batch_sali_page1.png' & 'batch_sali_page2.png'
#     show=False,
# )

# # 5. Plot GALI Batch Grid for selected orbits only (e.g., specific indices)
# selected_indices = [0, 2, 5, 8, 12, 14]
# gali_figs = detector.plot_gali_batch(
#     orbit_indices=selected_indices,
#     k_orders=[2, 3],
#     max_per_page=6,
#     save_path="batch_gali_selected.png",
#     show=False,
# )

figures = detector.plot_sali_gali_batch(
    orbit_indices=[0, 1, 2, 3, 4, 5, 6, 7],
    k_orders=[2, 3],
    max_orbits_per_page=4,
    save_path="dual_chaos_comparison.png",
    show=True,
)

# detector.plot_dashboard(
#     orbit_idx=5,
#     backend="matplotlib",
#     show=True,
# )
