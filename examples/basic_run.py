import numpy as np
import agama

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocd_gd.orbit_detector import OrbitChaosDetector

data = np.load("data/initial_conditions/chaotic_ics_benchmark.npz")
ic = [
    -3.524073026703971,
    -0.3075483741544035,
    0.6725163026327883,
    80.8593413219783,
    61.60518653400349,
    -80.74656028726356,
]
pot = agama.Potential("data/potentials/MWPotentialHunter24_full.ini")
TIME_SPAN = 10.0  # Integration time (needs enough time for MLE convergence)
NUM_STEPS = 1000  # Integration steps

for i in range(500):
    ic = data["ics"][i]

    ocd = OrbitChaosDetector(ic=ic, pot=pot)
    x, y, lyap_arr = agama.orbit(
        ic=ic,
        potential=pot,
        time=TIME_SPAN,
        trajSize=NUM_STEPS,
        lyapunov=True,
        separateTime=True,
    )

    sum = ocd.detect_chaos()
    lyap = ocd.lyapunov_exponents

    if data["mles"][0] != lyap[0][0]:
        print(data["mles"][0], lyap[0][0])
