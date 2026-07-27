import sys
from pathlib import Path
import agama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocd_gd.orbit_detector import OrbitChaosDetector

ic = [0, 0, 0, 0, 0, 0]
potential = agama.Potential(type="Plummer", mass=1e11, scaleRadius=5.0)
ocd = OrbitChaosDetector([ic, ic], potential)
print(ocd.gali_array.shape)
