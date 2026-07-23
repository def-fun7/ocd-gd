import agama
from pathlib import Path
import numpy as np
import pytest
from ocd_gd.orbit_detector import OrbitChaosDetector

BENCHMARK_NPZ = Path("data/initial_conditions/chaotic_ics_benchmark_185_0.1.npz")
MW_POTENTIAL = "data/potentials/MWPotentialHunter24_full.ini"


def load_benchmark_ics():
    """Helper to load initial conditions and MLE ground truth from NPZ."""
    if not BENCHMARK_NPZ.exists():
        pytest.skip(f"{BENCHMARK_NPZ} not found, skipping benchmark tests.")

    data = np.load(BENCHMARK_NPZ)
    return data["ics"], data["mles"]


class TestChaosSensitivity:
    """Benchmark tests checking detector performance against verified chaotic ICs."""

    def test_detector_benchmark_sensitivity(self):
        """Evaluate detector sensitivity across the full benchmark set of chaotic orbits"""
        ics, mles = load_benchmark_ics()
        mw_potential = agama.Potential(MW_POTENTIAL)

        # Run detection on all benchmark ICs
        detector = OrbitChaosDetector(ic=ics, pot=mw_potential)
        summary = detector.detect_chaos(check_only=True)

        sali_passed = summary.sali_check.flatten().astype(bool)
        gali_passed = summary.gali_check.flatten().astype(bool)

        total_orbits = len(ics)
        sali_rate = np.sum(sali_passed) / total_orbits
        gali_rate = np.sum(gali_passed) / total_orbits

        assert sali_rate >= 0.90, f"SALI sensitivity regression: {sali_rate:.1%}"
        assert gali_rate >= 0.95, f"GALI sensitivity regression: {gali_rate:.1%}"
