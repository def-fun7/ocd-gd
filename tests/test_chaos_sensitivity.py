import agama
import csv
from pathlib import Path
import numpy as np
import pytest
from ocd_gd.orbit_detector import OrbitChaosDetector

BENCHMARK_NPZ = Path("data/initial_conditions/chaotic_ics_benchmark_105.npz")
MW_POTENTIAL = "data/potentials/MWPotentialHunter24_full.ini"


def load_benchmark_ics():
    """Helper to load initial conditions and MLE ground truth from CSV."""
    if not BENCHMARK_NPZ.exists():
        pytest.skip(f"{BENCHMARK_NPZ} not found, skipping benchmark tests.")

    data = np.load(BENCHMARK_NPZ)
    return data["ics"], data["mles"]


class TestChaosSensitivity:
    """Benchmark tests checking detector performance against verified chaotic ICs."""

    def test_detector_benchmark_sensitivity(self):
        """Evaluate detector sensitivity across the full benchmark set of chaotic orbits."""
        ics, mles = load_benchmark_ics()
        mw_potential = agama.Potential(MW_POTENTIAL)

        # Run detection on all benchmark ICs
        detector = OrbitChaosDetector(ic=ics, pot=mw_potential, iter_time=14)

        summary = detector.detect_chaos(check_only=True)

        # Flatten check arrays to 1D boolean masks
        sali_passed = summary.sali_check.flatten().astype(bool)
        gali_passed = summary.gali_check.flatten().astype(bool)

        total_orbits = len(ics)
        sali_hits = int(np.sum(sali_passed))
        gali_hits = int(np.sum(gali_passed))

        sali_misses = total_orbits - sali_hits
        gali_misses = total_orbits - gali_hits

        # Print summary statistics
        print("\n" + "=" * 50)
        print(f"BENCHMARK CHAOS DETECTION RESULTS ({total_orbits} Total Orbits)")
        print("=" * 50)
        print(
            f"SALI Sensitivity : {sali_hits}/{total_orbits} ({sali_hits/total_orbits:.1%}) | Missed: {sali_misses}"
        )
        print(
            f"GALI Sensitivity : {gali_hits}/{total_orbits} ({gali_hits/total_orbits:.1%}) | Missed: {gali_misses}"
        )
        print("-" * 50)

        # Log specific orbits where SALI or GALI failed to detect chaos
        if sali_misses > 0 or gali_misses > 0:
            print("MISSED ORBITS DETAIL:")
            for idx in range(total_orbits):
                if not (sali_passed[idx] and gali_passed[idx]):
                    print(
                        f"Orbit #{idx:03d} | MLE: {mles[idx]:.5f} | "
                        f"SALI: {'PASS' if sali_passed[idx] else 'MISS (inf)'} | "
                        f"GALI: {'PASS' if gali_passed[idx] else 'MISS (inf)'}"
                    )
        print("=" * 50 + "\n")

        sali_rate = sali_hits / total_orbits
        gali_rate = gali_hits / total_orbits

        # Lock in statistical baselines for 10 Gyr integration
        assert sali_rate >= 0.90, f"SALI sensitivity regression: {sali_rate:.1%}"
        assert gali_rate >= 0.95, f"GALI sensitivity regression: {gali_rate:.1%}"
