import agama
from pathlib import Path
import numpy as np
import pytest
from ocd_gd.orbit_detector import OrbitChaosDetector

REGULAR_BENCHMARK_NPZ = Path("data/initial_conditions/regular_ics_benchmark_226.npz")
MW_POTENTIAL = "data/potentials/MWPotentialHunter24_full.ini"


def load_regular_benchmark_ics():
    """Helper to load regular initial conditions and MLE ground truth from NPZ."""
    if not REGULAR_BENCHMARK_NPZ.exists():
        pytest.skip(
            f"{REGULAR_BENCHMARK_NPZ} not found, skipping regular benchmark tests."
        )

    data = np.load(REGULAR_BENCHMARK_NPZ)
    return data["ics"], data["mles"]


class TestRegularSensitivity:
    """Benchmark tests evaluating detector performance on verified regular ICs."""

    def test_detector_regular_specificity(
        self, output_dir=Path("./outputs/benchmark_results")
    ):
        """
        Evaluate detector performance on regular orbits
        """
        ics, mles = load_regular_benchmark_ics()
        mw_potential = agama.Potential(MW_POTENTIAL)

        # Run detection on all regular benchmark ICs
        detector = OrbitChaosDetector(
            ic=ics,
            pot=mw_potential,
        )
        summary = detector.detect_chaos(check_only=True)

        # Flatten check arrays (True = Chaos detected, False = Regular detected)
        sali_chaotic = summary.sali_check.flatten().astype(bool)
        gali_chaotic = summary.gali_check.flatten().astype(bool)

        total_orbits = len(ics)

        # Count True Regulars (False in check array) and False Positives (True in check array)
        sali_regular_hits = int(np.sum(~sali_chaotic))
        gali_regular_hits = int(np.sum(~gali_chaotic))

        sali_spec_rate = sali_regular_hits / total_orbits
        gali_spec_rate = gali_regular_hits / total_orbits

        # Optional baseline assertions
        assert (
            sali_spec_rate >= 0.90
        ), f"SALI specificity regression: {sali_spec_rate:.1%}"
        assert (
            gali_spec_rate >= 0.90
        ), f"GALI specificity regression: {gali_spec_rate:.1%}"
