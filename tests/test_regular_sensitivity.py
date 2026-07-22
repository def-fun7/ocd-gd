import agama
from pathlib import Path
import numpy as np
import pytest
from ocd_gd.orbit_detector import OrbitChaosDetector

REGULAR_BENCHMARK_NPZ = Path("data/initial_conditions/regular_ics_benchmark_226.npz")
MW_POTENTIAL = "data/potentials/MWPotentialHunter24_full.ini"

PREFIX = "S5_S3_G20"
SALI_THRESHOLD = 1e-3
GALI_THRESHOLD = 1e-20
SALI_WIND_S = 5
GALI_WIND_S = 100
ITER_TIME = 10


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
        Evaluate detector performance on regular orbits and write report to a .txt file.
        """
        ics, mles = load_regular_benchmark_ics()
        mw_potential = agama.Potential(MW_POTENTIAL)

        # Run detection on all regular benchmark ICs
        detector = OrbitChaosDetector(
            ic=ics,
            pot=mw_potential,
            iter_time=ITER_TIME,
            sali_threshold=SALI_THRESHOLD,
            gali_threshold=GALI_THRESHOLD,
            sali_window_size=SALI_WIND_S,
            gali_window_size=GALI_WIND_S,
        )
        summary = detector.detect_chaos(check_only=True)

        # Get stopping & integration criteria from detector property
        criteria = detector.criteria

        # Flatten check arrays (True = Chaos detected, False = Regular detected)
        sali_chaotic = summary.sali_check.flatten().astype(bool)
        gali_chaotic = summary.gali_check.flatten().astype(bool)

        total_orbits = len(ics)

        # Count True Regulars (False in check array) and False Positives (True in check array)
        sali_regular_hits = int(np.sum(~sali_chaotic))
        gali_regular_hits = int(np.sum(~gali_chaotic))

        sali_false_positives = total_orbits - sali_regular_hits
        gali_false_positives = total_orbits - gali_regular_hits

        sali_spec_rate = sali_regular_hits / total_orbits
        gali_spec_rate = gali_regular_hits / total_orbits

        # Prepare summary report content
        report_lines = [
            "=" * 60,
            "BENCHMARK REGULAR ORBIT DETECTOR REPORT",
            "=" * 60,
            f"Dataset NPZ Path     : {REGULAR_BENCHMARK_NPZ}",
            f"Potential File       : {MW_POTENTIAL}",
            f"Total Orbits Tested  : {total_orbits}",
            "-" * 60,
            "DETECTOR INTEGRATION CRITERIA:",
            f"  • iter_time        : {criteria.iter_time}",
            f"  • sali_threshold   : {criteria.sali_threshold}",
            f"  • gali_threshold   : {criteria.gali_threshold}",
            f"  • sali_window_size : {criteria.sali_window_size}",
            f"  • gali_window_size : {criteria.gali_window_size}",
            f"  • accuracy         : {criteria.accuracy}",
            f"  • max_num_steps    : {criteria.max_num_steps}",
            "-" * 60,
            "SUMMARY STATISTICS:",
            f"  • SALI Specificity : {sali_regular_hits}/{total_orbits} ({sali_spec_rate:.1%}) | False Positives: {sali_false_positives}",
            f"  • GALI Specificity : {gali_regular_hits}/{total_orbits} ({gali_spec_rate:.1%}) | False Positives: {gali_false_positives}",
            "=" * 60,
        ]

        # Append false positive details if any exist
        if sali_false_positives > 0 or gali_false_positives > 0:
            report_lines.append(
                "\nFALSE POSITIVE ORBITS DETAIL (Regular ICs Flagged as Chaotic):"
            )
            report_lines.append("-" * 60)
            for idx in range(total_orbits):
                if sali_chaotic[idx] or gali_chaotic[idx]:
                    sali_status = "FALSE CHAOS" if sali_chaotic[idx] else "REGULAR"
                    gali_status = "FALSE CHAOS" if gali_chaotic[idx] else "REGULAR"
                    report_lines.append(
                        f"Orbit #{idx:03d} | MLE: {mles[idx]:.5f} | SALI: {sali_status:<11} | GALI: {gali_status:<11}"
                    )
            report_lines.append("=" * 60)

        # Write to output text file
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{PREFIX}_regular_sensitivity.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        # Optional baseline assertions
        # assert sali_spec_rate >= 0.90, f"SALI specificity regression: {sali_spec_rate:.1%}"
        # assert gali_spec_rate >= 0.90, f"GALI specificity regression: {gali_spec_rate:.1%}"
