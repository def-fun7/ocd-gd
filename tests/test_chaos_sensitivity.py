import agama
from pathlib import Path
import numpy as np
import pytest
from ocd_gd.orbit_detector import OrbitChaosDetector

BENCHMARK_NPZ = Path("data/initial_conditions/chaotic_ics_benchmark_185_0.1.npz")
MW_POTENTIAL = "data/potentials/MWPotentialHunter24_full.ini"

PREFIX = "S2.75_G20"
SALI_THRESHOLD = 0.0018
GALI_THRESHOLD = 1e-20
SALI_WIND_S = 10
GALI_WIND_S = 100
ITER_TIME = 10


def load_benchmark_ics():
    """Helper to load initial conditions and MLE ground truth from NPZ."""
    if not BENCHMARK_NPZ.exists():
        pytest.skip(f"{BENCHMARK_NPZ} not found, skipping benchmark tests.")

    data = np.load(BENCHMARK_NPZ)
    return data["ics"], data["mles"]


class TestChaosSensitivity:
    """Benchmark tests checking detector performance against verified chaotic ICs."""

    def test_detector_benchmark_sensitivity(
        self, output_dir=Path("./outputs/benchmark_results")
    ):
        """
        Evaluate detector sensitivity across the full benchmark set of chaotic orbits
        and write report to a .txt file.
        """
        ics, mles = load_benchmark_ics()
        mw_potential = agama.Potential(MW_POTENTIAL)

        # Run detection on all benchmark ICs
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

        # Flatten check arrays to 1D boolean masks (True = Chaos detected)
        sali_passed = summary.sali_check.flatten().astype(bool)
        gali_passed = summary.gali_check.flatten().astype(bool)

        total_orbits = len(ics)
        sali_hits = int(np.sum(sali_passed))
        gali_hits = int(np.sum(gali_passed))

        sali_misses = total_orbits - sali_hits
        gali_misses = total_orbits - gali_hits

        sali_rate = sali_hits / total_orbits
        gali_rate = gali_hits / total_orbits

        # Prepare summary report content
        report_lines = [
            "=" * 60,
            "BENCHMARK CHAOTIC ORBIT DETECTOR REPORT",
            "=" * 60,
            f"Dataset NPZ Path     : {BENCHMARK_NPZ}",
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
            f"  • SALI Sensitivity : {sali_hits}/{total_orbits} ({sali_rate:.1%}) | Missed: {sali_misses}",
            f"  • GALI Sensitivity : {gali_hits}/{total_orbits} ({gali_rate:.1%}) | Missed: {gali_misses}",
            "=" * 60,
        ]

        # Append missed chaotic orbit details if any exist
        if sali_misses > 0 or gali_misses > 0:
            report_lines.append(
                "\nMISSED ORBITS DETAIL (Chaotic ICs Missed as Regular):"
            )
            report_lines.append("-" * 60)
            for idx in range(total_orbits):
                if not (sali_passed[idx] and gali_passed[idx]):
                    sali_status = "PASS" if sali_passed[idx] else "MISS (inf)"
                    gali_status = "PASS" if gali_passed[idx] else "MISS (inf)"
                    report_lines.append(
                        f"Orbit #{idx:03d} | MLE: {mles[idx]:.5f} | SALI: {sali_status:<10} | GALI: {gali_status:<10}"
                    )
            report_lines.append("=" * 60)

        # Write to output text file
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{PREFIX}_chaotic_sensitivity.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        # # Baseline sensitivity assertions
        # assert sali_rate >= 0.90, f"SALI sensitivity regression: {sali_rate:.1%}"
        # assert gali_rate >= 0.95, f"GALI sensitivity regression: {gali_rate:.1%}"
