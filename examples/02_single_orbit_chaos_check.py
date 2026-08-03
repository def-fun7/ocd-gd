"""
02_single_orbit_chaos_check.py

Demonstrates running chaos detection (SALI, GALI, and Lyapunov exponents)
on a single integrated orbit and interpreting the summary vs full report.
"""

import logging
import sys

import agama
import numpy as np

from ocd_gd._logging_config import print_banner, print_kv_table, setup_logging
from ocd_gd.orbit_detector import OrbitChaosDetector


def build_potential() -> "agama.Potential":
    """Construct a basic Hernquist-like spheroid potential."""
    agama.setUnits(mass=1, length=1, velocity=1)
    return agama.Potential(
        type="Spheroid",
        mass=1e10,
        scaleRadius=5.0,
        gamma=1.0,
        beta=4.0,
    )


def _to_scalar(val: str | float | np.ndarray) -> float:
    """Helper to cleanly extract a Python float from a float, int, or 0D/1D NumPy array."""
    if isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val)


def main() -> None:
    # 1. Setup Logging
    setup_logging(level=logging.INFO)

    print_banner("ocd-gd", "Single Orbit Chaos Check Example")

    # 2. Integrate Orbit (Internal logs fire first)
    pot = build_potential()
    ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]

    detector = OrbitChaosDetector(
        ic=ic,
        pot=pot,
        iter_time=10.0,
        sali_threshold=1e-3,
        gali_threshold=1e-20,
    )

    # 3. Detect Chaos (Summary)
    summary = detector.detect_chaos(check_only=True)

    # 4. Detect Chaos (Full Report for deeper inspectability)
    full_report = detector.detect_chaos(check_only=False)

    # Clean scalar conversions for formatted printing
    sali_chk = _to_scalar(summary.sali_check)
    sali_t = _to_scalar(summary.sali_time)
    gali_chk = _to_scalar(summary.gali_check)
    gali_t = _to_scalar(summary.gali_time)
    lyap_chk = _to_scalar(summary.lyapunov_check)
    lyap_v = _to_scalar(summary.lyapunov_time)

    # Extract full-report array values safely
    sali_final = _to_scalar(full_report.sali_array[-1])
    gali_final = _to_scalar(full_report.gali_array[-1])
    lyap_final = (
        (full_report.lyapunov_array[-1])
        if np.ndim(full_report.lyapunov_array) > 0
        else (full_report.lyapunov_array)
    )

    # ------------------------------------------------------------------
    # 5. Presentation Phase
    # ------------------------------------------------------------------

    # Classification Summary Table
    print_kv_table(
        title="Chaos Detection Summary (check_only=True)",
        data={
            "SALI Chaos Check": "Chaotic (1.0)" if sali_chk == 1.0 else "Regular (0.0)",
            "SALI Convergence Time": f"{sali_t:.2f} time units",
            "GALI Chaos Check": "Chaotic (1.0)" if gali_chk == 1.0 else "Regular (0.0)",
            "GALI Convergence Time": f"{gali_t:.2f} time units",
            "Lyapunov Check": "Chaotic (1.0)" if lyap_chk == 1.0 else "Regular (0.0)",
            "Lyapunov Value": f"{lyap_v:.4e}",
        },
        header_style="bold magenta",
    )

    # Full Report Details Table
    print_kv_table(
        title="Full Diagnostic Report (check_only=False)",
        data={
            "Time Array Points": str(len(full_report.timestamps)),
            "SALI Sequence Length": str(len(full_report.sali_array)),
            "GALI Sequence Length": str(len(full_report.gali_array)),
            "Final SALI Value": f"{sali_final:.4e}",
            "Final GALI Value": f"{gali_final:.4e}",
            "Final Lyapunov Exp": f"{lyap_final:.4e}",
        },
        header_style="bold green",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\u2717 Example failed: {exc}", file=sys.stderr)
        sys.exit(1)
