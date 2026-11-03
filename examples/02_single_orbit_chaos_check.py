"""
02_single_orbit_chaos_check.py

Demonstrates running chaos detection (SALI, GALI, and Lyapunov exponents)
on a single integrated orbit and interpreting the summary vs full report.
"""

import logging
import sys

import agama
import numpy as np

from ocd_gd import print_banner, print_kv_table, setup_logging
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

    # 3. Detect Chaos
    classification = detector.detect_chaos()

    # Clean scalar conversions for formatted printing
    sali_chk = _to_scalar(classification.sali_check)
    sali_t = _to_scalar(classification.sali_time)
    gali_chk = _to_scalar(classification.gali_check)
    gali_t = _to_scalar(classification.gali_time)
    lyap_chk = _to_scalar(classification.lyapunov_check)
    lyap_v = _to_scalar(classification.lyapunov_time)

    # Extract indicator values safely from detector
    sali_final = _to_scalar(detector.sali_array[..., -1].min())
    gali_final = _to_scalar(detector.gali_array[..., -1])
    lyap_final = (
        _to_scalar(detector.lyapunov_exponents[-1])
        if detector.lyapunov_exponents is not None and len(detector.lyapunov_exponents) > 0
        else float("nan")
    )

    # ------------------------------------------------------------------
    # 5. Presentation Phase
    # ------------------------------------------------------------------

    # Classification Summary Table
    print_kv_table(
        title="Chaos Detection Summary",
        data={
            "SALI Chaos Check": "Chaotic (1.0)" if sali_chk == 1.0 else "Regular (0.0)",
            "SALI Convergence Time": f"{sali_t:.2f} time units",
            "GALI Chaos Check": "Chaotic (1.0)" if gali_chk == 1.0 else "Regular (0.0)",
            "GALI Convergence Time": f"{gali_t:.2f} time units",
            "Lyapunov Check": "Chaotic (1.0)" if lyap_chk == 1.0 else "Regular (0.0)",
            "Lyapunov Time": f"{lyap_v:.2f}" if np.isfinite(lyap_v) else "inf",
        },
        header_style="bold magenta",
    )

    # Full Indicator Details Table
    n_pts = len(detector.timestamps[0]) if detector.timestamps.ndim == 2 else len(detector.timestamps)
    print_kv_table(
        title="Diagnostic Indicator Metrics",
        data={
            "Time Array Points": str(n_pts),
            "SALI Envelope Length": str(detector.sali_array.shape[-1]),
            "GALI Sequence Length": str(detector.gali_array.shape[-1]),
            "Final Min SALI Value": f"{sali_final:.4e}",
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
