import logging
import sys

import agama

from ocd_gd._logging_config import print_banner, print_kv_table, setup_logging
from ocd_gd.orbit_detector import OrbitChaosDetector


def build_potential() -> "agama.Potential":
    """Construct a basic Hernquist-like spheroid potential."""
    agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
    return agama.Potential(
        type="Spheroid",
        mass=1e10,
        scaleRadius=5.0,
        gamma=1.0,
        beta=4.0,
    )


def main() -> None:
    # 1. Logging Setup
    setup_logging(level=logging.INFO)

    print_banner("ocd-gd", "Single Orbit Setup Example")

    # 2. Computation Phase (All internal log messages fire here at the top)
    pot = build_potential()
    ic = [8.0, 0.0, 0.05, 0.0, 180.0, 5.0]

    detector = OrbitChaosDetector(ic=ic, pot=pot, iter_time=10.0)

    # ------------------------------------------------------------------
    # 3. Presentation Phase (All tables rendered uninterrupted at bottom)
    # ------------------------------------------------------------------

    # Potential Properties Table
    print_kv_table(
        title="Potential Properties",
        data={
            "Type": str(pot),
            "Φ(R₀, 0, 0)": f"{pot.potential([8.0, 0, 0]):.2f}",
            "ρ(R₀, 0, 0)": f"{pot.density([8.0, 0, 0]):.2f}",
        },
        header_style="bold magenta",
    )

    # Integration State Table
    print_kv_table(
        title="Integration State",
        data={
            "Orbits Integrated": detector.num_orbits,
            "Timestamps Shape": str(detector.timestamps.shape),
            "Iter Time": f"{detector.criteria.iter_time} gyro-periods",
            "Target Accuracy": f"{detector.criteria.accuracy:.1e}",
            "Lyapunov Exponents": str(detector.lyapunov_exponents),
        },
        header_style="bold green",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\u2717 Example failed: {exc}", file=sys.stderr)
        sys.exit(1)
