"""
03_batch_orbits.py

Demonstrates running OrbitChaosDetector on a batch of initial conditions
and extracting population-level chaos statistics with chaos_summary().
"""

import sys
import logging
import numpy as np
import agama

from ocd_gd.orbit_detector import OrbitChaosDetector
from ocd_gd._logging_config import setup_logging
from ocd_gd._logging_config import print_banner, print_kv_table, print_dataframe_table


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


def generate_batch_ics(num_orbits: int = 5) -> np.ndarray:
    """Generate initial conditions with varying radial distances."""
    radii = np.linspace(2.0, 10.0, num_orbits)
    ics = []
    for r in radii:
        # [x, y, z, vx, vy, vz]
        ics.append([r, 0.0, 0.05, 0.0, 180.0, 5.0])
    return np.array(ics)


def main() -> None:
    # 1. Setup Logging
    setup_logging(level=logging.INFO)

    print_banner("ocd-gd", "Batch Orbits & Chaos Summary Example")

    # 2. Setup Potential & Initial Conditions Batch
    pot = build_potential()
    ics = generate_batch_ics(num_orbits=5)

    # 3. Integrate Batch Orbits
    detector = OrbitChaosDetector(
        ic=ics,
        pot=pot,
        iter_time=10.0,
        sali_threshold=1e-3,
        gali_threshold=1e-20,
    )

    # 4. Extract Survey Statistics Across the Batch
    survey = detector.chaos_summary()

    # ------------------------------------------------------------------
    # 5. Presentation Phase
    # ------------------------------------------------------------------

    # Indicator Statistics Comparison
    print_dataframe_table(
        title="Indicator Classification Breakdown",
        headers=["Indicator", "Chaotic Orbits", "Regular Orbits", "Chaotic Fraction"],
        rows=[
            [
                "SALI",
                str(survey.sali.n_chaotic),
                str(survey.sali.n_regular),
                f"{survey.sali.chaotic_fraction:.1%}",
            ],
            [
                "GALI",
                str(survey.gali.n_chaotic),
                str(survey.gali.n_regular),
                f"{survey.gali.chaotic_fraction:.1%}",
            ],
            [
                "Lyapunov",
                str(survey.lyapunov.n_chaotic),
                str(survey.lyapunov.n_regular),
                f"{survey.lyapunov.chaotic_fraction:.1%}",
            ],
        ],
        header_style="bold magenta",
    )

    # Cross-Indicator Agreement Metrics
    print_kv_table(
        title="Cross-Indicator Agreement",
        data={
            "SALI <-> GALI Agreement": f"{survey.agreement.sali_gali_agreement:.1%}",
            "SALI <-> Lyapunov Agreement": f"{survey.agreement.sali_lyapunov_agreement:.1%}",
            "GALI <-> Lyapunov Agreement": f"{survey.agreement.gali_lyapunov_agreement:.1%}",
            "All Agree Chaotic": survey.agreement.all_agree_chaotic,
            "All Agree Regular": survey.agreement.all_agree_regular,
            "Total Disagreements": survey.agreement.disagreement,
        },
        header_style="bold green",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\u2717 Example failed: {exc}", file=sys.stderr)
        sys.exit(1)
