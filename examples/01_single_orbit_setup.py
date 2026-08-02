"""
Example 01 — Single Orbit Setup
================================

The smallest possible end-to-end setup:

1. Build a basic Agama spheroid potential.
2. Print some basic info about that potential.
3. Define a single orbit's initial conditions.
4. Construct an `OrbitChaosDetector` (this integrates the orbit immediately).
5. Print its integration criteria and basic post-integration state.

This example does NOT call `detect_chaos()` — it only covers getting a
detector successfully constructed and integrated. See `02_*` for the next
step (actually running chaos detection).

Run from the repository root:

    python examples/01_single_orbit_setup.py
"""

import sys
import logging
import agama

from ocd_gd.orbit_detector import OrbitChaosDetector
from ocd_gd._logging_config import setup_logging

log = logging.getLogger("examples.01_single_orbit_setup")


def build_potential() -> "agama.Potential":
    """Construct a basic Hernquist-like spheroid potential."""
    agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
    pot = agama.Potential(
        type="Spheroid",
        mass=1e10,
        scaleRadius=5.0,
        gamma=1.0,
        beta=4.0,
    )
    return pot


def print_potential_info(pot: "agama.Potential", R_0: float = 8.0) -> None:
    """Print a few basic derived quantities for the potential."""
    log.info("Potential built: %s", pot)
    log.info("Potential value at (R_0, 0, 0) = %.6g", pot.potential([R_0, 0, 0]))
    log.info("Density at (R_0, 0, 0)        = %.6g", pot.density([R_0, 0, 0]))


def build_single_orbit_ic(R_0: float = 8.0) -> list:
    """A single, mildly perturbed near-circular orbit's initial conditions."""
    # [x, y, z, vx, vy, vz] — perturbed off a purely circular orbit so the
    # deviation vectors have something non-trivial to track.
    ic = [R_0, 0.0, 0.05, 0.0, 180.0, 5.0]
    log.info("Single orbit IC: %s", ic)
    return ic


def main() -> None:
    setup_logging(level=logging.INFO)

    pot = build_potential()
    print_potential_info(pot)

    ic = build_single_orbit_ic()

    log.info("Constructing OrbitChaosDetector (this integrates immediately)...")
    detector = OrbitChaosDetector(ic=ic, pot=pot, iter_time=10.0)

    log.info("Integration criteria: %s", detector.criteria)
    log.info("Number of orbits integrated: %d", detector.num_orbits)
    log.info("Timestamps shape: %s", detector.timestamps.shape)
    log.info("Lyapunov exponents (per orbit): %s", detector.lyapunov_exponents)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - deliberately broad for a clean CLI report
        print(f"\u2717 Example failed: {exc}", file=sys.stderr)
        if __debug__ and "OCD_GD_DEBUG" in __import__("os").environ:
            raise
        sys.exit(1)
    else:
        print("\u2713 Example completed successfully")
        sys.exit(0)
