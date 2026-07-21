"""Integration tests checking chaos indicators against known analytical physics."""

import numpy as np
import pytest
import agama

from ocd_gd.orbit_detector import OrbitChaosDetector

# Enable units in Agama (Kpc, Msun, km/s)
agama.setUnits(length=1, mass=1, velocity=1)


@pytest.fixture(scope="module")
def spherical_potential():
    """Spherical Hernquist potential — strictly integrable, zero chaos."""
    return agama.Potential(
        type="Spheroid",
        mass=1e11,
        scaleradius=10.0,  # Note: AGAMA uses lowercase 'scaleradius' or 'scaleRadius'
        gamma=1,  # Inner slope for Hernquist
        beta=4,  # Outer slope for Hernquist
        xi=1,  # Axis ratio z/x (1 = spherical)
    )


@pytest.fixture(scope="module")
def triaxial_potential():
    """Triaxial Ferrers potential — highly chaotic regime in central regions."""
    return agama.Potential("data/potentials/MWPotentialHunter24_full.ini")


class TestPhysicalPotentials:
    """Physics-driven integration tests verifying true/false chaos classification."""

    def test_spherical_potential_is_strictly_regular(self, spherical_potential):
        """Spherical potentials must yield strictly regular orbits (SALI > threshold, GALI > threshold)."""
        # Circular-like orbit in spherical potential
        ic = [10.0, 0.0, 0.0, 0.0, 150.0, 0.0]

        detector = OrbitChaosDetector(
            ic=ic,
            pot=spherical_potential,
        )

        summary = detector.detect_chaos(check_only=True)

        # In integrable systems:
        # 1. SALI levels off to a non-zero constant (~1.0 - 2.0).
        # 2. Maximum Lyapunov Exponent (MLE) drops to ~0.0.
        assert (
            summary.sali_check[0] == 0
        ), "Spherical orbit misclassified as chaotic by SALI"
        assert (
            summary.gali_check[0] == 0
        ), "Spherical orbit misclassified as chaotic by GALI"
        assert np.isclose(detector.lyapunov_exponents[0, 0], 0.0, atol=1e-2)

    def test_triaxial_potential_detects_chaos(self, triaxial_potential):
        """Triaxial potentials with high energy should trigger rapid SALI/GALI decay."""
        # High energy box orbit near triaxial core
        ic = [1.0, 0.1, 0, 0.0, 10.0, 10.0]

        detector = OrbitChaosDetector(
            ic=ic,
            pot=triaxial_potential,
        )

        summary = detector.detect_chaos(check_only=True)

        # In chaotic motion, SALI and GALI decay exponentially toward 0.0
        assert (
            detector.lyapunov_exponents[0, 0] > 1e-3
        ), "Lyapunov exponent failed to grow"
        assert (
            summary.sali_check[0] == 1
        ), "Chaotic triaxial orbit went undetected by SALI"
        assert (
            summary.gali_check[0] == 1
        ), "Chaotic triaxial orbit went undetected by GALI"
