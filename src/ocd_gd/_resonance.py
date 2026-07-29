"""
Resonance-radius diagnostics for a rotating potential.

Used to overlay corotation and Lindblad radii on GridChaosDetector's chaos
maps, connecting resonance islands to the classical orbital-resonance
theory that predicts where they should appear. Everything here is derived
from `potential.force`/`potential.potential` alone (the same primitives
`_circular_velocity` already uses), so it works for any agama potential —
no assumption about rotation curves being available as a separate API.
"""

from typing import Any, NamedTuple, Optional, Tuple
import numpy as np


class ResonanceRadii(NamedTuple):
    """Corotation and Lindblad radii for a potential at a given pattern
    speed. Any radius with no root in the search range is None — most
    commonly `corotation`/`inner_lindblad`/`outer_lindblad` are all None for
    a non-rotating potential (`omega == 0`), since there's nothing to be
    resonant with.
    """

    corotation: Optional[float]
    inner_lindblad: Optional[float]
    outer_lindblad: Optional[float]


def _circular_velocity_curve(potential: Any, r_vals: np.ndarray) -> np.ndarray:
    """Circular velocity v_circ(R) at each radius, evaluated along the
    x-axis (y=z=0)."""
    pos = np.column_stack([r_vals, np.zeros_like(r_vals), np.zeros_like(r_vals)])
    force = potential.force(pos)
    return np.sqrt(r_vals * np.abs(force[:, 0]))


def _angular_velocity_curve(potential: Any, r_vals: np.ndarray) -> np.ndarray:
    """Circular angular velocity Omega_circ(R) = v_circ(R) / R."""
    return _circular_velocity_curve(potential, r_vals) / r_vals


def _epicyclic_frequency_curve(potential: Any, r_vals: np.ndarray) -> np.ndarray:
    """Epicyclic frequency kappa(R) via the standard relation
    kappa^2 = R d(Omega^2)/dR + 4 Omega^2, using a numerical derivative of
    Omega_circ(R) over the same r_vals grid.
    """
    omega_circ = _angular_velocity_curve(potential, r_vals)
    d_omega_sq_dR = np.gradient(omega_circ**2, r_vals)
    kappa_sq = r_vals * d_omega_sq_dR + 4.0 * omega_circ**2
    return np.sqrt(np.maximum(kappa_sq, 0.0))


def _first_root(r_vals: np.ndarray, diff: np.ndarray) -> Optional[float]:
    """Linear-interpolate the first sign change in `diff` over `r_vals`, or
    None if `diff` never changes sign (no resonance in the search range)."""
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]
    return float(np.interp(0.0, [diff[i], diff[i + 1]], [r_vals[i], r_vals[i + 1]]))


def compute_resonance_radii(
    potential: Any,
    omega_pattern: float,
    m: int = 2,
    r_search_range: Tuple[float, float] = (0.1, 30.0),
    search_resolution: int = 2000,
) -> ResonanceRadii:
    """Locate the corotation radius and inner/outer Lindblad radii for a
    potential rotating at `omega_pattern`.

    Corotation is where the circular angular velocity equals the pattern
    speed. The Lindblad radii are where `omega_pattern = Omega_circ(R) ∓
    kappa(R)/m` — the standard condition for an m-fold (m=2 for a bar)
    resonance between the pattern and epicyclic oscillation.

    Parameters
    ----------
    potential : agama.Potential
        Potential to evaluate.
    omega_pattern : float
        Pattern speed (e.g. the `omega` a GridChaosDetector was built with).
        If 0.0, no radius will be found — there's no resonance without
        rotation.
    m : int, default 2
        Resonance order (2 for a standard bar).
    r_search_range : tuple of float, default (0.1, 30.0)
        Radius range to scan for roots. Should comfortably bracket the
        expected resonance radii for the potential in question.
    search_resolution : int, default 2000
        Number of scan points; higher gives a more precise root location.

    Returns
    -------
    ResonanceRadii
        `corotation`, `inner_lindblad`, `outer_lindblad` — each None if no
        root was found in `r_search_range`.
    """
    r_vals = np.linspace(r_search_range[0], r_search_range[1], search_resolution)
    omega_circ = _angular_velocity_curve(potential, r_vals)
    kappa = _epicyclic_frequency_curve(potential, r_vals)

    corotation = _first_root(r_vals, omega_circ - omega_pattern)
    inner_lindblad = _first_root(r_vals, (omega_circ - kappa / m) - omega_pattern)
    outer_lindblad = _first_root(r_vals, (omega_circ + kappa / m) - omega_pattern)

    return ResonanceRadii(
        corotation=corotation,
        inner_lindblad=inner_lindblad,
        outer_lindblad=outer_lindblad,
    )
