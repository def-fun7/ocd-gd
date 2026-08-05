"""
Box vs. loop orbit-family classification ("family check").

The cheapest meaningful orbit-family label available from an already-
integrated trajectory, with no extra integration required: a loop orbit
circulates around the center with one consistent sense of rotation, so its
specific in-plane angular momentum L_z(t) = x*vy - y*vx keeps one sign for
the whole integration; a box orbit doesn't circulate at all, so L_z(t)
crosses zero repeatedly. This is a standard first-pass split in the
bar-orbit literature (e.g. Pfenniger 1984; Skokos, Patsis & Athanassoula
2002).

This deliberately stops at box/loop -- it does NOT distinguish which
resonance family a loop orbit belongs to (e.g. x1 vs x2 near the ILR).
That needs frequency analysis of the trajectory (FFT + peak-picking to get
the two fundamental in-plane frequencies, then their ratio) and is a
separate, heavier piece of work; see `orbit_family`'s docstring on
`OrbitChaosDetector` for where that would plug in alongside this.
"""

from __future__ import annotations

__all__ = ["FamilyStats", "classify_box_loop", "summarize_family"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FamilyStats:
    """Counts/fractions/indices for the box/loop split, mirroring
    `MethodChaosStats`'s shape so the two can be reported side by side
    (or combined, e.g. "regular AND loop" fractions)."""

    n_loop: int
    n_box: int
    n_total: int
    loop_fraction: float
    box_fraction: float
    loop_indices: npt.NDArray[np.intp]
    box_indices: npt.NDArray[np.intp]


def classify_box_loop(
    traj_arr: npt.NDArray[np.float64], rel_tol: float = 1e-8
) -> npt.NDArray[np.str_]:
    """Classify each orbit in a trajectory batch as `"loop"` or `"box"`.

    Parameters
    ----------
    traj_arr : ndarray, shape (n_orbits, n_times, 6)
        Integrated trajectories (x, y, z, vx, vy, vz per timestep), i.e.
        `OrbitChaosDetector.trajectories`.
    rel_tol : float, default 1e-8
        A sign change only counts if |L_z| at both surrounding samples
        exceeds `rel_tol` times that orbit's own max |L_z| -- otherwise a
        genuinely near-zero-but-noisy L_z (e.g. a near-radial orbit
        passing close to the center) would register spurious flips from
        floating-point noise around zero rather than an actual reversal
        of circulation sense.

    Returns
    -------
    ndarray of str, shape (n_orbits,)
        `"loop"` if L_z(t) keeps one sign (within `rel_tol`) for the
        whole integration, `"box"` if it changes sign at least once.
    """
    x, y = traj_arr[:, :, 0], traj_arr[:, :, 1]
    vx, vy = traj_arr[:, :, 3], traj_arr[:, :, 4]
    Lz = x * vy - y * vx  # shape (n_orbits, n_times)

    scale = np.abs(Lz).max(axis=1, keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    significant = np.abs(Lz) > rel_tol * scale

    sign = np.sign(Lz)
    # A flip only counts between two consecutive samples that are both
    # "significant" -- a pass through genuinely tiny L_z is ignored rather
    # than counted as circulation reversing.
    flips = (sign[:, :-1] * sign[:, 1:] < 0) & significant[:, :-1] & significant[:, 1:]
    is_box = flips.any(axis=1)
    return np.where(is_box, "box", "loop")


def summarize_family(family: npt.NDArray[np.str_]) -> FamilyStats:
    """Build a `FamilyStats` block from a `classify_box_loop` result."""
    is_loop = family == "loop"
    loop_indices = np.where(is_loop)[0]
    box_indices = np.where(~is_loop)[0]
    n_total = len(family)
    return FamilyStats(
        n_loop=len(loop_indices),
        n_box=len(box_indices),
        n_total=n_total,
        loop_fraction=len(loop_indices) / n_total if n_total else float("nan"),
        box_fraction=len(box_indices) / n_total if n_total else float("nan"),
        loop_indices=loop_indices,
        box_indices=box_indices,
    )
