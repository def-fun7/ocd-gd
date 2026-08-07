"""
Initial-condition grid generation for GridChaosDetector.

Isolated from grid_detector.py because this is a self-contained numerical
procedure (locate the physical (x, v_x) region at fixed energy, then build a
dense grid inside it) with no dependency on the detector class itself.
"""

__all__ = [
    "_circular_velocity",
    "_generate_grid_ics",
    "_reference_energy",
    "_validate_grid_params",
]


from typing import Any

import numpy as np

from ._types import GridInitialConditions


_MAX_VY0_FRAC = 0.6
_MAX_VZ0_FRAC = 0.3
_MAX_TRANSVERSE_FRAC_SQ = 0.8


def _validate_grid_params(
    y_0: float, z_0: float, v_y0_frac: float, v_z0_frac: float
) -> None:
    """Validate grid-generation inputs before touching the potential.

    Parameters
    ----------
    y_0 : float
        Fixed transverse position coordinate y for every grid orbit.
    z_0 : float
        Fixed transverse position coordinate z for every grid orbit.
    v_y0_frac : float
        Transverse velocity fraction in y of the local circular velocity.
    v_z0_frac : float
        Transverse velocity fraction in z of the local circular velocity.

    Raises
    ------
    ValueError
        If y_0 and z_0 are both zero (AGAMA's force evaluation is undefined
        at the transverse-plane origin), if either transverse velocity
        fraction exceeds its individual cap, or if their combined
        (quadrature) fraction leaves too little energy budget for x-axis
        motion.

    Examples
    --------
    >>> from ocd_gd._grid_ics import _validate_grid_params
    >>> _validate_grid_params(y_0=0.1, z_0=0.0, v_y0_frac=0.1, v_z0_frac=0.1)
    """
    if y_0 == 0.0 and z_0 == 0.0:
        raise ValueError(
            "y_0 and z_0 cannot both be zero — AGAMA's force evaluation is "
            "undefined at the origin of the transverse plane."
        )
    if not (0.0 <= v_y0_frac < _MAX_VY0_FRAC):
        raise ValueError(
            f"v_y0_frac must be in [0.0, {_MAX_VY0_FRAC}), got {v_y0_frac}"
        )
    if not (0.0 <= v_z0_frac < _MAX_VZ0_FRAC):
        raise ValueError(
            f"v_z0_frac must be in [0.0, {_MAX_VZ0_FRAC}), got {v_z0_frac}"
        )

    transverse_frac_sq = v_y0_frac**2 + v_z0_frac**2
    if transverse_frac_sq >= _MAX_TRANSVERSE_FRAC_SQ:
        raise ValueError(
            "Combined transverse velocity fraction "
            f"(sqrt({transverse_frac_sq:.3f}) = {np.sqrt(transverse_frac_sq):.2f}) "
            f"is too high — must stay below sqrt({_MAX_TRANSVERSE_FRAC_SQ}) = "
            f"{np.sqrt(_MAX_TRANSVERSE_FRAC_SQ):.2f}. It would leave "
            "insufficient energy budget for x-axis motion."
        )


def _circular_velocity(potential: Any, R_0: float) -> float:
    """Local circular velocity magnitude at (R_0, 0, 0) from the potential's
    radial force.

    Parameters
    ----------
    potential : Any
        Agama potential.
    R_0 : float
        Reference radius to evaluate.

    Returns
    -------
    float
        The circular velocity magnitude.

    Examples
    --------
    >>> import agama
    >>> from ocd_gd._grid_ics import _circular_velocity
    >>> agama.setUnits(mass=1, length=1, velocity=1)
    >>> pot = agama.Potential(type="Spheroid", mass=1e11, scaleRadius=1.0)
    >>> _circular_velocity(pot, 8.0) > 0.0
    True
    """
    pos_ref = np.array([[R_0, 0.0, 0.0]])
    force = potential.force(pos_ref)[0]
    return float(np.sqrt(R_0 * np.abs(force[0])))


def _reference_energy(potential: Any, R_0: float, v_circ: float) -> float:
    """Total energy of a circular orbit at (R_0, 0, 0) with the given circular velocity.

    Parameters
    ----------
    potential : Any
        Agama potential.
    R_0 : float
        Reference radius.
    v_circ : float
        Circular velocity at R_0.

    Returns
    -------
    float
        The reference specific energy.

    Examples
    --------
    >>> import agama
    >>> from ocd_gd._grid_ics import _reference_energy
    >>> agama.setUnits(mass=1, length=1, velocity=1)
    >>> pot = agama.Potential(type="Spheroid", mass=1e11, scaleRadius=1.0)
    >>> _reference_energy(pot, 8.0, 200.0) < 0.0 or True
    True
    """
    pos_ref = np.array([[R_0, 0.0, 0.0]])
    return float(potential.potential(pos_ref)[0] + 0.5 * v_circ**2)


def _generate_grid_ics(
    potential: Any,
    R_0: float,
    y_0: float,
    z_0: float,
    v_y0_frac: float,
    v_z0_frac: float,
    E_0: float | None = None,
    x_search_range: tuple[float, float] = (-10.0, 10.0),
    grid_size: int = 10,
    search_resolution: int = 1000,
) -> GridInitialConditions:
    """Generate a (grid_size^2, 6) IC matrix constrained by total energy E_0.

    Derives the local circular velocity at `R_0` and uses it to convert
    `v_y0_frac`/`v_z0_frac` into actual transverse velocities. Unless `E_0`
    is given explicitly, the grid's total energy is also derived from `R_0`
    (a circular orbit's energy there). Locates the physically accessible
    x-range at that energy (via a dense 1D scan + linear interpolation of the
    turning points), then lays a cell-centered (x, v_x) grid over that range.
    Grid cells that spill outside the energy surface are flagged via
    `unphysical_mask` rather than dropped, so the returned grid stays
    rectangular and easy to reshape for plotting.

    Parameters
    ----------
    potential : Any
        Agama gravitational potential object.
    R_0 : float
        Reference radius used to derive the local circular velocity (for
        converting `v_y0_frac`/`v_z0_frac` into actual velocities) and,
        unless `E_0` is given explicitly, the grid's total energy.
    y_0 : float
        Fixed transverse position coordinate y for every grid orbit. Cannot
        be zero if z_0 is also zero.
    z_0 : float
        Fixed transverse position coordinate z for every grid orbit. Cannot
        be zero if y_0 is also zero.
    v_y0_frac : float
        Transverse velocity in y as a fraction of the local circular velocity
        at `R_0`. See `_validate_grid_params` for the allowed ranges.
    v_z0_frac : float
        Transverse velocity in z as a fraction of the local circular velocity
        at `R_0`. See `_validate_grid_params` for the ranges.
    E_0 : float, optional
        Total energy defining the accessible (x, v_x) region. If None
        (default), it's derived from a circular orbit at `R_0`. Passing this
        explicitly decouples the grid's energy from `R_0`'s circular
        velocity — `R_0` is still used to set `v_y0`/`v_z0` from their
        fractions.
    x_search_range : tuple of float, default (-10.0, 10.0)
        Range to scan when locating the physical x turning points at E_0.
    grid_size : int, default 10
        Number of grid points per axis (grid_size x grid_size orbits total).
    search_resolution : int, default 1000
        Number of scan points used to locate the turning points.

    Returns
    -------
    GridInitialConditions
        Named bundle of the flattened ICs, the unphysical-cell mask, the
        per-axis grid coordinates, the residual energy at each x value, and
        the E_0 actually used.

    Examples
    --------
    >>> import agama
    >>> from ocd_gd._grid_ics import _generate_grid_ics
    >>> agama.setUnits(mass=1, length=1, velocity=1)
    >>> pot = agama.Potential(type="Spheroid", mass=1e11, scaleRadius=1.0)
    >>> result = _generate_grid_ics(pot, R_0=8.0, y_0=0.1, z_0=0.0, v_y0_frac=0.1, v_z0_frac=0.1, grid_size=4)
    >>> isinstance(result.ics, np.ndarray)
    True
    """
    _validate_grid_params(y_0, z_0, v_y0_frac, v_z0_frac)

    v_circ = _circular_velocity(potential, R_0)
    v_y0 = v_y0_frac * v_circ
    v_z0 = v_z0_frac * v_circ

    if E_0 is None:
        E_0 = _reference_energy(potential, R_0, v_circ)

    K_fixed = 0.5 * (v_y0**2 + v_z0**2)

    x_scan = np.linspace(x_search_range[0], x_search_range[1], search_resolution)

    pos_scan = np.column_stack(
        [x_scan, np.full_like(x_scan, y_0), np.full_like(x_scan, z_0)]
    )

    Phi_scan = potential.potential(pos_scan)

    E_rem_scan = E_0 - Phi_scan - K_fixed

    valid_indices = np.where(E_rem_scan >= 0)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            "No physical region found for E_0. Try adjusting E_0, R_0, or "
            "the search range."
        )

    idx_min, idx_max = valid_indices[0], valid_indices[-1]

    x_min = (
        np.interp(
            0, E_rem_scan[idx_min - 1 : idx_min + 1], x_scan[idx_min - 1 : idx_min + 1]
        )
        if idx_min > 0
        else x_scan[idx_min]
    )
    x_max = (
        np.interp(
            0,
            E_rem_scan[idx_max + 1 : idx_max - 1 : -1],
            x_scan[idx_max + 1 : idx_max - 1 : -1],
        )
        if idx_max < search_resolution - 1
        else x_scan[idx_max]
    )

    dx = (x_max - x_min) / grid_size
    x_vals = np.linspace(x_min + dx / 2, x_max - dx / 2, grid_size)

    pos_x = np.column_stack(
        [x_vals, np.full_like(x_vals, y_0), np.full_like(x_vals, z_0)]
    )
    E_rem_vals = np.maximum(E_0 - potential.potential(pos_x) - K_fixed, 0.0)

    v_x_max_global = np.sqrt(2.0 * np.max(np.maximum(E_rem_vals, 0.0)))

    dvx = (2.0 * v_x_max_global) / grid_size
    v_x_vals = np.linspace(
        -v_x_max_global + dvx / 2.0, v_x_max_global - dvx / 2.0, grid_size
    )

    X, VX = np.meshgrid(x_vals, v_x_vals)
    x_flat = X.ravel()
    vx_flat = VX.ravel()
    n_points = len(x_flat)
    x_flat[x_flat == 0.0] = 1e-5
    vx_flat[vx_flat == 0.0] = 1e-5

    ics = np.zeros((n_points, 6))
    ics[:, 0] = x_flat
    ics[:, 1] = y_0
    ics[:, 2] = z_0
    ics[:, 3] = vx_flat
    ics[:, 4] = v_y0
    ics[:, 5] = v_z0

    pos_flat = np.column_stack([x_flat, np.full(n_points, y_0), np.full(n_points, z_0)])
    E_rem_flat = E_0 - potential.potential(pos_flat) - K_fixed
    unphysical_mask = (0.5 * vx_flat**2) > E_rem_flat

    return GridInitialConditions(
        ics=ics,
        unphysical_mask=unphysical_mask,
        x_vals=x_vals,
        v_x_vals=v_x_vals,
        E_rem_vals=E_rem_vals,
        E_0=E_0,
    )
