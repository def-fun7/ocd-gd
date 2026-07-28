"""
Initial-condition grid generation for GridChaosDetector.

Isolated from grid_detector.py because this is a self-contained numerical
procedure (locate the physical (x, v_x) region at fixed energy, then build a
dense grid inside it) with no dependency on the detector class itself.
"""

from typing import Any, Tuple
import numpy as np

from ._types import GridInitialConditions


def _generate_grid_ics(
    potential: Any,
    E_0: float,
    y_0: float,
    z_0: float,
    v_y0: float,
    v_z0: float,
    x_search_range: Tuple[float, float] = (-10.0, 10.0),
    grid_size: int = 10,
    search_resolution: int = 1000,
) -> GridInitialConditions:
    """Generate a (grid_size^2, 6) IC matrix constrained by total energy E_0.

    Locates the physically accessible x-range at the given energy (via a
    dense 1D scan + linear interpolation of the turning points), then lays a
    cell-centered (x, v_x) grid over that range with y_0/z_0/v_y0/v_z0 held
    fixed for every point. Grid cells that spill outside the energy surface
    are flagged via `unphysical_mask` rather than dropped, so the returned
    grid stays rectangular and easy to reshape for plotting.

    Parameters
    ----------
    potential : agama.Potential
        Agama gravitational potential object.
    E_0 : float
        Fixed total energy defining the accessible (x, v_x) region.
    y_0, z_0 : float
        Fixed transverse position coordinates for every grid orbit.
    v_y0, v_z0 : float
        Fixed transverse velocity components for every grid orbit.
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
        per-axis grid coordinates, and the residual energy at each x value.
    """
    # 1. Fixed kinetic energy from off-axis motion
    K_fixed = 0.5 * (v_y0**2 + v_z0**2)

    # 2. Build dense 1D probe points along x to find turning points
    x_scan = np.linspace(x_search_range[0], x_search_range[1], search_resolution)

    # AGAMA expects an (N, 3) position array [x, y, z]
    pos_scan = np.column_stack(
        [x_scan, np.full_like(x_scan, y_0), np.full_like(x_scan, z_0)]
    )

    # Evaluate potential: potential(pos) returns a 1D array of Phi(x, y_0, z_0)
    Phi_scan = potential.potential(pos_scan)

    # Calculate residual energy available for x and v_x
    E_rem_scan = E_0 - Phi_scan - K_fixed

    # 3. Locate physical region (E_rem >= 0)
    valid_indices = np.where(E_rem_scan >= 0)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            "No physical region found for E_0. Try adjusting E_0 or search range."
        )

    idx_min, idx_max = valid_indices[0], valid_indices[-1]

    # Refine boundaries via linear interpolation
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

    # 4. Create spatial x grid
    dx = (x_max - x_min) / grid_size
    x_vals = np.linspace(x_min + dx / 2, x_max - dx / 2, grid_size)

    # Evaluate E_rem at exact x_vals points
    pos_x = np.column_stack(
        [x_vals, np.full_like(x_vals, y_0), np.full_like(x_vals, z_0)]
    )
    E_rem_vals = np.maximum(E_0 - potential.potential(pos_x) - K_fixed, 0.0)

    # 5. Global v_x velocity bounds
    v_x_max_global = np.sqrt(2.0 * np.max(np.maximum(E_rem_vals, 0.0)))

    # 6. Cell-centered v_x grid (prevents exact 0.0)
    dvx = (2.0 * v_x_max_global) / grid_size
    v_x_vals = np.linspace(
        -v_x_max_global + dvx / 2.0, v_x_max_global - dvx / 2.0, grid_size
    )

    # 7. Build 2D meshgrid and flatten
    X, VX = np.meshgrid(x_vals, v_x_vals)
    x_flat = X.ravel()
    vx_flat = VX.ravel()
    n_points = len(x_flat)
    x_flat[x_flat == 0.0] = 1e-5
    vx_flat[vx_flat == 0.0] = 1e-5

    # 8. Stack into (grid_size^2, 6) IC array
    ics = np.zeros((n_points, 6))
    ics[:, 0] = x_flat
    ics[:, 1] = y_0
    ics[:, 2] = z_0
    ics[:, 3] = vx_flat
    ics[:, 4] = v_y0
    ics[:, 5] = v_z0

    # 9. Mark points that spill over the energy curve as unphysical
    pos_flat = np.column_stack([x_flat, np.full(n_points, y_0), np.full(n_points, z_0)])
    E_rem_flat = E_0 - potential.potential(pos_flat) - K_fixed
    unphysical_mask = (0.5 * vx_flat**2) > E_rem_flat

    return GridInitialConditions(
        ics=ics,
        unphysical_mask=unphysical_mask,
        x_vals=x_vals,
        v_x_vals=v_x_vals,
        E_rem_vals=E_rem_vals,
    )
