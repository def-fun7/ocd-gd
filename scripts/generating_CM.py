from datetime import datetime
import numpy as np
import agama

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocd_gd.orbit_detector import OrbitChaosDetector

# Ensure units match your AGAMA setup (e.g., kpc, km/s, Msun)
agama.setUnits(length=1, mass=1, velocity=1)


def plot_chaos_maps(
    sali_grid,
    gali_grid,
    lyapunov_grid,
    x_vals,
    v_x_vals,
    E_rem_vals=None,
    cmap_colors=["#1f4e78", "#f2c811"],  # [Regular (0), Chaotic (1)]
    masked_color="#333333",  # Dark gray for unphysical NaN regions
):
    """
    Plots SALI, GALI, and Lyapunov 2D chaos maps side-by-side (1x3) using a
    clean, unified legend instead of a colorbar.
    """
    grid_size_y, grid_size_x = sali_grid.shape
    extent = [x_vals[0], x_vals[-1], v_x_vals[0], v_x_vals[-1]]

    # 1. Build discrete 2-color map handling masked NaNs
    cmap = ListedColormap(cmap_colors)
    cmap.set_bad(color=masked_color)

    # 2. Setup 1x3 Subplot Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)

    if E_rem_vals is not None:
        v_zvc = np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0))

    panel_configs = [
        (axes[0], sali_grid, f"SALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (axes[1], gali_grid, f"GALI Chaos Map ({grid_size_x}x{grid_size_y})"),
        (axes[2], lyapunov_grid, f"Lyapunov Map ({grid_size_x}x{grid_size_y})"),
    ]

    for ax, grid, title in panel_configs:
        ax.imshow(
            grid,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        # Overlay Zero-Velocity Curve (ZVC)
        if E_rem_vals is not None:
            ax.plot(x_vals, v_zvc, color="red", linestyle="--", linewidth=1.5)
            ax.plot(x_vals, -v_zvc, color="red", linestyle="--", linewidth=1.5)

        ax.set_xlabel("$x$", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    axes[0].set_ylabel("$v_x$", fontsize=12)

    # 3. Create Custom Legend Elements (Including Unphysical Domain Patch)
    legend_elements = [
        Patch(facecolor=cmap_colors[0], edgecolor="none", label="Regular Orbit"),
        Patch(facecolor=cmap_colors[1], edgecolor="none", label="Chaotic Orbit"),
        Patch(facecolor=masked_color, edgecolor="none", label="Unphysical Domain"),
    ]

    if E_rem_vals is not None:
        legend_elements.append(
            Line2D([0], [0], color="red", lw=1.5, ls="--", label="Zero-Velocity Curve")
        )

    # 4. Attach Unified Legend to the Rightmost Subplot (Panel 3)
    axes[2].legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.tight_layout()
    plt.show()


def plot_composite_chaos_map(
    sali_grid,
    gali_grid,
    lyapunov_grid,
    x_vals,
    v_x_vals,
    E_rem_vals=None,
    masked_color=(0.2, 0.2, 0.2),  # Dark gray (RGB) for unphysical NaN regions
):
    """
    Overlays SALI, GALI, and Lyapunov indicators into a single RGB composite chaos map.

    Channel Mapping:
    - Red Channel   : Lyapunov Exponent
    - Green Channel : GALI
    - Blue Channel  : SALI
    """
    grid_shape = sali_grid.shape
    extent = [x_vals[0], x_vals[-1], v_x_vals[0], v_x_vals[-1]]

    # 1. Create a Mask for unphysical/NaN points
    # (Assuming NaNs indicate masked regions; adjust condition if using a separate mask)
    nan_mask = np.isnan(sali_grid) | np.isnan(gali_grid) | np.isnan(lyapunov_grid)

    # 2. Build 3D RGB array initialized with background color (Dark Blue for Regular)
    # Regular base color: #1f4e78 -> RGB (0.12, 0.31, 0.47)
    rgb_map = np.ones((*grid_shape, 3)) * np.array([0.12, 0.31, 0.47])

    # 3. Clean grids by filling NaNs with 0 for channel math
    s_clean = np.nan_to_num(sali_grid, nan=0)
    g_clean = np.nan_to_num(gali_grid, nan=0)
    l_clean = np.nan_to_num(lyapunov_grid, nan=0)

    # 4. Construct RGB blend where indicators trigger (value = 1)
    # Red = Lyapunov, Green = GALI, Blue/Yellow highlight = SALI
    # Adjusting channel intensity weights for distinct contrast:
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if nan_mask[i, j]:
                rgb_map[i, j] = masked_color
            else:
                l, g, s = l_clean[i, j], g_clean[i, j], s_clean[i, j]
                if l or g or s:
                    # R, G, B channel assignment for chaotic detection overlaps
                    r = 0.9 if l else 0.2
                    g_val = 0.8 if g else 0.2
                    b = 0.9 if s else 0.1
                    rgb_map[i, j] = [r, g_val, b]

    # 5. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(
        rgb_map,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    # Overlay Zero-Velocity Curve (ZVC)
    if E_rem_vals is not None:
        v_zvc = np.sqrt(2.0 * np.maximum(E_rem_vals, 0.0))
        ax.plot(x_vals, v_zvc, color="red", linestyle="--", linewidth=1.5)
        ax.plot(x_vals, -v_zvc, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$v_x$", fontsize=12)
    ax.set_title(
        f"Composite Chaos Overlay ({grid_shape[1]}x{grid_shape[0]})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # 6. Custom Legend Explaining Overlaps
    legend_elements = [
        Patch(facecolor=(0.12, 0.31, 0.47), label="Regular Orbit (All 0)"),
        Patch(facecolor=(0.9, 0.8, 0.9), label="All Indicators Agree (1,1,1)"),
        Patch(facecolor=(0.2, 0.8, 0.1), label="GALI Only"),
        Patch(facecolor=(0.9, 0.2, 0.1), label="Lyapunov Only"),
        Patch(facecolor=masked_color, label="Unphysical Domain"),
    ]

    if E_rem_vals is not None:
        legend_elements.append(
            Line2D([0], [0], color="red", lw=1.5, ls="--", label="Zero-Velocity Curve")
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        facecolor="#ffffff",
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.show()


def generate_ics_from_agama(
    potential,
    E_0,
    y_0,
    z_0,
    v_y0,
    v_z0,
    x_search_range=(-10.0, 10.0),
    grid_size=10,
    search_resolution=1000,
):
    """
    Generates an (N_grid^2, 6) IC matrix constrained by total energy E_0
    using an AGAMA Potential object.
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

    # 2. Cell-centered v_x grid (prevents exact 0.0)
    dvx = (2.0 * v_x_max_global) / grid_size
    v_x_vals = np.linspace(
        -v_x_max_global + dvx / 2.0, v_x_max_global - dvx / 2.0, grid_size
    )

    # 6. Build 2D Meshgrid and flatten
    X, VX = np.meshgrid(x_vals, v_x_vals)
    x_flat = X.ravel()
    vx_flat = VX.ravel()
    n_points = len(x_flat)
    x_flat[x_flat == 0.0] = 1e-5
    vx_flat[vx_flat == 0.0] = 1e-5

    # 7. Stack into (100, 6) IC array
    ics = np.zeros((n_points, 6))
    ics[:, 0] = x_flat
    ics[:, 1] = y_0
    ics[:, 2] = z_0
    ics[:, 3] = vx_flat
    ics[:, 4] = v_y0
    ics[:, 5] = v_z0

    # 8. Mark points that spill over energy curve as unphysical
    pos_flat = np.column_stack([x_flat, np.full(n_points, y_0), np.full(n_points, z_0)])
    E_rem_flat = E_0 - potential.potential(pos_flat) - K_fixed
    unphysical_mask = (0.5 * vx_flat**2) > E_rem_flat

    return ics, unphysical_mask, (x_vals, v_x_vals), E_rem_vals


MW_POTENTIAL_PATH = "data/potentials/MWPotentialHunter24_full.ini"

potential = agama.Potential(MW_POTENTIAL_PATH)
# potential = agama.Potential(type="Plummer", mass=1e11, scaleRadius=5.0)

# 1. Pick a reference orbit point at r = 8.0 kpc
r_ref = 8.0
pos_ref = np.array([[r_ref, 10.0, 10.0]])

# 2. Derive E_0 from a circular velocity orbit
# Force vector grad(Phi) -> F = -grad(Phi)
force = potential.force(pos_ref)[0]
v_circ = np.sqrt(r_ref * np.abs(force[0]))

# Total Energy E_0 = Phi(r) + 0.5 * v_circ^2
E_0 = potential.potential(pos_ref)[0] + 0.5 * v_circ**2
grid_size = 10

ics, mask, (x_grid, vx_grid), E_rem_vals = generate_ics_from_agama(
    potential=potential,
    E_0=E_0,
    y_0=10.0,
    z_0=10.0,
    v_y0=0.6 * v_circ,
    v_z0=0.0,
    grid_size=grid_size,
)
st = datetime.now()
detector = OrbitChaosDetector(ic=ics, pot=potential)
end = datetime.now()

print("for integrating:", end - st)
st = datetime.now()
summary = detector.detect_chaos()
end = datetime.now()

print("for detecting:", end - st)
sali_check, gali_check, lyap_check = (
    summary.sali_check,
    summary.gali_check,
    summary.lyapunov_check,
)
print(sum(lyap_check), sum(gali_check), sum(sali_check))

sali_check = sali_check.astype(float)
sali_check[mask] = np.nan
gali_check = gali_check.astype(float)
gali_check[mask] = np.nan
lyap_check = lyap_check.astype(float)
lyap_check[mask] = np.nan
sali_grid, gali_grid, lyap_grid = (
    sali_check.reshape(grid_size, grid_size).T,
    gali_check.reshape(grid_size, grid_size).T,
    lyap_check.reshape(grid_size, grid_size).T,
)

plot_composite_chaos_map(
    sali_grid=sali_grid,
    gali_grid=gali_grid,
    lyapunov_grid=lyap_grid,
    x_vals=x_grid,
    v_x_vals=vx_grid,
    E_rem_vals=E_rem_vals,
)
plot_chaos_maps(
    sali_grid=sali_grid,
    gali_grid=gali_grid,
    lyapunov_grid=lyap_grid,
    x_vals=x_grid,
    v_x_vals=vx_grid,
    E_rem_vals=E_rem_vals,
)
