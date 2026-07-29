import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import agama

# ==============================================================================
# 1. SETUP UNITS & DUMMY INI CREATOR (FOR TESTING PURPOSES)
# ==============================================================================
# AGAMA default units: length = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(length=1, velocity=1, mass=1)

INI_FILENAME = "./data/potentials/MWPotentialHunter24_full.ini"

# Create a sample .ini file if one doesn't exist so the script runs out-of-the-box
if not os.path.exists(INI_FILENAME):
    print(
        f"[{INI_FILENAME}] not found. Creating a sample Plummer + Miyamoto-Nagai potential file..."
    )
    ini_content = """
; Sample AGAMA Potential Configuration File
[Component 1]
type = Plummer
mass = 1.0e10
scaleRadius = 0.5

[Component 2]
type = MiyamotoNagai
mass = 5.0e10
scaleRadius = 3.0
scaleHeight = 0.3
"""
    with open(INI_FILENAME, "w") as f:
        f.write(ini_content)

# ==============================================================================
# 2. LOAD POTENTIAL FROM INI FILE
# ==============================================================================
print(f"Loading potential from '{INI_FILENAME}'...")
pot = agama.Potential(INI_FILENAME)

# ==============================================================================
# 3. DEFINE EVALUATION GRIDS
# ==============================================================================
grid_range = 10.0  # Physical extent (e.g., -10 to +10 kpc)
num_points = 200  # Grid resolution per axis

x = np.linspace(-grid_range, grid_range, num_points)
y = np.linspace(-grid_range, grid_range, num_points)
z = np.linspace(-grid_range, grid_range, num_points)

X_xy, Y_xy = np.meshgrid(x, y)
X_xz, Z_xz = np.meshgrid(x, z)

# Build (N, 3) coordinate arrays [x, y, z] for AGAMA evaluation
pos_xy = np.column_stack(
    [X_xy.ravel(), Y_xy.ravel(), np.zeros_like(X_xy).ravel()]
)  # z = 0
pos_xz = np.column_stack(
    [X_xz.ravel(), np.zeros_like(X_xz).ravel(), Z_xz.ravel()]
)  # y = 0

# 1D Radial array along the positive x-axis
r = np.linspace(0.01, grid_range, 300)
pos_1d = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])

# ==============================================================================
# 4. EVALUATE POTENTIAL VALUES
# ==============================================================================
Phi_xy = pot.potential(pos_xy).reshape(X_xy.shape)
Phi_xz = pot.potential(pos_xz).reshape(X_xz.shape)
Phi_1d = pot.potential(pos_1d)

# Find absolute potential minimum on the z=0 plane (bottom of the well)
min_idx = np.argmin(Phi_xy.ravel())
x_min = pos_xy[min_idx, 0]
y_min = pos_xy[min_idx, 1]

# ==============================================================================
# 5. VISUALIZATION DASHBOARD
# ==============================================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    f"AGAMA Potential Explorer: [{INI_FILENAME}]", fontsize=16, fontweight="bold"
)

# --- Plot 1: Equatorial Plane (x-y) Contours ---
ax1 = fig.add_subplot(2, 2, 1)
cf1 = ax1.contourf(X_xy, Y_xy, Phi_xy, levels=30, cmap="viridis")
fig.colorbar(cf1, ax=ax1, label=r"Potential $\Phi(x, y, 0)$")
c1 = ax1.contour(
    X_xy, Y_xy, Phi_xy, levels=12, colors="white", linewidths=0.5, alpha=0.6
)
ax1.clabel(c1, inline=True, fontsize=8, fmt="%.1e")
ax1.plot(x_min, y_min, "r*", markersize=12, label="Potential Minimum")
ax1.set_xlabel("x [kpc]")
ax1.set_ylabel("y [kpc]")
ax1.set_title("Equatorial Plane ($z=0$)")
ax1.legend(loc="upper right")
ax1.set_aspect("equal")

# --- Plot 2: Meridional Plane (x-z) Contours ---
ax2 = fig.add_subplot(2, 2, 2)
cf2 = ax2.contourf(X_xz, Z_xz, Phi_xz, levels=30, cmap="magma")
fig.colorbar(cf2, ax=ax2, label=r"Potential $\Phi(x, 0, z)$")
c2 = ax2.contour(
    X_xz, Z_xz, Phi_xz, levels=12, colors="white", linewidths=0.5, alpha=0.6
)
ax2.clabel(c2, inline=True, fontsize=8, fmt="%.1e")
ax2.set_xlabel("x [kpc]")
ax2.set_ylabel("z [kpc]")
ax2.set_title("Meridional Plane ($y=0$)")
ax2.set_aspect("equal")

# --- Plot 3: 1D Radial Well Profile ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(r, Phi_1d, color="navy", lw=2, label=r"$\Phi(r)$")
ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax3.set_xlabel("Radius r [kpc]")
ax3.set_ylabel(r"Potential $\Phi(r)$")
ax3.set_title("1D Radial Potential Profile")
ax3.grid(True, linestyle=":", alpha=0.7)
ax3.legend()

# --- Plot 4: 3D Surface View of the Equatorial Well ---
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
surf = ax4.plot_surface(
    X_xy, Y_xy, Phi_xy, cmap="viridis", edgecolor="none", alpha=0.85
)
ax4.set_xlabel("x [kpc]")
ax4.set_ylabel("y [kpc]")
ax4.set_zlabel(r"$\Phi$")
ax4.set_title("3D Potential Well Surface ($z=0$)")
ax4.view_init(elev=35, azim=-45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
