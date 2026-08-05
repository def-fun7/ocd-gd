#!/usr/bin/env python
"""
Build a composite toy Milky-Way-like potential made of three components:
    1. an axisymmetric stellar Disk       (Miyamoto-Nagai)
    2. a triaxial Bar                     (Ferrers ellipsoid, fixed n=2 index)
    3. a central massive object (CBH)     (softened Plummer / point mass)

The two physically meaningful "knobs" of the model are:
    Q_b    - the bar-torque strength (Combes & Sanders 1981; Buta & Block 2001),
             Q_b = max_R[ max_phi(F_phi) / mean_phi(F_R) ]
    f_bh   - the central-mass fraction, Convention B: f_bh = M_bh / M_total,
             where M_total = M_disk + M_bar + M_bh

The Disk and Bar SHAPE (scale lengths, axis ratios) are held fixed and taken
from the idealized Ferrers-bar-in-a-disk models used in the classic
bar-orbit/chaos literature:

    Skokos, C., Patsis, P.A. & Athanassoula, E. (2002)
    "Orbital dynamics of three-dimensional bars I & II", MNRAS 333, 847 & 861

    (see also Pfenniger, D. 1984, A&A 134, 373, for the original
    Ferrers-bar + Miyamoto-Nagai disk + central mass setup)

Only the BAR MASS (which sets Q_b) and the CBH MASS (which sets f_bh) are
varied; everything else is a fixed, cited choice -- treat the exact numeric
defaults below as placeholders to replace with the precise values from
whichever paper/table you finally cite.

Author: <your name>
"""

import argparse
import json
import os
from pathlib import Path

import agama
import numpy
import scipy.optimize
from _cli_common import add_clear_cache_arg, add_qb_fbh_args

from ocd_gd import (
    CorotationSetup,
    omega_for_corotation_ratio,
    get_logger,
    print_banner,
    print_dataframe_table,
    print_kv_table,
    setup_logging,
)

log = get_logger(__name__)

agama.setUnits(
    length=1, mass=1, velocity=1
)  # 1 kpc, 1e10 Msun, 1 km/s (adjust if needed)

# All generated .ini / cache files are written here, next to this script,
# rather than into whatever directory the script happens to be run from.
BASE_DIR = Path(__file__).resolve().parent / "outputs"
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 0. Saving and loading potential params in json
# ----------------------------------------------------------------------------


def save_potential_config(filepath, disk_params, bar_params, bh_params, metadata=None):
    """Save the exact parameters needed to reconstruct the composite potential."""
    config = {
        "metadata": metadata or {},
        "components": {
            "disk": disk_params,
            "bar": bar_params,
            "bh": bh_params,
        },
    }
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)


def load_composite_potential(filepath):
    """Reconstruct the Agama Potential directly from saved JSON parameters."""
    with open(filepath, "r") as f:
        config = json.load(f)

    comps = config["components"]
    disk_pot = agama.Potential(**comps["disk"])
    bar_pot = agama.Potential(**comps["bar"])
    bh_pot = agama.Potential(**comps["bh"])

    composite = agama.Potential(disk_pot, bar_pot, bh_pot)
    return composite, config.get("metadata", {})


# ----------------------------------------------------------------------------
# 1. DISK POTENTIAL
# ----------------------------------------------------------------------------
def makeDiskPotential(mass=8e10, scaleRadius=3.0, scaleHeight=0.3):
    """
    Axisymmetric Miyamoto-Nagai disk, used as the fixed stellar background.

    Default scaleHeight/scaleRadius ~ 0.1 follows the typical thin-disk
    flattening adopted in Skokos, Patsis & Athanassoula (2002).
    All parameters are optional so the geometry can be revisited later
    without touching the calling code.

    Always built fresh in-process from (mass, scaleRadius, scaleHeight) --
    there is nothing to calibrate here, these are direct inputs every call,
    so there is no benefit to caching. The .ini file is still written out
    (best effort) purely for external inspection; it is never read back,
    since Agama's .ini export for MiyamotoNagai does not actually serialize
    its parameters (re-loading it silently falls back to internal defaults,
    e.g. mass=1.0, rather than raising an error -- this is what caused
    M_disk to read back as 1.00 in an earlier run).
    """

    log.info(
        "Building disk potential (mass=%.1f, scaleRadius=%.2f, scaleHeight=%.2f)",
        mass,
        scaleRadius,
        scaleHeight,
    )
    disk_pot_params = {
        "type": "MiyamotoNagai",
        "mass": mass,
        "scaleRadius": scaleRadius,
        "scaleHeight": scaleHeight,
    }
    pot = agama.Potential(**disk_pot_params)
    return pot, disk_pot_params


# ----------------------------------------------------------------------------
# 2. BAR POTENTIAL  (shape fixed, mass calibrated to reach a target Q_b)
# ----------------------------------------------------------------------------
def _barPotential(mass, a, q, qz):
    """
    Build a single Ferrers bar with given mass and shape (internal helper).

    Agama's Ferrers potential parameters (see potential_factory.cpp /
    reference.pdf): mass, scaleRadius (semi-major axis a), axisRatioY (b/a),
    axisRatioZ (c/a). It is a fixed n=2 density-index model -- there is no
    separate shape-index parameter to set.
    """
    bar_pot_params = {
        "type": "Ferrers",
        "mass": mass,
        "scaleRadius": a,
        "axisRatioY": q,
        "axisRatioZ": qz,
    }
    return agama.Potential(**bar_pot_params), bar_pot_params


def _computeQb(pot, radii, nphi=180):
    """
    Compute Q_b for a given potential: at each radius in `radii`, take the
    max-over-azimuth tangential force divided by the azimuthally-averaged
    radial force; Q_b is the max of that ratio over all radii probed.
    (Combes & Sanders 1981; Buta & Block 2001 definition.)
    """
    phi = numpy.linspace(0, 2 * numpy.pi, nphi, endpoint=False)
    cosphi, sinphi = numpy.cos(phi), numpy.sin(phi)
    QT = numpy.zeros(len(radii))
    for i, R in enumerate(radii):
        points = numpy.column_stack([R * cosphi, R * sinphi, numpy.zeros_like(phi)])
        Fx, Fy, _Fz = pot.force(points).T
        F_R = Fx * cosphi + Fy * sinphi  # negative (points inward)
        F_phi = -Fx * sinphi + Fy * cosphi
        QT[i] = numpy.max(numpy.abs(F_phi)) / numpy.abs(numpy.mean(F_R))
    return numpy.max(QT)


def _calibrateBarMass(diskPot, Qb_target, a, q, qz, radii, mass_bracket, max_expand=40):
    """
    Root-find the bar mass whose (disk+bar) Q_b matches Qb_target.

    A fixed (mlo, mhi) bracket is fragile -- the mass needed to reach a
    given Q_b depends on the target itself, the disk mass, and the bar
    shape, so a bracket that works for one Q_b/setup may not bracket the
    root for another (this is exactly the "f(a) and f(b) must have
    different signs" error). Instead, auto-expand the bracket outward
    from the given starting guess until the residual changes sign, then
    hand a *guaranteed* bracket to brentq.
    """

    def residual(mass):
        bar, _ = _barPotential(mass, a, q, qz)
        combined = agama.Potential(diskPot, bar)
        return _computeQb(combined, radii) - Qb_target

    mlo, mhi = mass_bracket
    flo, fhi = residual(mlo), residual(mhi)

    # expand downward if even the smallest mass already overshoots the target
    n = 0
    while flo > 0 and n < max_expand:
        mhi, fhi = mlo, flo
        mlo = mlo / 2.0
        flo = residual(mlo)
        n += 1

    # expand upward if even the largest mass isn't enough to reach the target
    n = 0
    while fhi < 0 and n < max_expand:
        mlo, flo = mhi, fhi
        mhi = mhi * 2.0
        fhi = residual(mhi)
        n += 1

    if flo > 0 or fhi < 0:
        raise RuntimeError(
            f"Could not bracket a root for Q_b={Qb_target:.3f} after expanding to "
            f"mass range [{mlo:.3g}, {mhi:.3g}] (residuals: {flo:.3g}, {fhi:.3g}). "
            "Check the disk mass/shape and target Q_b for consistency."
        )

    log.debug(
        "  bracket for Q_b=%.3f: [%.3g, %.3g] -> residuals [%.3g, %.3g]",
        Qb_target,
        mlo,
        mhi,
        flo,
        fhi,
    )
    return scipy.optimize.brentq(residual, mlo, mhi, xtol=1e-3)


def makeBarPotential(
    Qb,
    diskPot=None,
    a=3.5,
    q=0.4,
    qz=0.2,
    radii=None,
    mass_bracket=(1.0, 600.0),
    filename=None,
):
    """
    Ferrers bar with FIXED shape and mass calibrated so that the (disk+bar)
    system reproduces the requested bar-torque strength Q_b.

    Q_b is the only required argument -- everything else is an optional,
    cited default that can be revisited later:
      a  (semi-major axis / scaleRadius) = 3.5 kpc -- typical bar length
                                                       relative to the disk
                                                       scale length above
      q  = axisRatioY = b/a             = 0.4       -- in-plane axis ratio
      qz = axisRatioZ = c/a             = 0.2       -- vertical flattening
    (shape values follow the Ferrers-bar setups in Skokos, Patsis &
    Athanassoula 2002 / Pfenniger 1984; adjust if you adopt different
    literature values.)

    Caching strategy: the *mass* found for a given Q_b (the expensive part,
    since it requires a bisection search) is cached in a small sidecar
    text file, and the Ferrers potential object is always rebuilt fresh
    in-process from that cached mass -- rather than exported to / re-read
    from an .ini file. This works around an Agama round-trip quirk where
    re-loading a triaxial Ferrers potential from .ini can trip its own
    "0 < q < p < 1" validity check even though the same parameters built
    it successfully in-process. The .ini is still written out (best
    effort) purely for external inspection/reuse outside this script; it
    is never read back here.
    """
    if diskPot is None:
        diskPot = makeDiskPotential()
    if radii is None:
        radii = numpy.linspace(0.2 * a, 1.2 * a, 15)
    if filename is None:
        filename = str(BASE_DIR / (f"bar_potential_Qb{Qb:.3f}.ini"))
    massCacheFile = filename + ".masscache"

    if os.path.exists(massCacheFile):
        with open(massCacheFile) as f:
            mass = float(f.read().strip())
        log.debug("Reusing cached bar mass for Q_b=%.3f: M_bar=%.6f", Qb, mass)
    else:
        log.info("Calibrating bar mass for target Q_b=%.3f ...", Qb)
        mass = _calibrateBarMass(diskPot, Qb, a, q, qz, radii, mass_bracket)
        log.info("  -> calibrated M_bar = %.3f", mass)
        with open(massCacheFile, "w") as f:
            f.write(f"{mass:.10f}")

    pot, bar_pot_params = _barPotential(mass, a, q, qz)
    try:
        pot.export(filename)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not export bar potential to %s (%s); "
            "continuing with the in-memory potential.",
            filename,
            exc,
        )
    return pot, bar_pot_params


# ----------------------------------------------------------------------------
# 3. CENTRAL BLACK HOLE POTENTIAL
# ----------------------------------------------------------------------------
def makeCBHPotential(mass, scaleRadius=1e-4):
    """
    Central massive object as a softened Plummer sphere.

    `mass` is the only required argument (computed by the caller from the
    target f_bh, see makeCompositePotential below). `scaleRadius` is a
    purely numerical smoothing length -- not a physical parameter being
    studied -- kept far smaller than any other scale in the model.

    Always built fresh in-process from (mass, scaleRadius), same reasoning
    as makeDiskPotential above: nothing here is calibrated, so there is no
    need to read anything back from .ini (and, per the disk case, doing so
    is not reliably safe with this Agama version anyway).
    """

    log.info("Building CBH potential (mass=%.4f, scaleRadius=%.1e)", mass, scaleRadius)
    bh_pot_params = {"type": "Plummer", "mass": mass, "scaleRadius": scaleRadius}
    pot = agama.Potential(**bh_pot_params)
    return pot, bh_pot_params


# ----------------------------------------------------------------------------
# FINAL ASSEMBLY: build (Disk, Bar, CBH) for a given (Q_b, f_bh) pair
# ----------------------------------------------------------------------------
def makeCompositePotential(
    Qb, f_bh, diskParams=None, barShape=None, bhScaleRadius=1e-4, outFilename=None
):
    """
    Build the full composite potential for one (Q_b, f_bh) grid point.

    Qb    : target bar-torque strength (required)
    f_bh  : target central-mass fraction, Convention B: f_bh = M_bh / M_total
            (required)
    diskParams : optional dict of overrides passed to makeDiskPotential()
    barShape   : optional dict of overrides (a, q, qz, ...) passed to
                 makeBarPotential() -- everything except Qb itself
    bhScaleRadius : numerical softening length for the CBH

    Returns (potential, filename) where `potential` is the combined
    agama.Potential and `filename` is where it was exported.
    """
    diskParams = diskParams or {}
    barShape = barShape or {}

    diskPot, disk_pot_params = makeDiskPotential(**diskParams)
    barPot, bar_pot_params = makeBarPotential(Qb, diskPot=diskPot, **barShape)

    omega, R_corotation = omega_for_corotation_ratio(
        potential=barPot, a_bar=bar_pot_params["scaleRadius"]
    )

    M_disk = diskPot.totalMass()
    M_bar = barPot.totalMass()
    M_baryon = M_disk + M_bar

    # Convention B: f_bh = M_bh / (M_baryon + M_bh)  =>  M_bh = M_baryon * f_bh/(1-f_bh)
    M_bh = M_baryon * f_bh / (1.0 - f_bh)
    bhPot, bh_pot_params = makeCBHPotential(M_bh, scaleRadius=bhScaleRadius)
    composite = agama.Potential(diskPot, barPot, bhPot)
    if outFilename is None:
        outFilename = str(BASE_DIR / (f"composite_Qb{Qb:.3f}_fbh{f_bh:.4f}.json"))
    metadata = {
        "Qb": f"{Qb:.3f}",
        "f_bh": f"{f_bh:.4f}",
        "M_disk": f"{M_disk:.2f}",
        "M_bar": f"{M_bar:.2f}",
        "M_bh": f"{M_bh:.4f}",
        "M_total": "%.2f" % (M_baryon + M_bh),
        "omega": f"{omega}",
        "R_Corotation": f"{R_corotation}",
        "output file": outFilename,
    }
    save_potential_config(
        outFilename, disk_pot_params, bar_pot_params, bh_pot_params, metadata=metadata
    )

    print_kv_table(
        title=f"Composite potential  (Q_b={Qb:.3f}, f_bh={f_bh:.4f})", data=metadata
    )

    return composite, outFilename


# ----------------------------------------------------------------------------
# CACHE MANAGEMENT
# ----------------------------------------------------------------------------
def clearCache(base_dir=BASE_DIR):
    """
    Delete every generated .ini and .masscache file under base_dir, so the
    next run rebuilds (and recalibrates) everything from scratch.

    Does not touch anything outside base_dir, and does not remove base_dir
    itself -- only the cached files inside it.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        log.info("Cache directory %s does not exist, nothing to clear.", base_dir)
        return

    removed = []
    for pattern in ("*.ini", "*.masscache"):
        for f in base_dir.glob(pattern):
            f.unlink()
            removed.append(f.name)

    if removed:
        log.info("Cleared %d cached file(s) from %s:", len(removed), base_dir)
        for name in sorted(removed):
            log.info("  - %s", name)
    else:
        log.info("No cached files found in %s.", base_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build composite Milky-Way-like potentials for a Q_b x f_bh grid."
    )
    add_qb_fbh_args(parser)
    add_clear_cache_arg(parser)
    return parser


# ----------------------------------------------------------------------------
# MAIN: only runs when this script is executed directly, not on import
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    setup_logging()  # only the top-level script should call this, not the library code above
    print_banner(
        "Composite MW-like potential builder", "Disk + Bar (Ferrers) + CBH grid"
    )

    if args.clear_cache:
        clearCache()

    rows = []
    potentials = {}
    for Qb in args.qb:
        for f_bh in args.fbh:
            pot, fname = makeCompositePotential(Qb, f_bh)
            potentials[(Qb, f_bh)] = pot
            rows.append([Qb, f_bh, fname])

    print_dataframe_table(
        title="Summary of built potentials",
        headers=["Q_b", "f_bh", "file"],
        rows=rows,
    )

    log.info("Built %d composite potentials.", len(potentials))
