"""
Agama <-> astropy unit conversion.

Agama itself is unitless: every float you pass to or get back from its C
API is a bare number in whatever (length, mass, velocity) system was last
set via `agama.setUnits(...)`. This module is the single place that maps
a raw Agama number to a physically-labeled `astropy.units.Quantity` and
back, so the rest of the codebase doesn't have to carry "R_0 is really in
kpc" as a comment -- it's an actual unit on the value.

Matches the convention already used throughout this repo (see the comment
in build_potential.py): by default, 1 Agama length unit = 1 kpc, 1 Agama
velocity unit = 1 km/s, and 1 Agama mass unit = 1e10 Msun -- Agama's own
documented default unit system (a standard galactic-dynamics convention,
NOT derived by fixing G=1).
"""

from __future__ import annotations

__all__ = ["AgamaUnits", "tag_unit"]

from dataclasses import dataclass
from typing import Any

import agama
import astropy.units as u

# The most recently constructed AgamaUnits (via `from_setup`). None until
# `from_setup` has been called at least once in this process -- e.g.
# because a script called `agama.setUnits(...)` directly instead. Used as
# the default by OrbitChaosDetector/GridChaosDetector when no `units=` is
# passed explicitly, so existing code that never touches this module keeps
# working (metadata columns just stay bare floats, as before).
_CURRENT_UNITS: "AgamaUnits | None" = None


@dataclass(frozen=True)
class AgamaUnits:
    """The three base astropy Quantities one Agama unit is worth, plus
    every derived unit `OrbitChaosDetector`/`GridChaosDetector` need,
    computed from those three -- never hardcoded separately.

    Construct via `AgamaUnits.from_setup(...)` rather than directly --
    that's also what calls `agama.setUnits(...)`, so "what Agama thinks
    its units are" and "what this class thinks they are" can't drift
    apart.
    """

    length: u.Quantity
    velocity: u.Quantity
    mass: u.Quantity

    @classmethod
    def from_setup(
        cls, length: float = 1.0, mass: float = 1.0, velocity: float = 1.0
    ) -> "AgamaUnits":
        """Call `agama.setUnits(length=length, mass=mass, velocity=velocity)`
        and return the matching `AgamaUnits` (also registered as
        `AgamaUnits.current()` for the rest of this process).

        Arguments have the same meaning as `agama.setUnits`: how many
        kpc / 1e10 Msun / (km/s) one Agama unit equals. The (1, 1, 1)
        default matches every script in this repo.
        """
        agama.setUnits(length=length, mass=mass, velocity=velocity)
        instance = cls(
            length=length * u.kpc,
            velocity=velocity * (u.km / u.s),
            mass=mass * (1e10 * u.Msun),
        )
        global _CURRENT_UNITS
        _CURRENT_UNITS = instance
        return instance

    @classmethod
    def current(cls) -> "AgamaUnits | None":
        """The most recently constructed `AgamaUnits`, or None if
        `from_setup` hasn't been called yet in this process."""
        return _CURRENT_UNITS

    # ------------------------------------------------------------------
    # Derived units, built from the three base ones above -- never
    # hardcoded, so changing `from_setup`'s arguments updates all of them.
    # ------------------------------------------------------------------
    @property
    def time(self) -> u.Quantity:
        return (self.length / self.velocity).to(u.Gyr)

    @property
    def energy(self) -> u.Quantity:
        """Specific energy (per unit mass) -- what Agama's `E_0`,
        potential values, and orbit energies actually are."""
        return self.velocity**2

    @property
    def angular_momentum(self) -> u.Quantity:
        """Specific angular momentum (per unit mass)."""
        return self.length * self.velocity

    @property
    def frequency(self) -> u.Quantity:
        return 1 / self.time

    @property
    def acceleration(self) -> u.Quantity:
        return self.velocity**2 / self.length

    @property
    def density(self) -> u.Quantity:
        return self.mass / self.length**3

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def unit_for(self, kind: str | None) -> u.Quantity | None:
        """Return the base-unit Quantity for a named `kind` (e.g.
        "length", "energy", "time", "frequency", "angular_momentum",
        "acceleration", "density", "velocity", "mass"), or None for
        `kind=None`/`"dimensionless"` -- e.g. threshold values, fractions,
        grid_size: plain numbers with no physical unit."""
        if kind is None or kind == "dimensionless":
            return None
        if not hasattr(self, kind):
            raise ValueError(f"Unknown unit kind {kind!r}.")
        return getattr(self, kind)

    def to_quantity(self, value: Any, kind: str | None) -> Any:
        """Convert a raw Agama-unit value (float or array) to a physical
        `Quantity`. `kind=None` returns `value` unchanged."""
        unit = self.unit_for(kind)
        return value if unit is None else value * unit

    def to_raw(self, quantity: Any, kind: str | None) -> Any:
        """Convert a physical `Quantity` back to a raw Agama-unit float
        (or array) -- e.g. before passing a user-supplied `R_0` given in
        real kpc into code that calls Agama's C API directly. `kind=None`
        returns `quantity` unchanged (assumed already raw)."""
        unit = self.unit_for(kind)
        if unit is None:
            return quantity
        return (quantity / unit).to(u.dimensionless_unscaled).value


def tag_unit(
    units: "AgamaUnits | None", name: str, value: Any, lookup: dict[str, str | None]
) -> Any:
    """Attach a physical unit to a raw Agama-unit value, if one is known
    for `name` (via `lookup`) and `units` is given.

    Returns `value` unchanged if either is missing -- e.g. `units` is None
    because `AgamaUnits.from_setup(...)` was never called in this process,
    so the value stays a bare float (or `name` isn't in `lookup`, meaning
    it's dimensionless by convention -- a threshold, count, fraction).
    """
    if units is None:
        return value
    kind = lookup.get(name)
    return units.to_quantity(value, kind) if kind is not None else value
