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

_CURRENT_UNITS: AgamaUnits | None = None


@dataclass(frozen=True)
class AgamaUnits:
    """The three base astropy Quantities one Agama unit is worth, plus
    every derived unit `OrbitChaosDetector`/`GridChaosDetector` need,
    computed from those three -- never hardcoded separately.

    Construct via `AgamaUnits.from_setup(...)` rather than directly --
    that's also what calls `agama.setUnits(...)`, so "what Agama thinks
    its units are" and "what this class thinks they are" can't drift
    apart.

    Parameters
    ----------
    length : u.Quantity
        The length unit (typically kpc).
    velocity : u.Quantity
        The velocity unit (typically km/s).
    mass : u.Quantity
        The mass unit (typically Msun).

    Examples
    --------
    >>> import astropy.units as u
    >>> from ocd_gd._units import AgamaUnits
    >>> units = AgamaUnits(length=1.0 * u.kpc, velocity=1.0 * (u.km / u.s), mass=1.0 * u.Msun)
    >>> isinstance(units.length, u.Quantity)
    True
    """

    length: u.Quantity
    velocity: u.Quantity
    mass: u.Quantity

    @classmethod
    def from_setup(
        cls, length: float = 1.0, mass: float = 1.0, velocity: float = 1.0
    ) -> AgamaUnits:
        """Call `agama.setUnits(length=length, mass=mass, velocity=velocity)`
        and return the matching `AgamaUnits` (also registered as
        `AgamaUnits.current()` for the rest of this process).

        Arguments have the same meaning as `agama.setUnits`: how many
        kpc / 1e10 Msun / (km/s) one Agama unit equals. The (1, 1, 1)
        default matches every script in this repo.

        Parameters
        ----------
        length : float, default 1.0
            Length unit scale in kpc.
        mass : float, default 1.0
            Mass unit scale in 1e10 Msun.
        velocity : float, default 1.0
            Velocity unit scale in km/s.

        Returns
        -------
        AgamaUnits
            The constructed unit system.

        Examples
        --------
        >>> from ocd_gd._units import AgamaUnits
        >>> units = AgamaUnits.from_setup(length=1.0, mass=1.0, velocity=1.0)
        >>> isinstance(units, AgamaUnits)
        True
        """
        agama.setUnits(length=length, mass=mass, velocity=velocity)
        instance = cls(
            length=length * u.kpc,
            velocity=velocity * (u.km / u.s),
            mass=mass * u.Msun,
        )
        global _CURRENT_UNITS
        _CURRENT_UNITS = instance
        return instance

    @classmethod
    def current(cls) -> AgamaUnits | None:
        """The most recently constructed `AgamaUnits`, or None if
        `from_setup` hasn't been called yet in this process.

        Returns
        -------
        AgamaUnits or None
            The current active unit system, if registered.

        Examples
        --------
        >>> from ocd_gd._units import AgamaUnits
        >>> _ = AgamaUnits.from_setup(length=1.0, mass=1.0, velocity=1.0)
        >>> units = AgamaUnits.current()
        >>> units is not None
        True
        """
        return _CURRENT_UNITS

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

    def unit_for(self, kind: str | None) -> u.Quantity | None:
        """Return the base-unit Quantity for a named `kind` (e.g.
        "length", "energy", "time", "frequency", "angular_momentum",
        "acceleration", "density", "velocity", "mass"), or None for
        `kind=None`/`"dimensionless"` -- e.g. threshold values, fractions,
        grid_size: plain numbers with no physical unit.

        Parameters
        ----------
        kind : str or None
            The name of the physical quantity kind.

        Returns
        -------
        u.Quantity or None
            The matching unit as an astropy Quantity.

        Raises
        ------
        ValueError
            If `kind` is unknown.

        Examples
        --------
        >>> from ocd_gd._units import AgamaUnits
        >>> units = AgamaUnits.from_setup(length=1.0, mass=1.0, velocity=1.0)
        >>> units.unit_for("length")
        <Quantity 1. kpc>
        """
        if kind is None or kind == "dimensionless":
            return None
        if not hasattr(self, kind):
            raise ValueError(f"Unknown unit kind {kind!r}.")
        return getattr(self, kind)

    def to_quantity(self, value: Any, kind: str | None) -> Any:
        """Convert a raw Agama-unit value (float or array) to a physical
        `Quantity`. `kind=None` returns `value` unchanged.

        Parameters
        ----------
        value : Any
            The value or array in Agama units.
        kind : str or None
            The name of the physical quantity kind.

        Returns
        -------
        Any
            The converted astropy Quantity, or the raw value if `kind` is None.

        Examples
        --------
        >>> from ocd_gd._units import AgamaUnits
        >>> units = AgamaUnits.from_setup()
        >>> units.to_quantity(8.0, "length")
        <Quantity 8. kpc>
        """
        unit = self.unit_for(kind)
        return value if unit is None else value * unit

    def to_raw(self, quantity: Any, kind: str | None) -> Any:
        """Convert a physical `Quantity` back to a raw Agama-unit float
        (or array) -- e.g. before passing a user-supplied `R_0` given in
        real kpc into code that calls Agama's C API directly. `kind=None`
        returns `quantity` unchanged (assumed already raw).

        Parameters
        ----------
        quantity : Any
            The astropy Quantity to convert.
        kind : str or None
            The name of the physical quantity kind.

        Returns
        -------
        Any
            The raw numeric value.

        Examples
        --------
        >>> import astropy.units as u
        >>> from ocd_gd._units import AgamaUnits
        >>> units = AgamaUnits.from_setup()
        >>> q = 8.0 * u.kpc
        >>> units.to_raw(q, "length").item()
        8.0
        """
        unit = self.unit_for(kind)
        if unit is None:
            return quantity
        return (quantity / unit).to(u.dimensionless_unscaled).value


def tag_unit(
    units: AgamaUnits | None, name: str, value: Any, lookup: dict[str, str | None]
) -> Any:
    """Attach a physical unit to a raw Agama-unit value, if one is known
    for `name` (via `lookup`) and `units` is given.

    Returns `value` unchanged if either is missing -- e.g. `units` is None
    because `AgamaUnits.from_setup(...)` was never called in this process,
    so the value stays a bare float (or `name` isn't in `lookup`, meaning
    it's dimensionless by convention -- a threshold, count, fraction).

    Parameters
    ----------
    units : AgamaUnits or None
        The active units converter system.
    name : str
        The parameter or column name to look up.
    value : Any
        The raw numerical value or array.
    lookup : dict of str to str or None
        Dictionary mapping names to unit kinds.

    Returns
    -------
    Any
        The unit-tagged value (astropy Quantity) or raw value.

    Examples
    --------
    >>> from ocd_gd._units import AgamaUnits, tag_unit
    >>> units = AgamaUnits.from_setup()
    >>> lookup = {"R_0": "length"}
    >>> tag_unit(units, "R_0", 8.0, lookup)
    <Quantity 8. kpc>
    """
    if units is None:
        return value
    kind = lookup.get(name)
    return units.to_quantity(value, kind) if kind is not None else value
