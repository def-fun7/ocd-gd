# 01 — Single Orbit Setup

The smallest possible end-to-end setup: build a basic spheroid potential,
define one orbit's initial conditions, construct a `OrbitChaosDetector`
(which integrates immediately), and inspect its criteria and basic
post-integration state. This example stops short of running chaos
detection itself — see [02 — Single Orbit Chaos Check](02_single_orbit_chaos_check.md)
for that.

```{literalinclude} ../../../examples/01_single_orbit_setup.py
:language: python
:linenos:
```
