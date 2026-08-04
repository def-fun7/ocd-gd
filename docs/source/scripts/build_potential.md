# Composite Potential Builder

Builds the toy Milky-Way-like potential (Miyamoto-Nagai disk + Ferrers bar

- central massive object) that every chaos-detection example downstream
  runs orbits through. The bar mass is calibrated by root-finding until the
  disk+bar system reaches a target bar-torque strength `Q_b`; the CBH mass is
  derived from a target central-mass fraction `f_bh`. Calibrated bar masses
  are cached to disk so repeated builds at the same `Q_b` skip the
  root-find. Run directly (`python build_potential.py --qb 0.1 0.2 --fbh 0.0
0.005`) to write a grid of composite-potential JSON configs; see
  [Sweep Runner](run_sweep.md) for feeding that grid into
  `GridChaosDetector` runs.

```{literalinclude} ../../../scripts/build_potential.py
:language: python
:linenos:
```
