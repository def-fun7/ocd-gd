# Sweep Analysis

Reads back the `.ecsv` table written by
[04 — Sweep Runner](04_sweep_runner.md) and fits/plots one column against
another — by default, chaotic fraction (`sali_chaotic_fraction`) against
central-mass fraction (`f_bh`), grouped and colored by bar strength
(`Qb`), with a linear fit overlaid per group and slope/R² reported for
each. Kept as a standalone script rather than folded into the sweep
runner: it only ever touches the saved table, never `agama` or
`GridChaosDetector` directly, so changing which columns to plot, the fit
degree, or the grouping is a matter of rerunning this script against the
same file — no re-integration required.

```bash
python analyze_sweep.py --x f_bh --y sali_chaotic_fraction --group-by Qb
```

```{literalinclude} ../../../scripts/analyze_sweep.py
:language: python
:linenos:
```
