# Sweep Runner

Runs `GridChaosDetector` once per potential across a batch and stacks the
results into a single tidy `astropy.table.QTable`, one row per run — the
"many runs at once" path that neither `GridChaosDetector` nor
`OrbitChaosDetector` provide on their own. Each potential can be an
already-built `agama.Potential`, a path to a JSON config from
[03 — Composite Potential Builder](03_composite_potential_builder.md), or
a `{"Qb": ..., "f_bh": ...}` dict built fresh via `makeCompositePotential`;
each output row combines `GridChaosDetector.metadata_row()` (grid
geometry, integration settings), per-indicator chaotic-fraction/agreement
stats from `chaos_summary()`, and the potential's own build metadata
(`Q_b`, `f_bh`, component masses). Runs can be checkpointed to disk after
every completion and resumed later, and parallelized across worker
processes via `n_jobs` (process-level parallelism, since each run is
dominated by non-numeric work — agama's C++ integrator, file I/O — that
plain numba can't parallelize the way it does the numeric SALI kernel
inside `OrbitChaosDetector`). Fitting and plotting the resulting table
happens separately, in
[Sweep Analysis](sweep_analysis.md).

```bash
python run_sweep.py --qb 0.1 0.2 --fbh 0.0 0.005 --n-jobs 4 --resume
```

```{literalinclude} ../../../scripts/run_sweep.py
:language: python
:linenos:
```
