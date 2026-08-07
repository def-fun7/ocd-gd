#!/usr/bin/env python
"""
Sweep runner for GridChaosDetector.

Takes a list of potentials -- either already-built `agama.Potential`
objects, paths to composite-potential JSON configs written by
`build_potential.save_potential_config`, or dicts of
`{"Qb": ..., "f_bh": ...}` build parameters -- plus one shared set of fixed
GridChaosDetector settings (R_0, grid_size, energy, etc.), runs a
`GridChaosDetector` for each potential, and collects the results into a
single tidy `astropy.table.QTable` (one row per run).

This is the missing "many runs at once" code path: `GridChaosDetector`
itself only ever describes a single (potential, grid-settings) run, and
`GridChaosDetector.metadata_row()` only ever returns a single row. Fitting
something like "chaotic fraction vs. BH mass" needs many such rows stacked
together with their build parameters attached -- that's what `run_sweep`
below produces. Plotting/fitting against the result lives in a separate
script, `analyze_sweep.py`, deliberately kept apart: it only ever reads the
`.ecsv` this one writes, so changing a fit or a plot never means
re-running the (expensive) integrations.

Each output row is `GridChaosDetector.metadata_row()` (R_0, E_0, grid
geometry, etc.) extended with:
  - per-indicator chaotic-fraction/agreement stats from `chaos_summary()`,
  - whatever metadata came bundled with the potential (Q_b, f_bh, disk/bar/bh
    masses, source file, ...) if it was loaded from a JSON config or built
    fresh here,
  - any per-run `label` overrides passed in by the caller,
  - a `run_index` and `wall_time_s` for bookkeeping.

On parallelism: this does NOT use numba. Numba's `@njit(parallel=True)` /
`prange` accelerates a tight numeric loop *inside one jitted function* --
that's what `sali_kernel` (in `orbit_detector.py`) already uses it for. A
sweep run is the opposite shape of problem: each one is dominated by
non-numeric work numba can't touch in nopython mode -- calling into
agama's C++ orbit integrator, file I/O for cached bar masses / JSON
configs, logging -- with no shared numeric array to loop over *across*
runs. What does fit is plain process-level parallelism, since sweep runs
are otherwise fully independent (different potentials, no shared state):
`run_sweep(..., n_jobs=N)` uses `concurrent.futures.ProcessPoolExecutor`
for that.

Usage:
    python run_sweep.py --qb 0.1 0.2 --fbh 0.0 0.005 --n-jobs 4
    python run_sweep.py --help
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import agama
import numpy as np
import matplotlib

matplotlib.use("Agg")
from _cli_common import add_clear_cache_arg, add_qb_fbh_args
from astropy.table import QTable, vstack
from build_potential import clearCache, load_composite_potential, makeCompositePotential

from ocd_gd import (
    AgamaUnits,
    tag_unit,
    GridChaosDetector,
    get_logger,
    print_banner,
    print_dataframe_table,
    print_kv_table,
    setup_logging,
    set_publication_style,
)

CURRENT_UNITS = AgamaUnits.from_setup(length=1, mass=1, velocity=1)

_POTENTIAL_METADATA_UNITS: dict[str, str | None] = {
    "M_disk": "mass",
    "M_bar": "mass",
    "M_bh": "mass",
    "M_total": "mass",
}

log = get_logger(__name__)
# set_publication_style()
BASE_DIR = Path(__file__).resolve().parent / "outputs"
BASE_DIR.mkdir(parents=True, exist_ok=True)


# A single sweep entry's potential can be given in any of these forms; see
# `_resolve_potential` for how each is turned into (agama.Potential, dict).
PotentialSpec = agama.Potential | str | Path | dict


@dataclass
class SweepRun:
    """One entry in a sweep.

    Parameters
    ----------
    potential : PotentialSpec
        Either an already-built `agama.Potential`, a path to a JSON config
        written by `build_potential.save_potential_config` (loaded via
        `load_composite_potential`), or a dict `{"Qb": ..., "f_bh": ..., ...}`
        of build parameters passed straight to
        `build_potential.makeCompositePotential` (any extra keys besides
        "Qb"/"f_bh" are forwarded as its `diskParams`/`barShape`/etc kwargs).
        NOTE: for `run_sweep(..., n_jobs > 1)` this must be a path or a
        dict, not a live `agama.Potential` -- see `run_sweep`.
    label : dict, optional
        Extra columns to attach to this run's output row, merged in on top
        of (and overriding) any metadata that came with the potential itself
        -- e.g. `{"run_name": "fiducial"}`.
    grid_overrides : dict, optional
        Per-run overrides to the sweep's shared `grid_kwargs`, for e.g. one
        potential that needs a different `R_0` or `grid_size` than the rest.
    """

    potential: PotentialSpec
    label: dict[str, Any] = field(default_factory=dict)
    grid_overrides: dict[str, Any] = field(default_factory=dict)


def _normalize_potential_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Turn the string-valued metadata block written by
    `build_potential.save_potential_config` (e.g. `{"composite_Qb": "0.100",
    "f_bh": "0.0000", "M_bh": "0.0000", ...}`) into float-valued sweep
    columns with consistent names, so
    downstream analysis (`analyze_sweep.py`) doesn't need to know about
    that formatting quirk.
    """
    rename = {"R_Corotation": "R_0"}
    skip = {"output file"}
    corotation_keys = {"omega", "R_Corotation"}

    corotation_spec: dict[str, float] = {}
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in skip:
            continue
        name = rename.get(key, key)
        try:
            value = float(value)
            value = tag_unit(
                AgamaUnits.current(), name, value, _POTENTIAL_METADATA_UNITS
            )
        except (TypeError, ValueError):
            pass
        if key in corotation_keys:
            corotation_spec[name] = value
        else:
            out[name] = value
    return out, corotation_spec


def _resolve_potential(
    spec: PotentialSpec,
) -> tuple[agama.Potential, dict[str, Any], dict[str, float]]:
    """Turn one `PotentialSpec` into `(agama.Potential, extra_metadata)`.

    `extra_metadata` is whatever's known about the potential's construction
    beyond what `GridChaosDetector.metadata_row()` itself records (Q_b,
    f_bh, component masses, source file, ...) -- empty for a bare
    `agama.Potential` passed in directly, since nothing about how it was
    built is recoverable from the object alone.
    """
    if isinstance(spec, agama.Potential):
        return spec, {}, {}

    if isinstance(spec, (str, Path)):
        pot, metadata = load_composite_potential(str(spec))
        extra, corotationSpec = _normalize_potential_metadata(metadata)
        extra["potential_source"] = str(spec)
        return pot, extra, corotationSpec

    if isinstance(spec, dict):
        if "Qb" not in spec or "f_bh" not in spec:
            raise ValueError(
                "A dict potential spec must have 'Qb' and 'f_bh' keys "
                f"(got keys: {sorted(spec.keys())})."
            )
        build_kwargs = {k: v for k, v in spec.items() if k not in ("Qb", "f_bh")}
        pot, fname = makeCompositePotential(spec["Qb"], spec["f_bh"], **build_kwargs)
        # Re-load the metadata makeCompositePotential just wrote, rather than
        # hand-assembling a smaller dict here -- keeps this branch's columns
        # (M_disk/M_bar/M_bh/M_total/...) identical to the path-spec branch.
        _, metadata = load_composite_potential(fname)
        extra, corotationSpec = _normalize_potential_metadata(metadata)
        extra["potential_source"] = fname
        return pot, extra, corotationSpec

    raise TypeError(
        "potential must be an agama.Potential, a path to a JSON config, or "
        f"a dict of {{'Qb', 'f_bh', ...}} build params -- got {type(spec)!r}."
    )


def _chaos_fraction_columns(chaos_summary: Any) -> dict[str, Any]:
    """Flatten a `ChaosSurveySummary` (from `OrbitChaosDetector.chaos_summary()`)
    into scalar columns suitable for a metadata row.

    Only scalar stats are surfaced here -- per-indicator `MethodChaosStats`
    also carries `chaotic_indices`/`regular_indices`/`chaotic_ics`/
    `regular_ics`, which are batch-sized (one entry per orbit) and don't
    belong in a one-row-per-run sweep table. Call `detector.chaos_summary()`
    directly on an individual run if you need those.
    """
    cols: dict[str, Any] = {}
    for name in ("sali", "gali", "lyapunov"):
        stats = getattr(chaos_summary, name)
        cols[f"{name}_n_chaotic"] = stats.n_chaotic
        cols[f"{name}_n_regular"] = stats.n_regular
        cols[f"{name}_chaotic_fraction"] = stats.chaotic_fraction

    agreement = chaos_summary.agreement
    cols["sali_gali_agreement"] = agreement.sali_gali_agreement
    cols["sali_lyapunov_agreement"] = agreement.sali_lyapunov_agreement
    cols["gali_lyapunov_agreement"] = agreement.gali_lyapunov_agreement
    cols["all_agree_chaotic"] = agreement.all_agree_chaotic
    cols["all_agree_regular"] = agreement.all_agree_regular
    cols["disagreement"] = agreement.disagreement
    cols["n_undetermined"] = agreement.n_undetermined
    return cols


def _execute_run(
    i: int,
    run: SweepRun,
    grid_kwargs: dict[str, Any],
    chaos_map_dir: Path | None = None,
) -> tuple[int, QTable | None, dict[str, Any] | None]:
    """Run one sweep entry to completion: resolve its potential, build the
    `GridChaosDetector`, and produce its metadata row.

    Module-level (not a closure/method), and returns plain picklable data
    (a QTable row or an error dict) rather than raising, so it can also be
    submitted to a `ProcessPoolExecutor` by `run_sweep(..., n_jobs > 1)`.
    """
    try:
        pot, meta_from_spec, corotationSpec = _resolve_potential(run.potential)
        merged_grid_kwargs = {**grid_kwargs, **run.grid_overrides, **corotationSpec}
        t0 = time.perf_counter()
        detector = GridChaosDetector(pot, **merged_grid_kwargs)
        chaos_summary = detector.chaos_summary()
        extra = {
            **meta_from_spec,
            **_chaos_fraction_columns(chaos_summary),
            **run.label,
        }
        if chaos_map_dir is not None:
            chaos_map_dir.mkdir(parents=True, exist_ok=True)
            original_path = Path(extra["potential_source"])
            cons_chaos_map_filename = (
                original_path.stem.replace("composite", "cons_chaos_map") + ".png"
            )
            comp_chaos_map_filename = (
                original_path.stem.replace("composite", "comp_chaos_map") + ".png"
            )
            side_chaos_map_filename = (
                original_path.stem.replace("composite", "side_chaos_map") + ".png"
            )
            detector.save_chaos_maps(
                composite_path=f"{chaos_map_dir}/{comp_chaos_map_filename}",
                consensus_path=f"{chaos_map_dir}/{cons_chaos_map_filename}",
                side_by_side_path=f"{chaos_map_dir}/{side_chaos_map_filename}",
                theme="magma",
            )
        elapsed = time.perf_counter() - t0

        extra["run_index"] = i
        extra["wall_time_s"] = elapsed
        return i, detector.metadata_row(extra=extra), None

    except Exception as exc:  # noqa: BLE001
        return (
            i,
            None,
            {
                "run_index": i,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def _init_worker_logging() -> None:
    """`ProcessPoolExecutor` initializer: each worker starts with a fresh,
    unconfigured logger, so set it up the same way the main process did
    (needed on spawn-based platforms, e.g. macOS/Windows, where workers
    don't inherit the parent's logging config the way fork does)."""
    setup_logging()


def _write_checkpoint(rows: list[QTable], checkpoint_path: Path) -> None:
    table = vstack(rows, metadata_conflicts="silent")
    if "run_index" in table.colnames:
        table.sort("run_index")
    table.write(checkpoint_path, format="ascii.ecsv", overwrite=True)


def run_sweep(
    runs: Sequence[SweepRun | PotentialSpec],
    grid_kwargs: dict[str, Any] | None = None,
    chaos_map_dir: str | Path | None = None,
    on_error: str = "raise",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    n_jobs: int = 1,
) -> QTable:
    """Run `GridChaosDetector` once per entry in `runs` and stack the
    resulting metadata rows into one `QTable`.

    Parameters
    ----------
    runs : sequence of SweepRun or PotentialSpec
        The potentials to sweep over. Bare `PotentialSpec` entries (not
        wrapped in `SweepRun`) are treated as `SweepRun(potential=entry)`
        with no label/grid overrides.
    grid_kwargs : dict, optional
        Keyword arguments forwarded to every `GridChaosDetector(...)` call
        -- the fixed grid/integration settings shared across the whole
        sweep (R_0, grid_size, iter_time, sali_threshold, ...). Per-run
        `SweepRun.grid_overrides` are merged on top of these for that run.
    on_error : {"raise", "skip", "collect"}, default "raise"
        What to do if building or running a single sweep entry raises:
        "raise" stops the sweep immediately; "skip" logs and drops the
        run; "collect" logs, drops the row, and records the error
        (accessible via `table.meta["errors"]`) instead of raising.
    checkpoint_path : str or Path, optional
        If given, the accumulated table is rewritten to this path (ECSV)
        after *every* completed run -- not just at the end -- so a crash
        or Ctrl-C partway through a long sweep still leaves a valid,
        loadable table of whatever finished. Each rewrite is O(runs so
        far), which adds real overhead for very large sweeps; the
        alternative (writing once at the end) trades that for losing
        everything on a mid-sweep crash, which given `agama.orbit()` is
        the actual long pole here is usually the worse trade.
    resume : bool, default False
        If True and `checkpoint_path` already exists, load it first and
        skip any run whose `run_index` (its position in `runs`) is already
        present -- so a sweep interrupted partway through can be restarted
        with the same `runs` list instead of recomputing finished entries.
    n_jobs : int, default 1
        Number of worker processes. 1 (default) runs serially in-process.
        >1 uses `concurrent.futures.ProcessPoolExecutor` -- sweep runs are
        independent (different potentials, no shared state), so this is
        plain process-level parallelism, *not* numba (see the module
        docstring for why numba doesn't fit this). Every
        `SweepRun.potential` must be a path or a `{"Qb", "f_bh"}` dict
        when `n_jobs > 1`, not a live `agama.Potential` -- building the
        detector happens inside the worker process, and agama's C++
        potential objects generally can't be pickled across the process
        boundary.

    Returns
    -------
    QTable
        One row per successful run: `GridChaosDetector.metadata_row()`
        columns, per-indicator chaotic-fraction/agreement columns (see
        `_chaos_fraction_columns`), the potential's own metadata (if any),
        the run's `label`, `run_index`, and `wall_time_s`. Sorted by
        `run_index`. Rows whose columns don't exactly match other rows are
        stacked with those cells masked, not dropped.
    """
    if on_error not in ("raise", "skip", "collect"):
        raise ValueError(f"Unknown on_error={on_error!r}")

    grid_kwargs = dict(grid_kwargs or {})
    resolved_runs = [
        entry if isinstance(entry, SweepRun) else SweepRun(potential=entry)
        for entry in runs
    ]

    if n_jobs > 1:
        for run in resolved_runs:
            if isinstance(run.potential, agama.Potential):
                raise TypeError(
                    "n_jobs > 1 requires every SweepRun.potential to be a path "
                    "or a {'Qb', 'f_bh'} dict, not a live agama.Potential "
                    "(got one that is) -- see run_sweep's docstring."
                )

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    rows: list[QTable] = []
    errors: list[dict[str, Any]] = []
    completed_indices: set[int] = set()

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        existing = QTable.read(checkpoint_path, format="ascii.ecsv")
        rows.append(existing)
        completed_indices = {int(v) for v in existing["run_index"]}
        log.info(
            "Resuming from %s: %d run(s) already completed.",
            checkpoint_path,
            len(completed_indices),
        )

    pending = [
        (i, run) for i, run in enumerate(resolved_runs) if i not in completed_indices
    ]
    n_total = len(resolved_runs)

    def _handle_result(
        i: int, row: QTable | None, error: dict[str, Any] | None
    ) -> None:
        """Apply on_error policy + checkpointing to one run's result."""
        if error is not None:
            log.error("Sweep run %d/%d failed: %s", i + 1, n_total, error["error"])
            if on_error == "raise":
                raise RuntimeError(
                    f"Sweep run {i} failed: {error['error']}\n{error['traceback']}"
                )
            errors.append(error)
            return

        rows.append(row)
        log.info("Sweep run %d/%d finished.", i + 1, n_total)
        if checkpoint_path is not None:
            _write_checkpoint(rows, checkpoint_path)

    if n_jobs <= 1:
        for i, run in pending:
            log.info("Sweep run %d/%d: building GridChaosDetector ...", i + 1, n_total)
            _, row, error = _execute_run(i, run, grid_kwargs, chaos_map_dir)
            _handle_result(i, row, error)
    else:
        log.info(
            "Running %d pending sweep entries across %d worker process(es) ...",
            len(pending),
            n_jobs,
        )
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker_logging
        ) as pool:
            futures = [
                pool.submit(_execute_run, i, run, grid_kwargs, chaos_map_dir)
                for i, run in pending
            ]
            for future in as_completed(futures):
                i, row, error = future.result()
                _handle_result(i, row, error)

    if not rows:
        raise RuntimeError("Sweep produced no successful runs -- see log output above.")

    table = vstack(rows, metadata_conflicts="silent")
    if "run_index" in table.colnames:
        table.sort("run_index")
    table.meta["errors"] = errors
    table.meta["n_requested"] = n_total
    table.meta["n_succeeded"] = len(table)
    return table


def qb_fbh_grid_runs(
    Qb_values: Sequence[float], fbh_values: Sequence[float]
) -> list[SweepRun]:
    """Convenience builder: one `SweepRun` per (Q_b, f_bh) pair in the outer
    product of `Qb_values` x `fbh_values`, each built fresh via
    `makeCompositePotential`."""
    return [
        SweepRun(potential={"Qb": Qb, "f_bh": f_bh})
        for Qb in Qb_values
        for f_bh in fbh_values
    ]


def _is_constant(column: Any) -> bool:
    """True if every value in an astropy Column/Quantity is the same
    (NaN-aware for float columns: an all-NaN column counts as constant)."""
    arr = np.asarray(column)
    if len(arr) == 0:
        return True
    if arr.dtype.kind == "f":
        first = arr[0]
        if np.isnan(first):
            return bool(np.all(np.isnan(arr)))
        return bool(np.all(arr == first))
    try:
        return bool(np.all(arr == arr[0]))
    except Exception:  # noqa: BLE001 -- unhashable/uncomparable cell values
        return False


def summarize_sweep(table: QTable) -> None:
    """Print a sweep table split into what's fixed vs. what varies.

    Most columns of a sweep table (accuracy, max_num_steps, grid_size,
    thresholds, ...) come straight from `grid_kwargs` and are identical on
    every row -- printing all of them per-row just repeats the same values
    N times and buries the handful of columns (Qb, f_bh, chaotic
    fractions, ...) that actually differ between runs. This prints the
    constant columns once as a settings block, then a table of only the
    columns that vary.
    """
    if len(table) == 0:
        log.info("Sweep table is empty -- nothing to summarize.")
        return

    fixed: dict[str, Any] = {}
    varying: list[str] = []
    for name in table.colnames:
        if _is_constant(table[name]):
            fixed[name] = table[name][0]
        else:
            varying.append(name)

    if fixed:
        print_kv_table(
            title="Fixed across all runs",
            data={k: str(v) for k, v in fixed.items()},
        )

    print_dataframe_table(
        title=f"Sweep results ({len(table)} row(s), varying columns only)",
        headers=varying,
        rows=[[table[name][i] for name in varying] for i in range(len(table))],
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GridChaosDetector sweep over a Q_b x f_bh grid."
    )
    add_qb_fbh_args(parser)
    add_clear_cache_arg(parser)
    parser.add_argument(
        "--r0", type=float, default=8.0, help="Reference radius R_0 (default: 8.0)."
    )
    parser.add_argument(
        "--grid-size", type=int, default=10, help="Grid points per axis (default: 10)."
    )
    parser.add_argument(
        "--iter-time",
        type=float,
        default=10.0,
        help="Integration time per orbit (default: 10.0).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for the sweep (default: 1, serial).",
    )
    parser.add_argument(
        "--on-error",
        choices=["raise", "skip", "collect"],
        default="collect",
        help="What to do if one sweep entry fails (default: collect).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "sweep_results.ecsv",
        help="Where to write/checkpoint the sweep table "
        "(default: scripts/outputs/sweep_results.ecsv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output if it already exists, skipping completed runs.",
    )
    return parser


# ----------------------------------------------------------------------------
# MAIN: only runs when this script is executed directly
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    setup_logging()  # only the top-level script should call this, not the library code above
    print_banner("GridChaosDetector sweep runner", "Q_b x f_bh grid")

    if args.clear_cache:
        clearCache()

    grid_kwargs = {
        "R_0": args.r0,
        "grid_size": args.grid_size,
        "iter_time": args.iter_time,
    }
    runs = qb_fbh_grid_runs(Qb_values=args.qb, fbh_values=args.fbh)
    chaos_map_dir = BASE_DIR / "chaos_maps"
    results = run_sweep(
        runs,
        grid_kwargs=grid_kwargs,
        on_error=args.on_error,
        checkpoint_path=args.output,
        resume=args.resume,
        n_jobs=args.n_jobs,
        chaos_map_dir=chaos_map_dir,
    )

    summarize_sweep(results)

    if results.meta["errors"]:
        log.warning(
            "%d/%d sweep run(s) failed -- see results.meta['errors'] for details.",
            len(results.meta["errors"]),
            results.meta["n_requested"],
        )

    log.info("Wrote %d-row sweep table to %s", len(results), args.output)
