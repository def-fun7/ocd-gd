"""
Data containers for orbit chaos detection results.

Kept separate from the detector logic since these are plain structured
containers with no behavior of their own.
"""

__all__ = [
    "IntegrationCriteria",
    "ChaosSummary",
    "ChaosFullReport",
    "GridInitialConditions",
    "MethodChaosStats",
    "ChaosAgreement",
    "ChaosSurveySummary",
    "chaos_summary_row",
]

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from astropy.table import QTable


@dataclass(frozen=True)
class IntegrationCriteria:
    """Frozen snapshot of the integration and chaos-indicator stopping criteria."""

    iter_time: float
    gali_threshold: float
    sali_threshold: float
    gali_window_size: int
    sali_window_size: int
    accuracy: float
    max_num_steps: int


class ChaosSummary(NamedTuple):
    """Structured container holding processed summary chaos classifications."""

    gali_check: npt.NDArray[np.float64]
    gali_time: npt.NDArray[np.float64]
    sali_check: npt.NDArray[np.float64]
    sali_time: npt.NDArray[np.float64]
    lyapunov_check: npt.NDArray[np.float64]
    lyapunov_time: npt.NDArray[np.float64]


class ChaosFullReport(NamedTuple):
    """Complete diagnostic bundle containing summaries alongside raw arrays."""

    summary: ChaosSummary
    timestamps: npt.NDArray[np.float64]
    gali_array: npt.NDArray[np.float64]
    sali_array: npt.NDArray[np.float64]
    lyapunov_array: npt.NDArray[np.float64]


class GridInitialConditions(NamedTuple):
    """Bundle of initial conditions and metadata produced by grid generation.

    Replaces the previous "ics, mask, (x_vals, v_x_vals), E_rem_vals" nested
    tuple with named fields.
    """

    ics: npt.NDArray[np.float64]
    unphysical_mask: npt.NDArray[np.float64]
    x_vals: npt.NDArray[np.float64]
    v_x_vals: npt.NDArray[np.float64]
    E_rem_vals: npt.NDArray[np.float64]
    E_0: float


class MethodChaosStats(NamedTuple):
    """Chaotic/regular breakdown for a single indicator (SALI, GALI, or
    Lyapunov), including which orbits fell into each class.

    `chaotic_indices`/`regular_indices` are orbit_idx values — usable
    directly with `get_sali`, `get_trajectory`, `plot_sali`, etc.
    `chaotic_ics`/`regular_ics` are the matching rows of `self.ic`, provided
    so a chaotic/regular subset can be re-integrated, re-analyzed, or handed
    to another tool without re-deriving the indices first.
    """

    n_chaotic: int
    n_regular: int
    n_undetermined: int
    n_total: int
    chaotic_fraction: float
    chaotic_indices: npt.NDArray[np.float64]
    regular_indices: npt.NDArray[np.float64]
    undetermined_indices: npt.NDArray[np.float64]
    chaotic_ics: npt.NDArray[np.float64]
    regular_ics: npt.NDArray[np.float64]


class ChaosAgreement(NamedTuple):
    """Pairwise and three-way agreement between SALI, GALI, and Lyapunov
    classifications, for a single batch of orbits.

    The pairwise fields are agreement *rates* (fraction of orbits where two
    indicators reach the same chaotic/regular verdict); `all_agree_chaotic`
    and `all_agree_regular` are raw counts where all three agree;
    `disagreement` counts orbits where at least one indicator differs from
    the others.
    """

    sali_gali_agreement: float
    sali_lyapunov_agreement: float
    gali_lyapunov_agreement: float
    all_agree_chaotic: int
    all_agree_regular: int
    disagreement: int
    n_undetermined: int


class ChaosSurveySummary(NamedTuple):
    """Full chaos-detection summary across a batch of integrated orbits:
    per-indicator counts/fractions/indices plus cross-indicator agreement.

    Returned by `OrbitChaosDetector.chaos_summary()` — works for any
    integrated batch, grid-based or not.
    """

    n_total: int
    sali: MethodChaosStats
    gali: MethodChaosStats
    lyapunov: MethodChaosStats
    agreement: ChaosAgreement


def chaos_summary_row(summary: ChaosSurveySummary) -> QTable:
    """Flatten a ChaosSurveySummary's scalar statistics (counts, fractions,
    agreement) into a single-row QTable — omits the chaotic/regular index
    and IC arrays, which don't fit a tabular row.

    Pairs with `OrbitChaosDetector.metadata_row()`: put the two side by side
    (e.g. via `astropy.table.hstack`) to build one row of a sweep-results
    table — `metadata_row` for the run's inputs, this for its outputs.
    """
    columns = {
        "n_total": [summary.n_total],
        "sali_n_chaotic": [summary.sali.n_chaotic],
        "sali_n_regular": [summary.sali.n_regular],
        "sali_chaotic_fraction": [summary.sali.chaotic_fraction],
        "gali_n_chaotic": [summary.gali.n_chaotic],
        "gali_n_regular": [summary.gali.n_regular],
        "gali_chaotic_fraction": [summary.gali.chaotic_fraction],
        "lyapunov_n_chaotic": [summary.lyapunov.n_chaotic],
        "lyapunov_n_regular": [summary.lyapunov.n_regular],
        "lyapunov_chaotic_fraction": [summary.lyapunov.chaotic_fraction],
        "sali_gali_agreement": [summary.agreement.sali_gali_agreement],
        "sali_lyapunov_agreement": [summary.agreement.sali_lyapunov_agreement],
        "gali_lyapunov_agreement": [summary.agreement.gali_lyapunov_agreement],
        "all_agree_chaotic": [summary.agreement.all_agree_chaotic],
        "all_agree_regular": [summary.agreement.all_agree_regular],
        "disagreement": [summary.agreement.disagreement],
    }
    return QTable(columns)
