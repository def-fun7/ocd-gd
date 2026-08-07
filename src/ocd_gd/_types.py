"""
Data containers for orbit chaos detection results.

Kept separate from the detector logic since these are plain structured
containers with no behavior of their own.
"""

__all__ = [
    "ChaosAgreement",
    "ChaosFullReport",
    "ChaosSummary",
    "ChaosSurveySummary",
    "GridInitialConditions",
    "IntegrationCriteria",
    "MethodChaosStats",
    "chaos_summary_row",
]

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from astropy.table import QTable


@dataclass(frozen=True)
class IntegrationCriteria:
    """Frozen snapshot of the integration and chaos-indicator stopping criteria.

    Parameters
    ----------
    iter_time : float
        The time duration for each integration step.
    gali_threshold : float
        Convergence threshold for GALI.
    sali_threshold : float
        Convergence threshold for SALI.
    gali_window_size : int
        Window size to verify GALI convergence.
    sali_window_size : int
        Window size to verify SALI convergence.
    accuracy : float
        The target integration accuracy.
    max_num_steps : int
        Maximum number of integration steps allowed.

    Examples
    --------
    >>> from ocd_gd._types import IntegrationCriteria
    >>> criteria = IntegrationCriteria(
    ...     iter_time=100.0,
    ...     gali_threshold=1e-12,
    ...     sali_threshold=1e-12,
    ...     gali_window_size=10,
    ...     sali_window_size=10,
    ...     accuracy=1e-8,
    ...     max_num_steps=100000,
    ... )
    >>> criteria.iter_time
    100.0
    """

    iter_time: float
    gali_threshold: float
    sali_threshold: float
    gali_window_size: int
    sali_window_size: int
    accuracy: float
    max_num_steps: int


class ChaosSummary(NamedTuple):
    """Structured container holding processed summary chaos classifications.

    Parameters
    ----------
    gali_check : ndarray
        Flags indicating whether each orbit is chaotic according to GALI.
    gali_time : ndarray
        The time of chaos detection (or np.inf) for GALI.
    sali_check : ndarray
        Flags indicating whether each orbit is chaotic according to SALI.
    sali_time : ndarray
        The time of chaos detection (or np.inf) for SALI.
    lyapunov_check : ndarray
        Flags indicating whether each orbit is chaotic according to Lyapunov.
    lyapunov_time : ndarray
        The time of chaos detection (or np.inf) for Lyapunov.

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._types import ChaosSummary
    >>> summary = ChaosSummary(
    ...     gali_check=np.array([1, 0]),
    ...     gali_time=np.array([5.2, np.inf]),
    ...     sali_check=np.array([1, 0]),
    ...     sali_time=np.array([3.4, np.inf]),
    ...     lyapunov_check=np.array([0, 0]),
    ...     lyapunov_time=np.array([np.inf, np.inf]),
    ... )
    >>> summary.sali_check
    array([1, 0])
    """

    gali_check: npt.NDArray[np.float64]
    gali_time: npt.NDArray[np.float64]
    sali_check: npt.NDArray[np.float64]
    sali_time: npt.NDArray[np.float64]
    lyapunov_check: npt.NDArray[np.float64]
    lyapunov_time: npt.NDArray[np.float64]


class ChaosFullReport(NamedTuple):
    """Complete diagnostic bundle containing summaries alongside raw arrays.

    Parameters
    ----------
    summary : ChaosSummary
        Structured container with classification summaries.
    timestamps : ndarray
        The timestamps at which chaos metrics were recorded.
    gali_array : ndarray
        Raw GALI values over time for each orbit.
    sali_array : ndarray
        Raw SALI values over time for each orbit.
    lyapunov_array : ndarray
        Raw Lyapunov exponent estimates over time.

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._types import ChaosFullReport, ChaosSummary
    >>> summary = ChaosSummary(
    ...     gali_check=np.array([0]), gali_time=np.array([np.inf]),
    ...     sali_check=np.array([0]), sali_time=np.array([np.inf]),
    ...     lyapunov_check=np.array([0]), lyapunov_time=np.array([np.inf]),
    ... )
    >>> report = ChaosFullReport(
    ...     summary=summary,
    ...     timestamps=np.array([0.0, 1.0]),
    ...     gali_array=np.array([[1.0, 1.0]]),
    ...     sali_array=np.array([[1.0, 1.0]]),
    ...     lyapunov_array=np.array([[0.0, 0.0]]),
    ... )
    >>> isinstance(report.summary, ChaosSummary)
    True
    """

    summary: ChaosSummary
    timestamps: npt.NDArray[np.float64]
    gali_array: npt.NDArray[np.float64]
    sali_array: npt.NDArray[np.float64]
    lyapunov_array: npt.NDArray[np.float64]


class GridInitialConditions(NamedTuple):
    """Bundle of initial conditions and metadata produced by grid generation.

    Replaces the previous "ics, mask, (x_vals, v_x_vals), E_rem_vals" nested
    tuple with named fields.

    Parameters
    ----------
    ics : ndarray
        Initial conditions for the grid of orbits.
    unphysical_mask : ndarray
        Boolean/numeric mask where True indicates unphysical grid cells.
    x_vals : ndarray
        Grid coordinates along the x-axis.
    v_x_vals : ndarray
        Grid coordinates along the vx axis.
    E_rem_vals : ndarray
        Residual energy values for the grid positions.
    E_0 : float
        The reference energy used for the grid.

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._types import GridInitialConditions
    >>> grid_ic = GridInitialConditions(
    ...     ics=np.zeros((16, 6)),
    ...     unphysical_mask=np.zeros(16, dtype=bool),
    ...     x_vals=np.linspace(-1, 1, 4),
    ...     v_x_vals=np.linspace(-1, 1, 4),
    ...     E_rem_vals=np.zeros(4),
    ...     E_0=-1.5,
    ... )
    >>> grid_ic.E_0
    -1.5
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

    Parameters
    ----------
    n_chaotic : int
        Number of chaotic orbits.
    n_regular : int
        Number of regular orbits.
    n_undetermined : int
        Number of undetermined orbits.
    n_total : int
        Total number of orbits.
    chaotic_fraction : float
        Fraction of chaotic orbits in the batch.
    chaotic_indices : ndarray
        Indices of the chaotic orbits.
    regular_indices : ndarray
        Indices of the regular orbits.
    undetermined_indices : ndarray
        Indices of the undetermined orbits.
    chaotic_ics : ndarray
        Initial conditions of the chaotic orbits.
    regular_ics : ndarray
        Initial conditions of the regular orbits.

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._types import MethodChaosStats
    >>> stats = MethodChaosStats(
    ...     n_chaotic=1, n_regular=1, n_undetermined=0, n_total=2,
    ...     chaotic_fraction=0.5,
    ...     chaotic_indices=np.array([0]),
    ...     regular_indices=np.array([1]),
    ...     undetermined_indices=np.array([], dtype=np.intp),
    ...     chaotic_ics=np.zeros((1, 6)),
    ...     regular_ics=np.zeros((1, 6)),
    ... )
    >>> stats.chaotic_fraction
    0.5
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

    Parameters
    ----------
    sali_gali_agreement : float
        Fraction of orbits where SALI and GALI agree on classification.
    sali_lyapunov_agreement : float
        Fraction of orbits where SALI and Lyapunov agree on classification.
    gali_lyapunov_agreement : float
        Fraction of orbits where GALI and Lyapunov agree on classification.
    all_agree_chaotic : int
        Number of orbits where all three indicators classify as chaotic.
    all_agree_regular : int
        Number of orbits where all three indicators classify as regular.
    disagreement : int
        Number of orbits where indicators disagree.
    n_undetermined : int
        Number of orbits with undetermined classification.

    Examples
    --------
    >>> from ocd_gd._types import ChaosAgreement
    >>> agreement = ChaosAgreement(
    ...     sali_gali_agreement=1.0, sali_lyapunov_agreement=0.9,
    ...     gali_lyapunov_agreement=0.9, all_agree_chaotic=5,
    ...     all_agree_regular=4, disagreement=1, n_undetermined=0,
    ... )
    >>> agreement.disagreement
    1
    """

    sali_gali_agreement: float
    sali_lyapunov_agreement: float
    gali_lyapunov_agreement: float
    all_agree_chaotic: int
    all_agree_regular: int
    disagreement: int
    n_undetermined: int


class ChaosSurveySummary(NamedTuple):
    """Full chaos-detection summary across a batch of integrated orbits.

    Contains per-indicator counts/fractions/indices plus cross-indicator agreement.
    Returned by `OrbitChaosDetector.chaos_summary()` — works for any
    integrated batch, grid-based or not.

    Parameters
    ----------
    n_total : int
        Total number of orbits.
    sali : MethodChaosStats
        SALI statistics breakdown.
    gali : MethodChaosStats
        GALI statistics breakdown.
    lyapunov : MethodChaosStats
        Lyapunov statistics breakdown.
    agreement : ChaosAgreement
        Pairwise and three-way agreement statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._types import ChaosSurveySummary, MethodChaosStats, ChaosAgreement
    >>> m_stats = MethodChaosStats(
    ...     n_chaotic=1, n_regular=1, n_undetermined=0, n_total=2,
    ...     chaotic_fraction=0.5,
    ...     chaotic_indices=np.array([0]),
    ...     regular_indices=np.array([1]),
    ...     undetermined_indices=np.array([], dtype=np.intp),
    ...     chaotic_ics=np.zeros((1, 6)),
    ...     regular_ics=np.zeros((1, 6)),
    ... )
    >>> agreement = ChaosAgreement(
    ...     sali_gali_agreement=1.0, sali_lyapunov_agreement=1.0,
    ...     gali_lyapunov_agreement=1.0, all_agree_chaotic=1,
    ...     all_agree_regular=1, disagreement=0, n_undetermined=0,
    ... )
    >>> summary = ChaosSurveySummary(
    ...     n_total=2, sali=m_stats, gali=m_stats, lyapunov=m_stats, agreement=agreement,
    ... )
    >>> summary.n_total
    2
    """

    n_total: int
    sali: MethodChaosStats
    gali: MethodChaosStats
    lyapunov: MethodChaosStats
    agreement: ChaosAgreement


def chaos_summary_row(summary: ChaosSurveySummary) -> QTable:
    """Flatten a ChaosSurveySummary's scalar statistics into a single-row QTable.

    Omits the chaotic/regular index and IC arrays, which don't fit a tabular row.
    Pairs with `OrbitChaosDetector.metadata_row()` to build one row of a sweep-results
    table.

    Parameters
    ----------
    summary : ChaosSurveySummary
        The survey summary to flatten.

    Returns
    -------
    QTable
        Single-row Astropy QTable containing the scalar survey statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.table import QTable
    >>> from ocd_gd._types import ChaosSurveySummary, MethodChaosStats, ChaosAgreement, chaos_summary_row
    >>> m_stats = MethodChaosStats(
    ...     n_chaotic=1, n_regular=1, n_undetermined=0, n_total=2,
    ...     chaotic_fraction=0.5,
    ...     chaotic_indices=np.array([0]),
    ...     regular_indices=np.array([1]),
    ...     undetermined_indices=np.array([], dtype=np.intp),
    ...     chaotic_ics=np.zeros((1, 6)),
    ...     regular_ics=np.zeros((1, 6)),
    ... )
    >>> agreement = ChaosAgreement(
    ...     sali_gali_agreement=1.0, sali_lyapunov_agreement=1.0,
    ...     gali_lyapunov_agreement=1.0, all_agree_chaotic=1,
    ...     all_agree_regular=1, disagreement=0, n_undetermined=0,
    ... )
    >>> summary = ChaosSurveySummary(
    ...     n_total=2, sali=m_stats, gali=m_stats, lyapunov=m_stats, agreement=agreement,
    ... )
    >>> row = chaos_summary_row(summary)
    >>> isinstance(row, QTable)
    True
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
