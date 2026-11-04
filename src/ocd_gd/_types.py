"""
Data containers for orbit chaos detection results.

Kept separate from the detector logic since these are plain structured
containers with no behavior of their own.
"""

__all__ = [
    "_UNSET",
    "ChaosAgreement",
    "ChaosSurveySummary",
    "DecayClassification",
    "GridInitialConditions",
    "IntegrationCriteria",
    "MethodChaosStats",
    "chaos_summary_row",
]

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from astropy.table import QTable

_UNSET: Any = object()


@dataclass(frozen=True)
class IntegrationCriteria:
    """Frozen snapshot of the integration and chaos-indicator stopping criteria.

    Parameters
    ----------
    iter_time : float
        Total integration time in Agama time units.
    traj_size : int
        size of trajectory array
    accuracy : float
        The target integration accuracy.
    max_num_steps : int
        Maximum number of integration steps allowed.

    Examples
    --------
    >>> from ocd_gd._types import IntegrationCriteria
    >>> criteria = IntegrationCriteria(
    ...     iter_time=100.0, traj_size=1000,
    ...     accuracy=1e-8,
    ...     max_num_steps=100000,
    ... )
    >>> criteria.iter_time
    100.0
    """

    iter_time: float | npt.NDArray[np.float64]
    traj_size: int
    accuracy: float
    max_num_steps: int


class DecayClassification(NamedTuple):
    """Decay-model classification results for a batch of integrated orbits.

    Returned by `OrbitChaosDetector.detect_chaos()`. Each field is the raw
    output dict from `classify_decay` applied to the corresponding indicator
    array — see `_decay_classifier.LABELS` for the label convention
    (odd = regular, even > 0 = chaotic, 0 = undetermined).

    Parameters
    ----------
    sali : dict
        ``classify_decay`` output for the SALI time series.  Keys include
        ``"label"``, ``"score"``, ``"rate"``, ``"delta_bic"``, etc.
    gali : dict
        ``classify_decay`` output for the GALI time series.  Keys include
        ``"label"``, ``"score"``, ``"rate"``, ``"delta_bic"``, etc.
    lyapunov_check : ndarray, optional
        Per-orbit Lyapunov classification: 1.0 = chaotic (exponent > 0.1),
        0.0 = regular, NaN = undetermined.
    lyapunov_time : ndarray, optional
        Integration time at which the Lyapunov exponent was evaluated,
        or ``inf`` for undetermined orbits.

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._types import DecayClassification
    >>> dummy = {"label": np.array([1]), "score": np.array([0.1])}
    >>> dc = DecayClassification(sali=dummy, gali=dummy)
    >>> dc.sali["label"]
    array([1])
    """

    sali: dict
    gali: dict
    lyapunov_check: npt.NDArray[np.float64] | None = None
    lyapunov_time: npt.NDArray[np.float64] | None = None

    @property
    def sali_check(self) -> npt.NDArray[np.float64]:
        lbl = np.asarray(self.sali["label"])
        check = ((lbl > 0) & (lbl % 2 == 0)).astype(np.float64)
        if check.ndim == 2 and check.shape[0] == 15:
            return np.any(check > 0, axis=0).astype(np.float64)
        return check

    @property
    def gali_check(self) -> npt.NDArray[np.float64]:
        lbl = np.asarray(self.gali["label"])
        return ((lbl > 0) & (lbl % 2 == 0)).astype(np.float64)

    @property
    def sali_time(self) -> npt.NDArray[np.float64]:
        t_onset = np.asarray(self.sali.get("t_onset", np.nan), dtype=np.float64)
        t_floor = np.asarray(self.sali.get("t_floor", np.nan), dtype=np.float64)
        t_det = np.where(np.isfinite(t_onset), t_onset, t_floor)
        t_det = np.where(np.isfinite(t_det), t_det, np.inf)
        if t_det.ndim == 2 and t_det.shape[0] == 15:
            return np.min(t_det, axis=0)
        return t_det

    @property
    def gali_time(self) -> npt.NDArray[np.float64]:
        t_onset = np.asarray(self.gali.get("t_onset", np.nan), dtype=np.float64)
        t_floor = np.asarray(self.gali.get("t_floor", np.nan), dtype=np.float64)
        t_det = np.where(np.isfinite(t_onset), t_onset, t_floor)
        return np.where(np.isfinite(t_det), t_det, np.inf)


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
