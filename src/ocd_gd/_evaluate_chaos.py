"""
Chaos Evaluation Module.

This module provides tools for analyzing metrics across time series data,
specifically identifying when a system metric falls below a given threshold
and sustains that behavior over a specified rolling window.
"""

__all__ = ["evaluate_chaos"]

import numpy as np
import numpy.typing as npt


def evaluate_chaos(
    metric_arr: npt.NDArray[np.float64],
    time_arr: npt.NDArray[np.float64],
    threshold: float = 1e-12,
    separate: bool = False,
    window_size: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Determine convergence/chaos based on whether a metric sustains a value below a threshold.

    Parameters
    ----------
    metric_arr : ndarray
        SALI/GALI values over time. Can be 2D (n_orbits, n_times) or 3D
        (n_orbits, n_methods, n_times).
    time_arr : ndarray
        1D array of times corresponding to the last axis of `metric_arr`.
    threshold : float, default 1e-12
        Threshold below which the metric must fall.
    separate : bool, default False
        For 3D inputs, if True, evaluate convergence per method.
    window_size : int, default 10
        Number of consecutive steps the metric must stay below the threshold
        to confirm convergence.

    Returns
    -------
    check : ndarray
        Convergence flag (1 for converged/chaotic, 0 otherwise).
    time : ndarray
        First time at which sustained convergence was reached (or `np.inf`).

    Examples
    --------
    >>> import numpy as np
    >>> from ocd_gd._evaluate_chaos import evaluate_chaos
    >>> metric = np.array([[1.0, 1.0, 1e-13, 1e-14, 1e-15, 1e-15]])
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    >>> check, time = evaluate_chaos(metric, times, threshold=1e-12, window_size=3)
    >>> check
    array([1])
    >>> time
    array([2.])
    """
    time_flat = time_arr.ravel()
    n_time_steps = metric_arr.shape[-1]

    raw_mask = metric_arr < threshold

    false_count = (~raw_mask).astype(np.int32)
    cumsum = np.cumsum(false_count, axis=-1)
    cumsum = np.concatenate([np.zeros_like(cumsum[..., :1]), cumsum], axis=-1)
    window_false_count = cumsum[..., window_size:] - cumsum[..., :-window_size]
    sustained_mask = window_false_count == 0

    if metric_arr.ndim == 2:
        any_crossed = np.any(sustained_mask, axis=1)
        check = np.where(any_crossed, 1, 0)[:, np.newaxis]

        time = np.where(
            any_crossed,
            time_flat[np.argmax(sustained_mask, axis=1)],
            np.inf,
        )[:, np.newaxis]

        return check.flatten(), time.flatten()

    elif metric_arr.ndim == 3:
        if separate:
            any_crossed = np.any(sustained_mask, axis=2)
            check = np.where(any_crossed, 1, 0)[..., np.newaxis]
            time = np.where(
                any_crossed,
                time_flat[np.argmax(sustained_mask, axis=2)],
                np.inf,
            )[..., np.newaxis]

            return check, time
        else:
            any_crossed = np.any(sustained_mask, axis=(1, 2))
            check = np.where(any_crossed, 1, 0)[:, np.newaxis]
            n_reduced_steps = sustained_mask.shape[-1]
            temp_indices = np.where(
                sustained_mask, np.arange(n_reduced_steps), n_reduced_steps + 1
            )
            earliest_index = np.min(temp_indices, axis=(1, 2))

            time = np.where(
                any_crossed,
                time_flat[np.minimum(earliest_index, n_time_steps - 1)],
                np.inf,
            )[:, np.newaxis]

            return check.flatten(), time.flatten()

    return np.empty(0), np.empty(0)
