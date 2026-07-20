"""
Chaos Evaluation Module.

This module provides tools for analyzing metrics across time series data,
specifically identifying when a system metric falls below a given threshold
and sustains that behavior over a specified rolling window.
"""

from typing import Tuple, Union
import numpy as np


def evaluate_chaos(
    metric_arr: np.ndarray,
    time_arr: np.ndarray,
    threshold: float = 1e-12,
    separate: bool = False,
    window_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate chaos/convergence by finding sustained threshold drops.

    This function scans a metrics array along its final time axis to detect if
    and when the metrics drop below a given threshold and stay below it for
    a contiguous window of time steps. It supports both 2D and 3D arrays.

    Parameters
    ----------
    metric_arr : np.ndarray
        The metric data array. Expected shapes are (N, T) for 2D or (N, M, T)
        for 3D, where T represents the time steps.
    time_arr : np.ndarray
        Array containing time step values corresponding to the last axis of
        metric_arr.
    threshold : float, default 1e-12
        The upper limit below which a metric is considered to be in a converged
        or non-chaotic state.
    separate : bool, default False
        Only applies to 3D arrays. If True, evaluates threshold crossings for
        each element in the second axis separately. If False, aggregates across
        the second axis to find the earliest crossing point.
    window_size : int, default 10
        The number of consecutive time steps the metric must remain below the
        threshold to be classified as sustained.

    Returns
    -------
    check : np.ndarray
        An integer array (1 for True, 0 for False) indicating if the sustained
        threshold drop occurred. Shape is (N, 1) or (N, M, 1) based on inputs.
    time : np.ndarray
        A float array containing the exact timestamp where the sustained drop
        first started, or np.inf if the condition was never met. Same shape
        as check.

    Raises
    ------
    ValueError
        If metric_arr has an unsupported number of dimensions (not 2D or 3D).
    """
    time_flat = time_arr.ravel()
    n_time_steps = metric_arr.shape[-1]

    raw_mask = metric_arr < threshold
    sustained_mask = raw_mask[..., : -(window_size - 1)].copy()

    for i in range(1, window_size):
        sustained_mask &= raw_mask[..., i : n_time_steps - (window_size - 1 - i)]

    if metric_arr.ndim == 2:
        any_crossed = np.any(sustained_mask, axis=1)
        check = np.where(any_crossed, 1, 0)[:, np.newaxis]

        time = np.where(
            any_crossed,
            time_flat[np.argmax(sustained_mask, axis=1)],
            np.inf,
        )[:, np.newaxis]

        return check, time

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

            return check, time

    # Keeping code paths exact, but adding a safe fallback for invalid dimensions
    return np.empty(0), np.empty(0)
