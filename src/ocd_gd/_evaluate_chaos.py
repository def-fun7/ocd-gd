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
    """... (docstring unchanged) ..."""
    time_flat = time_arr.ravel()
    n_time_steps = metric_arr.shape[-1]

    raw_mask = metric_arr < threshold

    # Sliding-window "all True for window_size steps" via prefix sums of the
    # False count, instead of window_size-1 sequential in-place ANDs.
    # window_false_count[..., k] = number of False entries in
    # raw_mask[..., k : k+window_size]; sustained_mask is where that's zero.
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

    return np.empty(0), np.empty(0)
