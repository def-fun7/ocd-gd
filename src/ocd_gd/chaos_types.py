"""
Data containers for orbit chaos detection results.

Kept separate from the detector logic since these are plain structured
containers with no behavior of their own.
"""

from typing import NamedTuple
from dataclasses import dataclass
import numpy as np


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

    gali_check: np.ndarray
    gali_time: np.ndarray
    sali_check: np.ndarray
    sali_time: np.ndarray
    lyapunov_check: np.ndarray
    lyapunov_time: np.ndarray


class ChaosFullReport(NamedTuple):
    """Complete diagnostic bundle containing summaries alongside raw arrays."""

    summary: ChaosSummary
    timestamps: np.ndarray
    gali_array: np.ndarray
    sali_array: np.ndarray
    lyapunov_array: np.ndarray
