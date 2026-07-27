from numba import njit, prange
import numpy as np


@njit(parallel=True, fastmath=True, cache=True)
def _sali_kernel(arr: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    """Compute SALI per (orbit, pair, timestep) without materializing
    full-size w1/w2 arrays — only tiny fixed-length vectors ever exist
    in memory at once.
    """
    n_orbits, _, n_time, n_dim = arr.shape
    n_pairs = idx_i.shape[0]
    out = np.empty((n_orbits, n_pairs, n_time), dtype=arr.dtype)

    for orb in prange(n_orbits):
        for p in range(n_pairs):
            i = idx_i[p]
            j = idx_j[p]
            for t in range(n_time):
                sum_sq = 0.0
                diff_sq = 0.0
                for d in range(n_dim):
                    a = arr[orb, i, t, d]
                    b = arr[orb, j, t, d]
                    s = a + b
                    df = a - b
                    sum_sq += s * s
                    diff_sq += df * df
                sum_norm = np.sqrt(sum_sq)
                diff_norm = np.sqrt(diff_sq)
                out[orb, p, t] = sum_norm if sum_norm < diff_norm else diff_norm

    return out
