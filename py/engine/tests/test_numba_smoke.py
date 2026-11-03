import numpy as np
from numba import njit


@njit(cache=True)
def running_total(minutes: np.ndarray) -> np.ndarray:
    out = np.empty_like(minutes)
    total = 0
    for i in range(minutes.shape[0]):
        total += minutes[i]
        out[i] = total
    return out


def test_njit_kernel_matches_numpy() -> None:
    minutes = np.arange(1, 2_881, dtype=np.int64) * 15
    np.testing.assert_array_equal(running_total(minutes), np.cumsum(minutes))
