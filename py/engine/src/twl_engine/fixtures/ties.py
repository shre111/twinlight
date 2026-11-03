import math

import numpy as np
from numpy.typing import NDArray

from twl_engine.fixtures.types import TIE_FRIEND, TIE_NEIGHBOUR, Residents, Ties

CELL_M = 300.0
GRID_HALF_M = 5000.0
GRID_CELLS = math.ceil(2 * GRID_HALF_M / CELL_M)
LONG_RANGE_SHARE = 0.1
LOCAL_TRIES = 8
LOCAL_REACH_M = 600.0
CELL_SPREAD = 2.2

IntArray = NDArray[np.int64]


def build_ties(residents: Residents, rng: np.random.Generator) -> Ties:
    """Three to six directed ties each: a random friend, else a neighbour (prototype `buildTies`).

    Each local tie takes up to eight tries at a nearby 300 m cell and a member within 600 m;
    tries run as vectorised rounds over the ties still unplaced.
    """
    n = residents.x.size
    x = residents.x.astype(np.float64)
    y = residents.y.astype(np.float64)
    gx = np.clip(np.floor((x + GRID_HALF_M) / CELL_M), 0, GRID_CELLS - 1).astype(np.int64)
    gy = np.clip(np.floor((y + GRID_HALF_M) / CELL_M), 0, GRID_CELLS - 1).astype(np.int64)
    members = np.argsort(gy * GRID_CELLS + gx, kind="stable")
    cell_start = np.concatenate(
        ([0], np.cumsum(np.bincount(gy * GRID_CELLS + gx, minlength=GRID_CELLS**2)))
    )

    owner = np.repeat(np.arange(n), 3 + rng.integers(0, 4, size=n))
    friend = rng.random(owner.size) < LONG_RANGE_SHARE
    target = np.full(owner.size, -1, dtype=np.int64)
    target[friend] = _random_others(owner[friend], n, rng)

    pending = np.flatnonzero(~friend)
    for _ in range(LOCAL_TRIES):
        if pending.size == 0:
            break
        i = owner[pending]
        draw = rng.random((pending.size, 5))
        u = gx[i] + _cell_offset(draw[:, 0], draw[:, 1])
        v = gy[i] + _cell_offset(draw[:, 2], draw[:, 3])
        in_grid = (u >= 0) & (v >= 0) & (u < GRID_CELLS) & (v < GRID_CELLS)
        cell = np.where(in_grid, v * GRID_CELLS + u, 0)
        size = cell_start[cell + 1] - cell_start[cell]
        slot = cell_start[cell] + np.floor(draw[:, 4] * size).astype(np.int64)
        j = members[np.minimum(slot, n - 1)]
        near = (x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2 <= LOCAL_REACH_M**2
        placed = in_grid & (size > 0) & (j != i) & near
        target[pending[placed]] = j[placed]
        pending = pending[~placed]

    kept = (target >= 0) & (target != owner)
    return Ties(
        offsets=np.concatenate(([0], np.cumsum(np.bincount(owner[kept], minlength=n)))).astype(
            np.uint32
        ),
        targets=target[kept].astype(np.uint32),
        kind=np.where(friend[kept], TIE_FRIEND, TIE_NEIGHBOUR).astype(np.uint8),
        strength=np.ones(int(kept.sum()), dtype=np.float32),
    )


def _random_others(owner: IntArray, n: int, rng: np.random.Generator) -> IntArray:
    j = rng.integers(0, n, size=owner.size)
    return np.where(j == owner, (j + 1) % n, j)


def _cell_offset(a: NDArray[np.float64], b: NDArray[np.float64]) -> IntArray:
    """Triangular offset in -2..2 cells, rounded half up like JavaScript's Math.round."""
    return np.floor((a + b - 1) * CELL_SPREAD + 0.5).astype(np.int64)
