from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

KIND_NAMES = ("household", "neighbour", "friend", "work")
HOUSEHOLD, NEIGHBOUR, FRIEND, WORK = range(4)
NONE = 0xFFFFFFFF

Triple = tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.float32]]


def triple(pairs: Sequence[tuple[int, int, float]]) -> Triple:
    return (
        np.array([p[0] for p in pairs], np.uint32),
        np.array([p[1] for p in pairs], np.uint32),
        np.array([p[2] for p in pairs], np.float32),
    )


def merge(per_kind: Mapping[int, Triple], n: int) -> Any:
    from twl_citybake.graph.ties import merge_ties

    return merge_ties([per_kind.get(k, triple([])) for k in range(4)], n)


def build_graph(ties: Sequence[tuple[int, int, int, float]], n: int) -> Any:
    """Ties as (u, v, kind, strength)."""
    return merge({k: triple([(u, v, s) for u, v, kk, s in ties if kk == k]) for k in range(4)}, n)


def edges(graph: Any) -> dict[tuple[int, int], tuple[int, float]]:
    offsets, targets = graph.offsets.astype(np.int64), graph.targets
    out: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(len(offsets) - 1):
        for j in range(offsets[i], offsets[i + 1]):
            out[(i, int(targets[j]))] = (int(graph.kind[j]), float(graph.strength[j]))
    return out


def assert_csr_invariants(graph: Any, n: int) -> None:
    offsets, targets = graph.offsets, graph.targets
    assert offsets.dtype == np.uint32 and targets.dtype == np.uint32
    assert graph.kind.dtype == np.uint8 and graph.strength.dtype == np.float32
    assert len(offsets) == n + 1 and offsets[0] == 0 and offsets[-1] == len(targets)
    assert (np.diff(offsets.astype(np.int64)) >= 0).all()
    assert len(graph.kind) == len(targets) == len(graph.strength)
    if len(targets):
        assert int(targets.max()) < n
        assert (graph.strength > 0).all() and (graph.strength <= 1).all()
        assert int(graph.kind.max()) <= WORK
    source = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets.astype(np.int64)))
    assert (source != targets.astype(np.int64)).all()
    key = source * n + targets.astype(np.int64)
    assert (np.diff(key) > 0).all()
    table = edges(graph)
    for (a, b), (kind, strength) in table.items():
        assert table[(b, a)] == (kind, strength)


def pair_distance(
    x: NDArray[np.float64], y: NDArray[np.float64], u: NDArray[np.uint32], v: NDArray[np.uint32]
) -> NDArray[np.float64]:
    return np.hypot(x[u] - x[v], y[u] - y[v])


@dataclass(frozen=True)
class World:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    household: NDArray[np.uint32]
    segment: NDArray[np.uint16]

    @property
    def n(self) -> int:
        return len(self.x)


def uniform_world(
    n: int,
    seed: int = 7,
    extent_m: float = 3000.0,
    household_size: int = 1,
    shares: Sequence[float] = (0.5, 0.3, 0.2),
) -> World:
    rng = np.random.default_rng(seed)
    households = -(-n // household_size)
    hx = rng.uniform(0, extent_m, households)
    hy = rng.uniform(0, extent_m, households)
    household = (np.arange(n) // household_size).astype(np.uint32)
    segment = rng.choice(len(shares), n, p=list(shares)).astype(np.uint16)
    return World(hx[household], hy[household], household, segment)
