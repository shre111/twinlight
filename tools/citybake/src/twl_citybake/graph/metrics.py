from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from twl_citybake.config import GraphTargets
from twl_citybake.graph.ties import FRIEND, HOUSEHOLD, TIE_KINDS, Graph


def graph_report(
    graph: Graph,
    bank: NDArray[np.integer],
    segment: NDArray[np.integer],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    n_residents: int,
) -> dict[str, Any]:
    """The P-023 diagnostics, unrounded. A share over no ties is 0.0, never NaN."""
    source = np.repeat(np.arange(n_residents), np.diff(graph.offsets.astype(np.int64)))
    target = graph.targets.astype(np.intp)
    once = source < target
    u, v, kind = source[once], target[once], graph.kind[once]
    strength = graph.strength[once].astype(np.float64)
    xs, ys = np.asarray(x, np.float64), np.asarray(y, np.float64)
    length = np.hypot(xs[u] - xs[v], ys[u] - ys[v])
    by_kind = [kind == k for k in range(len(TIE_KINDS))]
    return {
        "degree": _degree(graph, n_residents),
        "clustering": {
            **_clustering(source, target, n_residents),
            "without_household": _clustering(
                source[graph.kind != HOUSEHOLD], target[graph.kind != HOUSEHOLD], n_residents
            ),
        },
        "cross_river": _cross_river(bank[u] != bank[v], by_kind),
        "homophily": _homophily(segment, u[by_kind[FRIEND]], v[by_kind[FRIEND]]),
        "distance_m": {
            name: {
                "p50": _percentile(length[by_kind[k]], 50),
                "p90": _percentile(length[by_kind[k]], 90),
            }
            for k, name in enumerate(TIE_KINDS)
        },
        "components": _components(source, target, n_residents),
        "strength": {
            **{name: _spread(strength[by_kind[k]]) for k, name in enumerate(TIE_KINDS)},
            "per_resident_mean": _spread(_mean_strength(graph, source, n_residents)),
        },
    }


def within_targets(report: dict[str, Any], targets: GraphTargets) -> dict[str, bool]:
    """Per target, whether the report meets it, and `all`."""
    checks = {
        "mean_degree": _inside(report["degree"]["mean"], targets.mean_degree),
        "clustering": _inside(report["clustering"]["average_local"], targets.clustering),
        "cross_river_non_household": _inside(
            report["cross_river"]["non_household"], targets.cross_river_non_household
        ),
        "largest_component_min": bool(
            report["components"]["largest_share"] >= targets.largest_component_min
        ),
        "isolated_max": bool(report["degree"]["isolated_share"] <= targets.isolated_max),
    }
    return {**checks, "all": all(checks.values())}


def _inside(value: float, bounds: tuple[float, float]) -> bool:
    return bool(bounds[0] <= value <= bounds[1])


def _degree(graph: Graph, n: int) -> dict[str, Any]:
    degree = np.diff(graph.offsets.astype(np.int64))
    p10, p50, p90, p99 = (_percentile(degree, q) for q in (10, 50, 90, 99))
    counts = np.bincount(graph.kind, minlength=len(TIE_KINDS))
    return {
        "mean": _share(len(graph.targets), n),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "max": int(degree.max()) if n else 0,
        "isolated_share": _share(int((degree == 0).sum()), n),
        "mean_by_kind": {name: _share(int(counts[k]), n) for k, name in enumerate(TIE_KINDS)},
    }


def _adjacency(source: NDArray[np.intp], target: NDArray[np.intp], n: int) -> csr_array:
    return csr_array((np.ones(len(source), np.int64), (source, target)), shape=(n, n))


def _clustering(source: NDArray[np.intp], target: NDArray[np.intp], n: int) -> dict[str, float]:
    """(A^2 element-wise A) summed over a row counts each triangle at that node twice, and
    k (k - 1) counts its ordered neighbour pairs, so their ratio is the local clustering and the
    ratio of their totals is the transitivity 3 T / triples."""
    adjacency = _adjacency(source, target, n)
    closed = np.asarray(
        csr_array(adjacency @ adjacency).multiply(adjacency).sum(axis=1), np.float64
    )
    degree = np.bincount(source, minlength=n).astype(np.float64)
    pairs = degree * (degree - 1)
    counted = degree >= 2
    return {
        "average_local": float((closed[counted] / pairs[counted]).mean()) if counted.any() else 0.0,
        "transitivity": _share(float(closed.sum()), float(pairs.sum())),
    }


def _cross_river(crosses: NDArray[np.bool_], by_kind: list[NDArray[np.bool_]]) -> dict[str, Any]:
    return {
        "all": _mean(crosses),
        "non_household": _mean(crosses[~by_kind[HOUSEHOLD]]),
        "by_kind": {name: _mean(crosses[by_kind[k]]) for k, name in enumerate(TIE_KINDS)},
    }


def _homophily(
    segment: NDArray[np.integer], u: NDArray[np.intp], v: NDArray[np.intp]
) -> dict[str, float]:
    _, counts = np.unique(segment, return_counts=True)
    shares = counts / max(len(segment), 1)
    return {
        "friend_same_segment_share": _mean(segment[u] == segment[v]),
        "expected_random_mixing": float((shares**2).sum()),
    }


def _components(source: NDArray[np.intp], target: NDArray[np.intp], n: int) -> dict[str, Any]:
    if n == 0:
        return {"count": 0, "largest_share": 0.0}
    count, labels = connected_components(_adjacency(source, target, n), directed=False)
    return {"count": int(count), "largest_share": float(np.bincount(labels).max() / n)}


def _mean_strength(graph: Graph, source: NDArray[np.intp], n: int) -> NDArray[np.float64]:
    """Per resident with ties, the mean strength over them."""
    degree = np.bincount(source, minlength=n)
    total = np.bincount(source, weights=graph.strength.astype(np.float64), minlength=n)
    return total[degree > 0] / degree[degree > 0]


def _spread(values: NDArray[np.floating]) -> dict[str, float]:
    return {
        "p10": _percentile(values, 10),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "mean": _mean(values),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _share(part: float, whole: float) -> float:
    return float(part / whole) if whole else 0.0
