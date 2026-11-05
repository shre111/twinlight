import heapq
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from twl_citybake.stores.travel import directed_arcs


@dataclass(frozen=True)
class Placement:
    """Stores in placement order; `covered_share` is cumulative weighted coverage after each."""

    node: NDArray[np.uint32]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    covered_share: NDArray[np.float64]


def main_component(
    nodes: Mapping[str, np.ndarray], edges: Mapping[str, np.ndarray]
) -> NDArray[np.bool_]:
    """Nodes in the largest strongly connected component: from any of them a rider can reach all
    the others and come back. A tie goes to the component holding the lowest node id."""
    n = len(next(iter(nodes.values())))
    if n == 0:
        return np.zeros(0, bool)
    tail, head, _ = directed_arcs(edges)
    adjacency = csr_array((np.ones(len(tail), np.int32), (tail, head)), shape=(n, n))
    _, labels = connected_components(adjacency, directed=True, connection="strong")
    sizes = np.bincount(labels)
    in_largest = np.flatnonzero(sizes[labels] == sizes.max())
    return labels == labels[in_largest[0]]


def candidate_nodes(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    in_component: NDArray[np.bool_],
    cell_m: float,
) -> NDArray[np.int64]:
    """Per grid cell, the node nearest the cell centre (ties to the lowest node id); ascending."""
    nodes = np.flatnonzero(in_component).astype(np.int64)
    xs, ys = np.asarray(x, np.float64)[nodes], np.asarray(y, np.float64)[nodes]
    column, row = np.floor(xs / cell_m), np.floor(ys / cell_m)
    offset = np.hypot(xs - (column + 0.5) * cell_m, ys - (row + 0.5) * cell_m)
    order = np.lexsort((nodes, offset, row, column))
    column, row = column[order], row[order]
    first = np.ones(len(order), bool)
    first[1:] = (column[1:] != column[:-1]) | (row[1:] != row[:-1])
    return np.sort(nodes[order[first]])


def place_stores(
    res_x: NDArray[np.floating],
    res_y: NDArray[np.floating],
    res_weight: NDArray[np.floating],
    cand_x: NDArray[np.floating],
    cand_y: NDArray[np.floating],
    cand_node: NDArray[np.integer],
    radius_m: float,
    coverage_target: float,
    max_stores: int,
) -> Placement:
    """Exact greedy max coverage, ties to the lowest node id. Lazy: a candidate's gain only falls
    as others are placed, so a stored gain bounds the true one and the top of the heap is the
    greedy choice as soon as its gain is confirmed still current."""
    candidate_x = np.asarray(cand_x, np.float64)
    candidate_y = np.asarray(cand_y, np.float64)
    uncovered = np.asarray(res_weight, np.float64).copy()
    total = float(uncovered.sum())
    reach = _residents_within(res_x, res_y, candidate_x, candidate_y, radius_m)
    heap = [(-float(uncovered[r].sum()), int(cand_node[i]), i) for i, r in enumerate(reach)]
    heapq.heapify(heap)
    chosen: list[int] = []
    shares: list[float] = []
    covered = 0.0
    while heap and len(chosen) < max_stores and total > 0 and covered / total < coverage_target:
        stored_gain, node, i = heapq.heappop(heap)
        gain = float(uncovered[reach[i]].sum())
        if gain <= 0:
            continue
        if gain != -stored_gain:
            heapq.heappush(heap, (-gain, node, i))
            continue
        uncovered[reach[i]] = 0.0
        covered += gain
        chosen.append(i)
        shares.append(min(covered / total, 1.0))
    picks = np.asarray(chosen, np.intp)
    return Placement(
        node=np.asarray(cand_node)[picks].astype(np.uint32),
        x=candidate_x[picks].astype(np.float32),
        y=candidate_y[picks].astype(np.float32),
        covered_share=np.asarray(shares, np.float64),
    )


def _residents_within(
    res_x: NDArray[np.floating],
    res_y: NDArray[np.floating],
    cand_x: NDArray[np.float64],
    cand_y: NDArray[np.float64],
    radius_m: float,
) -> list[NDArray[np.int32]]:
    tree = KDTree(np.column_stack([np.asarray(res_x, np.float64), np.asarray(res_y, np.float64)]))
    return [
        np.asarray(tree.query_ball_point((x, y), radius_m, return_sorted=True), np.int32)
        for x, y in zip(cand_x.tolist(), cand_y.tolist(), strict=True)
    ]
