from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

from twl_citybake.config import TravelBand
from twl_citybake.osm.tags import RoadClass

METRES_PER_MINUTE_PER_KMH = 1000.0 / 60.0


class TravelBackend(Protocol):
    """Travel minutes from road nodes to road nodes in one time band. OSRM would plug in here."""

    name: str

    def minutes(
        self, sources: NDArray[np.int64], targets: NDArray[np.int64], band: TravelBand
    ) -> NDArray[np.float64]: ...


def directed_arcs(
    edges: Mapping[str, np.ndarray],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.intp]]:
    """Tail, head and edge index of each arc: `oneway` 0 gives both directions, 1 u to v only and
    -1 v to u only."""
    u, v = edges["u"].astype(np.int64), edges["v"].astype(np.int64)
    forward = np.flatnonzero(edges["oneway"] >= 0)
    backward = np.flatnonzero(edges["oneway"] <= 0)
    return (
        np.concatenate([u[forward], v[backward]]),
        np.concatenate([v[forward], u[backward]]),
        np.concatenate([forward, backward]),
    )


class GraphTravelBackend:
    """D-022: Dijkstra over the directed road graph. All arcs scale by one band factor, so the
    fastest of several parallel arcs is the same in every band and is chosen once."""

    name = "graph-dijkstra"

    def __init__(
        self, n_nodes: int, road_edges: Mapping[str, np.ndarray], speed_kmh: Mapping[str, float]
    ) -> None:
        tail, head, edge = directed_arcs(road_edges)
        speeds = np.array([speed_kmh[road.name.lower()] for road in RoadClass], np.float64)
        length = road_edges["length_m"].astype(np.float64)
        speed = speeds[road_edges["road_class"]] * METRES_PER_MINUTE_PER_KMH
        minutes = length[edge] / speed[edge]
        keep = tail != head
        tail, head, minutes = tail[keep], head[keep], minutes[keep]
        order = np.lexsort((minutes, head, tail))
        tail, head, minutes = tail[order], head[order], minutes[order]
        first = np.ones(len(tail), bool)
        first[1:] = (tail[1:] != tail[:-1]) | (head[1:] != head[:-1])
        self._graph = csr_array(
            (minutes[first], (tail[first], head[first])), shape=(n_nodes, n_nodes)
        )

    def minutes(
        self, sources: NDArray[np.int64], targets: NDArray[np.int64], band: TravelBand
    ) -> NDArray[np.float64]:
        graph = csr_array(
            (self._graph.data / band.factor, self._graph.indices, self._graph.indptr),
            shape=self._graph.shape,
        )
        distances = dijkstra(graph, directed=True, indices=sources)
        return distances[:, targets]
