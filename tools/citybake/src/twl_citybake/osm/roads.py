from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import osmium
import shapely
from osmium.osm import Way
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import Polygon

from twl_citybake.osm.tags import RoadClass, oneway, road_class
from twl_citybake.projection import LocalProjection


@dataclass(frozen=True)
class RoadGraph:
    nodes: dict[str, np.ndarray]
    edges: dict[str, np.ndarray]
    geom: dict[str, np.ndarray]

    def components(self) -> npt.NDArray[np.int32]:
        """Weakly connected component label of every node."""
        n = len(self.nodes["osm_id"])
        u, v = self.edges["u"], self.edges["v"]
        adjacency = coo_matrix((np.ones(len(u), np.int8), (u, v)), shape=(n, n)).tocsr()
        _, labels = connected_components(adjacency, directed=True, connection="weak")
        return labels.astype(np.int32)


@dataclass
class RoadCollector:
    way_id: list[int] = field(default_factory=list)
    size: list[int] = field(default_factory=list)
    node_id: list[int] = field(default_factory=list)
    lon: list[float] = field(default_factory=list)
    lat: list[float] = field(default_factory=list)
    road_class: list[int] = field(default_factory=list)
    oneway: list[int] = field(default_factory=list)
    bridge: list[int] = field(default_factory=list)

    def add(self, way: Way) -> None:
        tags = way.tags
        highway = tags.get("highway")
        # area=yes highways are squares and parking aisles drawn as polygons, not centrelines
        if highway is None or tags.get("area") == "yes" or (cls := road_class(highway)) is None:
            return
        try:
            points = [(n.ref, n.lon, n.lat) for n in way.nodes]
        except osmium.InvalidLocationError:
            return
        if len(points) < 2:
            return
        refs, lons, lats = zip(*points, strict=True)
        flags = {key: tags[key] for key in ("oneway", "junction", "highway") if key in tags}
        self.way_id.append(way.id)
        self.size.append(len(points))
        self.node_id.extend(refs)
        self.lon.extend(lons)
        self.lat.extend(lats)
        self.road_class.append(cls)
        self.oneway.append(oneway(flags))
        self.bridge.append(int(tags.get("bridge", "no") != "no"))

    def finish(self, proj: LocalProjection, clip: Polygon) -> RoadGraph:
        way_id = np.asarray(self.way_id, np.int64)
        point_way = np.repeat(np.arange(len(way_id)), np.asarray(self.size, np.intp))
        order = np.argsort(way_id, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        point_order = np.lexsort((np.arange(len(point_way)), rank[point_way]))
        way = rank[point_way][point_order]
        node = np.asarray(self.node_id, np.int64)[point_order]
        lon = np.asarray(self.lon, np.float64)[point_order]
        lat = np.asarray(self.lat, np.float64)[point_order]
        keep = np.ones(len(node), bool)
        keep[1:] = (way[1:] != way[:-1]) | (node[1:] != node[:-1])
        way, node, lon, lat = way[keep], node[keep], lon[keep], lat[keep]
        x, y = proj.forward(lon, lat)
        per_way = {
            "road_class": np.asarray(self.road_class, np.uint8)[order],
            "oneway": np.asarray(self.oneway, np.int8)[order],
            "bridge": np.asarray(self.bridge, np.uint8)[order],
            "osm_way": way_id[order],
        }
        return _graph(way, node, x, y, per_way, clip)


def _graph_points(way: np.ndarray, node: np.ndarray) -> npt.NDArray[np.bool_]:
    """Way endpoints, nodes shared by several ways, and nodes a way visits twice."""
    n = len(node)
    first = np.ones(n, bool)
    first[1:] = way[1:] != way[:-1]
    last = np.ones(n, bool)
    last[:-1] = way[1:] != way[:-1]
    order = np.lexsort((way, node))
    node_s, way_s = node[order], way[order]
    new_node = np.ones(n, bool)
    new_node[1:] = node_s[1:] != node_s[:-1]
    new_pair = new_node.copy()
    new_pair[1:] |= way_s[1:] != way_s[:-1]
    pair = np.cumsum(new_pair) - 1
    group = np.cumsum(new_node) - 1
    ways_per_node = np.bincount(group, weights=new_pair)
    junction = (np.bincount(pair)[pair] >= 2) | (ways_per_node[group] >= 2)
    is_graph = first | last
    is_graph[order] |= junction
    return is_graph


def _graph(
    way: np.ndarray,
    node: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    per_way: dict[str, np.ndarray],
    clip: Polygon,
) -> RoadGraph:
    points = np.flatnonzero(_graph_points(way, node)) if len(node) else np.empty(0, np.intp)
    same_way = way[points[:-1]] == way[points[1:]]
    start, end = points[:-1][same_way], points[1:][same_way]
    segment = np.hypot(np.diff(x), np.diff(y))
    segment[way[1:] != way[:-1]] = 0.0
    length = np.add.reduceat(segment, start) if len(start) else np.empty(0)
    inside = shapely.contains_xy(clip, x, y)
    kept = inside[start] & inside[end]
    start, end, length = start[kept], end[kept], length[kept]
    edge_way = way[start]

    ends = np.concatenate([start, end])
    osm_id, first_seen = np.unique(node[ends], return_index=True)
    node_at = ends[first_seen]
    edges = {
        "u": np.searchsorted(osm_id, node[start]).astype(np.uint32),
        "v": np.searchsorted(osm_id, node[end]).astype(np.uint32),
        "length_m": _ceil_float32(length),
        **{key: values[edge_way] for key, values in per_way.items()},
    }
    return RoadGraph(
        nodes={"osm_id": osm_id, "x": x[node_at], "y": y[node_at]},
        edges=edges,
        geom=_polylines(start, end, x, y),
    )


def _ceil_float32(values: np.ndarray) -> npt.NDArray[np.float32]:
    """Round up so a stored length never undercuts the float64 distance it was measured from."""
    rounded = values.astype(np.float32)
    low = rounded.astype(np.float64) < values
    rounded[low] = np.nextafter(rounded[low], np.float32(np.inf))
    return rounded


def _polylines(start: np.ndarray, end: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    counts = end - start + 1
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    index = np.repeat(start - offsets[:-1].astype(np.intp), counts) + np.arange(offsets[-1])
    xy = np.column_stack([x[index], y[index]]).astype(np.float32).reshape(-1, 2)
    return {"offsets": offsets, "xy": xy}


def length_km_by_class(edges: dict[str, np.ndarray]) -> dict[str, float]:
    totals = np.bincount(
        edges["road_class"], weights=edges["length_m"].astype(np.float64), minlength=len(RoadClass)
    )
    return {cls.name: round(float(totals[cls]) / 1000, 3) for cls in RoadClass}
