import math

import numpy as np
import pytest
from osm_fixture import KEPT_ROAD_NODE_IDS, ROAD_NODES, ROAD_WAYS

from twl_citybake.osm.extract import OsmLayers
from twl_citybake.osm.tags import RoadClass

KEPT_WAYS = [100, 101, 102, 110, 120, 130, 131, 140]
EDGES_PER_WAY = {100: 2, 101: 1, 102: 2, 110: 1, 120: 2, 130: 1, 131: 1, 140: 1}


def node_xy(layers: OsmLayers, index: int) -> tuple[float, float]:
    nodes = layers.road_nodes
    return float(nodes["x"][index]), float(nodes["y"][index])


def edges_of(layers: OsmLayers, way: int) -> list[int]:
    return np.flatnonzero(layers.road_edges["osm_way"] == way).tolist()


def polyline(layers: OsmLayers, edge: int) -> np.ndarray:
    offsets = layers.road_geom["offsets"]
    return layers.road_geom["xy"][offsets[edge] : offsets[edge + 1]].astype(np.float64)


def test_node_and_edge_columns_have_the_specified_dtypes(layers: OsmLayers) -> None:
    nodes, edges = layers.road_nodes, layers.road_edges
    assert (nodes["osm_id"].dtype, nodes["x"].dtype, nodes["y"].dtype) == (
        np.int64,
        np.float64,
        np.float64,
    )
    expected = {
        "u": np.uint32,
        "v": np.uint32,
        "length_m": np.float32,
        "road_class": np.uint8,
        "oneway": np.int8,
        "bridge": np.uint8,
        "osm_way": np.int64,
    }
    assert {name: edges[name].dtype.type for name in expected} == expected


def test_graph_nodes_are_endpoints_and_shared_nodes_only(layers: OsmLayers) -> None:
    assert layers.road_nodes["osm_id"].tolist() == sorted(KEPT_ROAD_NODE_IDS)


def test_nodes_are_renumbered_in_ascending_osm_id(layers: OsmLayers) -> None:
    ids = layers.road_nodes["osm_id"]
    assert np.all(np.diff(ids) > 0)
    for index, osm_id in enumerate(ids.tolist()):
        assert node_xy(layers, index) == pytest.approx(ROAD_NODES[osm_id], abs=0.05)


def test_interior_node_used_by_one_way_and_a_footway_is_not_a_graph_node(
    layers: OsmLayers,
) -> None:
    assert 5013 not in layers.road_nodes["osm_id"].tolist()


def test_ways_split_at_t_junction_and_crossing(layers: OsmLayers) -> None:
    counts = {way: len(edges_of(layers, way)) for way in KEPT_WAYS}
    assert counts == EDGES_PER_WAY
    assert len(layers.road_edges["osm_way"]) == sum(EDGES_PER_WAY.values())


def test_t_junction_node_is_shared_by_both_ways(layers: OsmLayers) -> None:
    ids = layers.road_nodes["osm_id"]
    junction = int(np.flatnonzero(ids == 5005)[0])
    edges = layers.road_edges
    touching = {
        int(edges["osm_way"][e])
        for e in range(len(edges["u"]))
        if junction in (int(edges["u"][e]), int(edges["v"][e]))
    }
    assert touching == {100, 110}


def test_crossing_node_joins_four_edges(layers: OsmLayers) -> None:
    ids = layers.road_nodes["osm_id"]
    crossing = int(np.flatnonzero(ids == 5002)[0])
    edges = layers.road_edges
    degree = int(np.sum(edges["u"] == crossing) + np.sum(edges["v"] == crossing))
    assert degree == 4


def test_footway_is_excluded(layers: OsmLayers) -> None:
    assert edges_of(layers, 150) == []
    assert 5014 not in layers.road_nodes["osm_id"].tolist()


def test_edge_leaving_the_bbox_is_dropped_with_its_outside_node(layers: OsmLayers) -> None:
    assert edges_of(layers, 160) == []
    assert 5019 not in layers.road_nodes["osm_id"].tolist()
    assert 5006 in layers.road_nodes["osm_id"].tolist()


def test_edges_follow_way_order_and_are_sorted_by_way_then_position(layers: OsmLayers) -> None:
    ways = layers.road_edges["osm_way"]
    assert np.all(np.diff(ways) >= 0)

    for way in (100, 102, 120):
        first, second = edges_of(layers, way)
        assert layers.road_edges["v"][first] == layers.road_edges["u"][second]


def test_edge_endpoints_follow_way_direction(layers: OsmLayers) -> None:
    ids = layers.road_nodes["osm_id"]
    edges = layers.road_edges
    for way, (refs, _) in ROAD_WAYS.items():
        if way not in EDGES_PER_WAY:
            continue
        chain = [(int(ids[edges["u"][e]]), int(ids[edges["v"][e]])) for e in edges_of(layers, way)]
        assert chain[0][0] == refs[0]
        assert chain[-1][1] == refs[-1]


def test_edges_reference_valid_nodes(layers: OsmLayers) -> None:
    n = len(layers.road_nodes["osm_id"])
    edges = layers.road_edges
    assert np.all(edges["u"] < n) and np.all(edges["v"] < n)
    assert np.all(edges["u"] != edges["v"])


def test_only_nodes_used_by_kept_edges_are_kept(layers: OsmLayers) -> None:
    n = len(layers.road_nodes["osm_id"])
    used = set(layers.road_edges["u"].tolist()) | set(layers.road_edges["v"].tolist())
    assert used == set(range(n))


def test_oneway_is_relative_to_edge_direction(layers: OsmLayers) -> None:
    edges = layers.road_edges
    assert edges["oneway"][edges_of(layers, 130)[0]] == 1
    assert edges["oneway"][edges_of(layers, 131)[0]] == -1
    for way in (100, 101, 102, 110, 120, 140):
        assert all(edges["oneway"][e] == 0 for e in edges_of(layers, way))


def test_bridge_flag_is_set_only_for_bridge_ways(layers: OsmLayers) -> None:
    edges = layers.road_edges
    assert edges["bridge"][edges_of(layers, 101)[0]] == 1
    for way in (100, 102, 110, 120, 130, 131, 140):
        assert all(edges["bridge"][e] == 0 for e in edges_of(layers, way))


def test_road_class_follows_highway_tag(layers: OsmLayers) -> None:
    expected = {
        100: RoadClass.PRIMARY,
        101: RoadClass.PRIMARY,
        102: RoadClass.PRIMARY,
        110: RoadClass.LOCAL,
        120: RoadClass.SECONDARY,
        130: RoadClass.LOCAL,
        140: RoadClass.SERVICE,
    }
    for way, road in expected.items():
        assert all(layers.road_edges["road_class"][e] == road for e in edges_of(layers, way))


def test_edge_length_follows_the_polyline_and_is_at_least_straight_line(
    layers: OsmLayers,
) -> None:
    edges = layers.road_edges
    for e in range(len(edges["u"])):
        u, v = node_xy(layers, int(edges["u"][e])), node_xy(layers, int(edges["v"][e]))
        straight = math.dist(u, v)
        along = float(np.hypot(*np.diff(polyline(layers, e), axis=0).T).sum())
        assert edges["length_m"][e] >= straight - 0.05
        assert edges["length_m"][e] == pytest.approx(along, abs=0.05)


def test_bent_edge_is_longer_than_its_chord(layers: OsmLayers) -> None:
    e = edges_of(layers, 100)[0]
    expected = math.dist((-900, 0), (-450, 50))
    assert layers.road_edges["length_m"][e] == pytest.approx(expected, abs=0.1)

    bent = edges_of(layers, 120)[1]
    bent_expected = math.dist((450, 0), (480, 300)) + math.dist((480, 300), (450, 600))
    assert layers.road_edges["length_m"][bent] == pytest.approx(bent_expected, abs=0.1)
    assert len(polyline(layers, bent)) == 3


def test_polylines_start_and_end_at_their_nodes(layers: OsmLayers) -> None:
    edges = layers.road_edges
    for e in range(len(edges["u"])):
        line = polyline(layers, e)
        assert len(line) >= 2
        assert tuple(line[0]) == pytest.approx(node_xy(layers, int(edges["u"][e])), abs=0.1)
        assert tuple(line[-1]) == pytest.approx(node_xy(layers, int(edges["v"][e])), abs=0.1)


def test_geometry_offsets_are_monotone_and_terminal_offset_equals_vertex_count(
    layers: OsmLayers,
) -> None:
    offsets = layers.road_geom["offsets"]
    xy = layers.road_geom["xy"]

    assert offsets.dtype == np.uint32
    assert xy.dtype == np.float32
    assert xy.ndim == 2 and xy.shape[1] == 2
    assert len(offsets) == len(layers.road_edges["u"]) + 1
    assert offsets[0] == 0
    assert np.all(np.diff(offsets.astype(np.int64)) >= 2)
    assert offsets[-1] == len(xy)


def test_all_roads_lie_inside_the_clip(layers: OsmLayers) -> None:
    xy = layers.road_geom["xy"]
    assert np.abs(xy[:, 0]).max() < 1010
    assert np.abs(xy[:, 1]).max() < 1010


def test_road_report_counts_components_and_lengths(layers: OsmLayers) -> None:
    roads = layers.report["roads"]
    assert roads["nodes"] == len(KEPT_ROAD_NODE_IDS)
    assert roads["edges"] == sum(EDGES_PER_WAY.values())
    assert roads["components"] == 2
    assert 0.5 < roads["largest_component_share"] < 1.0
    total_km = float(layers.road_edges["length_m"].sum()) / 1000
    assert sum(roads["length_km_by_class"].values()) == pytest.approx(total_km, abs=0.01)
