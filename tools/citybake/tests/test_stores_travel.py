from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from stores_fixture import Edge, edge_columns

from twl_citybake.config import TravelBand
from twl_citybake.osm.tags import RoadClass
from twl_citybake.stores.travel import GraphTravelBackend

MAJOR, PRIMARY, SECONDARY, TERTIARY, LOCAL, SERVICE = (int(c) for c in RoadClass)
#  m/min:                1000      500        333.33       250     200   100
SPEEDS = {
    "major": 60.0,
    "primary": 30.0,
    "secondary": 20.0,
    "tertiary": 15.0,
    "local": 12.0,
    "service": 6.0,
}
BASE = TravelBand(name="base", start_minute=0, factor=1.0)


def backend(n: int, edges: Sequence[Edge]) -> GraphTravelBackend:
    return GraphTravelBackend(n, edge_columns(edges), SPEEDS)


def pairs(
    edges: Sequence[Edge], n: int, band: TravelBand = BASE
) -> np.ndarray[Any, np.dtype[np.float64]]:
    everyone = np.arange(n, dtype=np.int64)
    return backend(n, edges).minutes(everyone, everyone, band)


def test_backend_is_named_after_its_method() -> None:
    assert backend(2, [(0, 1, 100.0, LOCAL, 0)]).name == "graph-dijkstra"


def test_minutes_follow_the_fastest_route_by_time_not_by_length() -> None:
    edges: list[Edge] = [
        (0, 1, 1000.0, MAJOR, 0),
        (1, 2, 500.0, PRIMARY, 0),
        (2, 3, 1000.0, TERTIARY, 0),
        (0, 3, 2000.0, LOCAL, 0),
    ]
    m = pairs(edges, 4)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[0, 2] == pytest.approx(2.0)
    assert m[0, 3] == pytest.approx(6.0)
    assert m[3, 0] == pytest.approx(6.0)
    assert m[1, 3] == pytest.approx(5.0)
    assert (np.diag(m) == 0).all()


def test_result_has_one_row_per_source_and_one_column_per_target_in_the_order_given() -> None:
    edges: list[Edge] = [
        (0, 1, 1000.0, MAJOR, 0),
        (1, 2, 500.0, PRIMARY, 0),
        (2, 3, 1000.0, TERTIARY, 0),
    ]
    sources = np.array([3, 0], np.int64)
    targets = np.array([2, 0, 3, 2], np.int64)
    out = backend(4, edges).minutes(sources, targets, BASE)
    assert out.shape == (2, 4)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[4.0, 6.0, 0.0, 4.0], [2.0, 0.0, 6.0, 2.0]])


def test_a_oneway_edge_is_travelled_only_from_u_to_v_and_the_return_takes_the_detour() -> None:
    edges: list[Edge] = [
        (0, 1, 1000.0, MAJOR, 1),
        (1, 2, 1000.0, MAJOR, 0),
        (2, 0, 2000.0, MAJOR, 0),
    ]
    m = pairs(edges, 3)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx(3.0)


def test_a_negative_oneway_edge_is_travelled_only_from_v_to_u() -> None:
    edges: list[Edge] = [
        (0, 1, 1000.0, MAJOR, -1),
        (0, 2, 2000.0, MAJOR, 0),
        (2, 1, 1000.0, MAJOR, 0),
    ]
    m = pairs(edges, 3)
    assert m[1, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(3.0)


def test_oneway_edges_with_no_return_leave_the_far_side_unreachable() -> None:
    edges: list[Edge] = [(0, 1, 1000.0, MAJOR, 1), (3, 2, 1000.0, MAJOR, -1)]
    m = pairs(edges, 4)
    assert m[0, 1] == pytest.approx(1.0)
    assert np.isinf(m[1, 0])
    assert m[2, 3] == pytest.approx(1.0)
    assert np.isinf(m[3, 2])


def test_nodes_in_other_components_are_infinite() -> None:
    m = pairs([(0, 1, 1000.0, MAJOR, 0), (2, 3, 1000.0, MAJOR, 0)], 4)
    assert np.isinf(m[0, 2]) and np.isinf(m[3, 1])
    assert m[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize("slow_first", [True, False])
def test_the_fastest_of_parallel_edges_wins_wherever_it_appears(slow_first: bool) -> None:
    slow: Edge = (0, 1, 2000.0, LOCAL, 0)
    fast: Edge = (0, 1, 1000.0, MAJOR, 0)
    edges = [slow, fast] if slow_first else [fast, slow]
    m = pairs(edges, 2)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx(1.0)


def test_parallel_edges_are_not_summed_and_a_reversed_duplicate_still_competes() -> None:
    edges: list[Edge] = [
        (0, 1, 2000.0, LOCAL, 0),
        (1, 0, 1000.0, MAJOR, 0),
        (0, 1, 1500.0, MAJOR, 0),
    ]
    m = pairs(edges, 2)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx(1.0)


def test_parallel_edges_of_different_direction_keep_each_direction_separately() -> None:
    edges: list[Edge] = [(0, 1, 2000.0, LOCAL, 0), (0, 1, 1000.0, MAJOR, 1)]
    m = pairs(edges, 2)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("road_class", "minutes"),
    [
        (MAJOR, 1.0),
        (PRIMARY, 2.0),
        (SECONDARY, 3.0),
        (TERTIARY, 4.0),
        (LOCAL, 5.0),
        (SERVICE, 10.0),
    ],
)
def test_each_road_class_uses_its_own_speed(road_class: int, minutes: float) -> None:
    m = pairs([(0, 1, 1000.0, road_class, 0)], 2)
    assert m[0, 1] == pytest.approx(minutes, rel=1e-6)


def test_a_slow_class_is_avoided_when_a_longer_faster_route_beats_it() -> None:
    edges: list[Edge] = [
        (0, 1, 1000.0, SERVICE, 0),
        (0, 2, 2500.0, MAJOR, 0),
        (2, 1, 2500.0, MAJOR, 0),
    ]
    assert pairs(edges, 3)[0, 1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(1.0, 6.0), (0.5, 12.0), (2.0, 3.0), (1.25, 4.8), (0.8, 7.5), (0.7, 6.0 / 0.7)],
)
def test_band_factor_divides_the_speed_so_times_scale_exactly(
    factor: float, expected: float
) -> None:
    edges: list[Edge] = [
        (0, 1, 1000.0, MAJOR, 0),
        (1, 2, 500.0, PRIMARY, 0),
        (2, 3, 1000.0, TERTIARY, 0),
    ]
    band = TravelBand(name="b", start_minute=0, factor=factor)
    assert pairs(edges, 4, band)[0, 3] == pytest.approx(expected, rel=1e-9)


def test_one_backend_serves_several_bands_without_leaking_between_them() -> None:
    b = backend(2, [(0, 1, 1000.0, MAJOR, 0)])
    s, t = np.array([0], np.int64), np.array([1], np.int64)
    slow = b.minutes(s, t, TravelBand(name="slow", start_minute=0, factor=0.5))
    fast = b.minutes(s, t, TravelBand(name="fast", start_minute=60, factor=2.0))
    again = b.minutes(s, t, TravelBand(name="slow", start_minute=0, factor=0.5))
    assert slow[0, 0] == pytest.approx(2.0)
    assert fast[0, 0] == pytest.approx(0.5)
    assert again[0, 0] == pytest.approx(2.0)


def test_self_loops_are_ignored() -> None:
    edges: list[Edge] = [
        (0, 0, 100.0, MAJOR, 0),
        (1, 1, 0.0, MAJOR, 1),
        (0, 1, 1000.0, MAJOR, 0),
        (2, 2, 500.0, MAJOR, 0),
    ]
    m = pairs(edges, 3)
    assert m[0, 0] == 0.0 and m[1, 1] == 0.0 and m[2, 2] == 0.0
    assert m[0, 1] == pytest.approx(1.0)
    assert np.isinf(m[0, 2])


def test_a_zero_length_edge_still_joins_its_nodes() -> None:
    m = pairs([(0, 1, 0.0, LOCAL, 0), (1, 2, 1000.0, MAJOR, 0)], 3)
    assert m[0, 1] == 0.0
    assert m[0, 2] == pytest.approx(1.0)


def test_isolated_nodes_and_an_edgeless_graph_are_valid() -> None:
    m = backend(3, []).minutes(np.arange(3, dtype=np.int64), np.arange(3, dtype=np.int64), BASE)
    assert (np.diag(m) == 0).all()
    assert np.isinf(m[0, 1])
