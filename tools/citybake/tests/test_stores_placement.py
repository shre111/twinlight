from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from twl_citybake.stores.placement import (
    Placement,
    candidate_nodes,
    main_component,
    place_stores,
)

LOCAL = 4


def roads(
    n: int, edges: list[tuple[int, int, int]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    nodes = {
        "osm_id": np.arange(n, dtype=np.int64),
        "x": np.arange(n, dtype=np.float64),
        "y": np.zeros(n, np.float64),
    }
    table = {
        "u": np.array([e[0] for e in edges], np.uint32),
        "v": np.array([e[1] for e in edges], np.uint32),
        "length_m": np.full(len(edges), 100.0, np.float32),
        "road_class": np.full(len(edges), LOCAL, np.uint8),
        "oneway": np.array([e[2] for e in edges], np.int8),
        "bridge": np.zeros(len(edges), np.uint8),
        "osm_way": np.arange(len(edges), dtype=np.int64),
    }
    return nodes, table


def test_main_component_is_strong_and_excludes_dead_ends_entries_and_islands() -> None:
    nodes, edges = roads(
        9,
        [
            (0, 1, 0),
            (2, 1, -1),
            (2, 0, 1),
            (2, 3, 1),
            (4, 0, 1),
            (5, 6, 0),
            (7, 8, 1),
        ],
    )
    mask = main_component(nodes, edges)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, True, False, False, False, False, False, False]


def test_main_component_reads_a_negative_oneway_as_v_to_u() -> None:
    nodes, edges = roads(3, [(0, 1, 0), (1, 2, 1), (0, 2, -1)])
    assert main_component(nodes, edges).tolist() == [True, True, True]
    nodes, edges = roads(3, [(0, 1, 0), (1, 2, 1), (2, 0, -1)])
    assert main_component(nodes, edges).tolist() == [True, True, False]


def test_main_component_picks_the_larger_of_two_strong_components() -> None:
    nodes, edges = roads(5, [(0, 1, 0), (2, 3, 0), (3, 4, 0)])
    assert main_component(nodes, edges).tolist() == [False, False, True, True, True]


CELL = 100.0
CAND_X = np.array([10, 45, 80, 140, 160, 150, -30, 30, -100, 250], np.float64)
CAND_Y = np.array([10, 52, 80, 50, 50, 50, 40, 40, 0, 50], np.float64)
CAND_IN = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0], bool)


def test_candidate_nodes_take_the_node_nearest_each_cell_centre_in_the_component() -> None:
    picked = candidate_nodes(CAND_X, CAND_Y, CAND_IN, CELL)
    assert picked.dtype == np.int64
    assert sorted(picked.tolist()) == [1, 3, 6]


def test_candidate_nodes_break_ties_to_the_lowest_node_id_whatever_the_order() -> None:
    x = np.array([160.0, 140.0, 150.0, 150.0])
    y = np.array([50.0, 50.0, 40.0, 60.0])
    in_component = np.array([1, 1, 0, 0], bool)
    assert candidate_nodes(x, y, in_component, CELL).tolist() == [0]
    assert candidate_nodes(x[[1, 0]], y[[1, 0]], in_component[:2], CELL).tolist() == [0]


def test_candidate_nodes_keep_one_node_per_cell_across_the_origin() -> None:
    x = np.array([-10.0, 10.0, -90.0, 90.0])
    y = np.array([10.0, 10.0, 60.0, 60.0])
    picked = candidate_nodes(x, y, np.ones(4, bool), CELL)
    assert sorted(picked.tolist()) == [2, 3]


def test_candidate_nodes_skip_cells_with_no_component_node() -> None:
    x = np.array([50.0, 250.0])
    y = np.array([50.0, 50.0])
    assert candidate_nodes(x, y, np.array([False, True]), CELL).tolist() == [1]
    assert len(candidate_nodes(x, y, np.zeros(2, bool), CELL)) == 0


RADIUS = 100.0
RES_X = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 5000.0])
RES_Y = np.zeros(6)
RES_W = np.array([1.5, 1.0, 1.0, 1.0, 1.0, 2.0], np.float32)
TOTAL = 7.5
#            P      S      Q       Z
CAND_PX = np.array([30.0, 150.0, 900.0, 5000.0])
CAND_PN = np.array([10, 11, 12, 20], np.uint32)


def greedy(**changes: Any) -> Placement:
    args: dict[str, Any] = {
        "res_x": RES_X,
        "res_y": RES_Y,
        "res_weight": RES_W,
        "cand_x": CAND_PX,
        "cand_y": np.zeros(len(CAND_PX)),
        "cand_node": CAND_PN,
        "radius_m": RADIUS,
        "coverage_target": 1.0,
        "max_stores": 10,
    }
    args.update(changes)
    return place_stores(**args)


def test_greedy_updates_gains_after_each_pick_not_the_initial_ranking() -> None:
    placed = greedy(max_stores=2)
    assert placed.node.tolist() == [11, 20]
    np.testing.assert_allclose(placed.covered_share, [4 / TOTAL, 6 / TOTAL])


def test_greedy_runs_to_full_coverage_in_gain_order() -> None:
    placed = greedy()
    assert placed.node.tolist() == [11, 20, 10]
    np.testing.assert_allclose(placed.covered_share, [4 / TOTAL, 6 / TOTAL, 1.0])


def test_greedy_is_not_nearest_to_the_resident_centroid() -> None:
    centroid = float((RES_X * RES_W).sum() / RES_W.sum())
    nearest = int(CAND_PN[np.argmin(np.abs(CAND_PX - centroid))])
    assert nearest == 12
    assert greedy().node[0] != nearest


def test_weights_decide_the_pick_where_resident_counts_would_tie() -> None:
    placed = greedy(max_stores=2)
    assert 10 not in placed.node.tolist()
    unweighted = greedy(res_weight=np.ones(6, np.float32), max_stores=2)
    assert unweighted.node.tolist() == [11, 10]


def test_placement_reports_the_candidate_coordinates_of_each_store() -> None:
    placed = greedy()
    assert placed.x.dtype == np.float32 and placed.y.dtype == np.float32
    assert placed.node.dtype == np.uint32
    assert placed.x.tolist() == [150.0, 5000.0, 30.0]
    assert placed.y.tolist() == [0.0, 0.0, 0.0]
    assert placed.covered_share.dtype == np.float64


def test_placement_stops_once_the_coverage_target_is_reached() -> None:
    placed = greedy(coverage_target=0.5)
    assert placed.node.tolist() == [11]
    placed = greedy(coverage_target=0.55)
    assert placed.node.tolist() == [11, 20]


def test_placement_stops_at_max_stores_below_the_target() -> None:
    placed = greedy(max_stores=1)
    assert placed.node.tolist() == [11]
    assert placed.covered_share[-1] < 1.0


def test_placement_stops_when_no_candidate_adds_coverage() -> None:
    res_x = np.append(RES_X, 9000.0)
    res_w = np.append(RES_W, np.float32(1.0))
    placed = greedy(res_x=res_x, res_y=np.zeros(7), res_weight=res_w)
    assert placed.node.tolist() == [11, 20, 10]
    np.testing.assert_allclose(placed.covered_share[-1], TOTAL / (TOTAL + 1.0))


def test_a_candidate_adding_nothing_is_never_placed_even_with_room_left() -> None:
    placed = greedy(coverage_target=1.0, max_stores=50)
    assert 12 not in placed.node.tolist()
    assert len(placed.node) == 3


def test_equal_gain_goes_to_the_lowest_node_id_in_any_candidate_order() -> None:
    res_x = np.array([0.0, 1000.0])
    cand_x = np.array([0.0, 1000.0, 0.0])
    nodes = np.array([30, 10, 20], np.uint32)
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        placed = place_stores(
            res_x,
            np.zeros(2),
            np.ones(2, np.float32),
            cand_x[order],
            np.zeros(3),
            nodes[order],
            RADIUS,
            1.0,
            10,
        )
        assert placed.node.tolist() == [10, 20]


def test_placement_is_identical_on_rerun() -> None:
    a, b = greedy(), greedy()
    for name in ("node", "x", "y", "covered_share"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def reference_greedy(
    res: np.ndarray,
    weight: np.ndarray,
    cand: np.ndarray,
    node: np.ndarray,
    radius: float,
    target: float,
    max_stores: int,
) -> tuple[list[int], list[float]]:
    d = np.hypot(cand[:, None, 0] - res[None, :, 0], cand[:, None, 1] - res[None, :, 1])
    within = d <= radius
    covered = np.zeros(len(res), bool)
    total = float(weight.sum())
    chosen: list[int] = []
    shares: list[float] = []
    while len(chosen) < max_stores and (not shares or shares[-1] < target):
        gain = (within & ~covered[None, :]).astype(np.float64) @ weight.astype(np.float64)
        best = gain.max()
        if best <= 0:
            break
        pick = min(np.flatnonzero(gain == best), key=lambda i: node[i])
        chosen.append(int(node[pick]))
        covered |= within[pick]
        shares.append(float(weight[covered].sum()) / total)
    return chosen, shares


@settings(max_examples=60, deadline=None)
@given(
    res=st.lists(
        st.tuples(st.integers(0, 40), st.integers(0, 40), st.integers(1, 5)),
        min_size=1,
        max_size=40,
    ),
    cand=st.lists(st.tuples(st.integers(0, 40), st.integers(0, 40)), min_size=1, max_size=15),
    radius=st.integers(1, 15),
    target=st.sampled_from([0.3, 0.6, 0.9, 1.0]),
    max_stores=st.integers(1, 6),
    seed=st.integers(0, 1000),
)
def test_placement_equals_a_brute_force_greedy(
    res: list[tuple[int, int, int]],
    cand: list[tuple[int, int]],
    radius: int,
    target: float,
    max_stores: int,
    seed: int,
) -> None:
    rx = np.array([r[0] for r in res], np.float64)
    ry = np.array([r[1] for r in res], np.float64)
    weight = np.array([r[2] for r in res], np.float32)
    cx = np.array([c[0] for c in cand], np.float64)
    cy = np.array([c[1] for c in cand], np.float64)
    node = np.random.default_rng(seed).permutation(len(cand)).astype(np.uint32)
    placed = place_stores(rx, ry, weight, cx, cy, node, float(radius) + 0.5, target, max_stores)
    want_nodes, want_shares = reference_greedy(
        np.column_stack([rx, ry]),
        weight,
        np.column_stack([cx, cy]),
        node,
        float(radius) + 0.5,
        target,
        max_stores,
    )
    assert placed.node.tolist() == want_nodes
    np.testing.assert_allclose(placed.covered_share, want_shares, rtol=1e-9)
    assert (np.diff(placed.covered_share) > 0).all()
    assert len(set(placed.node.tolist())) == len(placed.node)
