from itertools import combinations

import numpy as np
from graph_fixture import (
    FRIEND,
    HOUSEHOLD,
    NEIGHBOUR,
    NONE,
    WORK,
    Triple,
    World,
    assert_csr_invariants,
    build_graph,
    building_table,
    edges,
    merge,
    pair_distance,
    run_work,
    triple,
    uniform_world,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial import KDTree

from twl_citybake.graph.ties import (
    friend_ties,
    household_ties,
    neighbour_ties,
    workplace_masks,
)
from twl_citybake.osm.tags import BuildingKind


def rng(seed: int = 1) -> np.random.Generator:
    return np.random.default_rng(seed)


def neighbours(world: World, seed: int = 1, **overrides: float) -> Triple:
    params = {"degree": 1.6, "scale_m": 250.0, "candidates": 64, **overrides}
    return neighbour_ties(
        x=world.x,
        y=world.y,
        household=world.household,
        degree=params["degree"],
        scale_m=params["scale_m"],
        candidates=int(params["candidates"]),
        rng=rng(seed),
    )


def friends(world: World, seed: int = 1, **overrides: float) -> Triple:
    params = {"degree": 1.2, "scale_m": 3000.0, "homophily": 4.0, "candidates": 16, **overrides}
    return friend_ties(
        x=world.x,
        y=world.y,
        household=world.household,
        segment=world.segment,
        degree=params["degree"],
        scale_m=params["scale_m"],
        homophily=params["homophily"],
        candidates=int(params["candidates"]),
        rng=rng(seed),
    )


def assert_pair_shape(t: Triple, n: int) -> None:
    u, v, s = t
    assert (u.dtype, v.dtype, s.dtype) == (np.uint32, np.uint32, np.float32)
    assert len(u) == len(v) == len(s)
    assert (u < v).all()
    assert int(v.max(initial=0)) < n
    assert (s > 0).all() and (s <= 1).all()


def test_household_ties_make_a_clique_per_household_and_none_for_singletons() -> None:
    household = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3], np.uint32)
    spouse = np.full(10, NONE, np.uint32)
    u, v, _ = household_ties(household, spouse)
    assert_pair_shape((u, v, _), 10)
    expected = {(1, 2), (1, 3), (2, 3), (4, 5), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)}
    assert set(zip(u.tolist(), v.tolist(), strict=True)) == expected
    assert len(u) == len(expected)
    assert 0 not in u and 0 not in v


def test_household_ties_give_spouses_full_strength_and_others_point_eight() -> None:
    household = np.array([0, 0, 0, 1, 1], np.uint32)
    spouse = np.array([1, 0, NONE, 4, 3], np.uint32)
    u, v, s = household_ties(household, spouse)
    strength = {(int(a), int(b)): float(w) for a, b, w in zip(u, v, s, strict=True)}
    assert strength == {(0, 1): 1.0, (0, 2): np.float32(0.8), (1, 2): np.float32(0.8), (3, 4): 1.0}


def test_household_ties_of_all_singletons_are_empty() -> None:
    u, v, s = household_ties(np.arange(5, dtype=np.uint32), np.full(5, NONE, np.uint32))
    assert len(u) == len(v) == len(s) == 0


def test_neighbour_mean_degree_is_near_the_target() -> None:
    world = uniform_world(6000, household_size=3)
    u, v, s = neighbours(world)
    assert_pair_shape((u, v, s), world.n)
    assert 1.45 <= 2 * len(u) / world.n <= 1.65


def test_neighbour_ties_never_join_a_household_or_a_resident_to_themselves() -> None:
    world = uniform_world(3000, household_size=4, extent_m=1500.0)
    u, v, _ = neighbours(world)
    assert len(u) > 1000
    assert (u != v).all()
    assert (world.household[u] != world.household[v]).all()


def test_neighbours_are_drawn_from_the_nearest_candidates_only() -> None:
    world = uniform_world(3000, extent_m=3000.0)
    u, v, _ = neighbours(world)
    points = np.column_stack([world.x, world.y])
    reach = KDTree(points).query(points, k=65)[0][:, -1]
    distance = pair_distance(world.x, world.y, u, v)
    assert (distance <= np.maximum(reach[u], reach[v]) + 1e-6).all()


def test_nearer_candidates_are_chosen_more_often_as_the_scale_shrinks() -> None:
    world = uniform_world(5000, extent_m=3000.0)
    points = np.column_stack([world.x, world.y])
    baseline = float(KDTree(points).query(points, k=65)[0][:, 1:].mean())

    def mean_distance(scale_m: float) -> float:
        u, v, _ = neighbours(world, scale_m=scale_m)
        return float(pair_distance(world.x, world.y, u, v).mean())

    assert 0.95 * baseline <= mean_distance(1e9) <= 1.05 * baseline
    assert mean_distance(40.0) < 0.75 * baseline
    assert mean_distance(40.0) < mean_distance(100.0) < mean_distance(1e9)


def test_neighbour_strength_falls_with_distance_and_is_clipped() -> None:
    world = uniform_world(3000, extent_m=6000.0)
    u, v, s = neighbours(world, scale_m=1000.0)
    distance = pair_distance(world.x, world.y, u, v)
    expected = np.clip(0.5 * np.exp(-distance / 500.0), 0.1, 0.5)
    np.testing.assert_allclose(s, expected, rtol=1e-4, atol=1e-5)
    assert s.min() >= np.float32(0.1) and s.max() <= np.float32(0.5)


def test_neighbour_ties_are_deterministic_per_seed_and_differ_between_seeds() -> None:
    world = uniform_world(1500)
    a, b, c = neighbours(world, seed=3), neighbours(world, seed=3), neighbours(world, seed=4)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    assert (
        len(a[0]) != len(c[0]) or not np.array_equal(a[0], c[0]) or not np.array_equal(a[1], c[1])
    )


def test_zero_neighbour_degree_gives_no_ties() -> None:
    world = uniform_world(500)
    assert len(neighbours(world, degree=0.0)[0]) == 0


def expected_same_segment(segment: np.ndarray, homophily: float) -> float:
    share = np.bincount(segment) / len(segment)
    return float((share**2 * homophily / (share * homophily + 1 - share)).sum())


def same_segment_share(world: World, t: Triple) -> float:
    return float((world.segment[t[0]] == world.segment[t[1]]).mean())


def test_friend_same_segment_share_matches_the_homophily_formula() -> None:
    world = uniform_world(8000, extent_m=6000.0, household_size=2)
    for homophily in (1.0, 4.0, 10.0):
        observed = same_segment_share(world, friends(world, homophily=homophily))
        assert abs(observed - expected_same_segment(world.segment, homophily)) < 0.03


def test_friend_same_segment_share_rises_with_homophily() -> None:
    world = uniform_world(6000, extent_m=6000.0, household_size=2)
    shares = [same_segment_share(world, friends(world, homophily=h)) for h in (1.0, 3.0, 9.0)]
    assert shares[0] + 0.05 < shares[1] < shares[2] - 0.05


def test_friend_mean_degree_is_near_the_target() -> None:
    world = uniform_world(8000, extent_m=6000.0, household_size=2)
    t = friends(world)
    assert_pair_shape(t, world.n)
    assert 1.08 <= 2 * len(t[0]) / world.n <= 1.30


def test_friend_ties_have_no_self_loops_and_almost_no_household_ties() -> None:
    world = uniform_world(8000, extent_m=6000.0, household_size=3)
    u, v, _ = friends(world)
    assert (u != v).all()
    assert (world.household[u] == world.household[v]).mean() < 0.005


def test_friend_ties_reach_across_the_whole_city() -> None:
    world = uniform_world(8000, extent_m=6000.0, household_size=2)
    u, v, _ = friends(world)
    distance = pair_distance(world.x, world.y, u, v)
    assert distance.max() > 5000.0
    assert (distance > 3000.0).mean() > 0.15
    flat_u, flat_v, _ = friends(world, scale_m=1e9)
    flat_mean = pair_distance(world.x, world.y, flat_u, flat_v).mean()
    assert flat_mean > 1.15 * distance.mean()
    assert 0.48 * 6000 < flat_mean < 0.57 * 6000


def test_friend_strength_is_between_point_four_and_point_eight() -> None:
    world = uniform_world(6000, extent_m=6000.0, household_size=2)
    s = friends(world)[2]
    assert s.min() >= np.float32(0.4) and s.max() <= np.float32(0.8)
    assert abs(float(s.mean()) - 0.6) < 0.02


def test_friend_ties_are_deterministic_per_seed() -> None:
    world = uniform_world(1500, household_size=2)
    a, b = friends(world, seed=5), friends(world, seed=5)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


WEST, EAST = 0, 12_000.0


def two_clusters(per_cluster: int = 600, seed: int = 2) -> tuple[World, np.ndarray]:
    r = np.random.default_rng(seed)
    n = 2 * per_cluster
    cx = np.repeat([WEST, EAST], per_cluster)
    x = cx + r.uniform(-300, 300, n)
    y = r.uniform(-300, 300, n)
    household = (np.arange(n) // 2).astype(np.uint32)
    segment = np.zeros(n, np.uint16)
    occupied = r.random(n) < 0.7
    return World(x, y, household, segment), occupied


def commercial(x: float, y: float, bank: int = 0, district: int = 0) -> tuple:
    return (x, y, int(BuildingKind.COMMERCIAL), 2000.0, 2, district, bank)


def test_work_ties_join_only_occupied_residents() -> None:
    world, occupied = two_clusters()
    buildings = building_table([commercial(0, 0), commercial(EAST, 0, bank=1)])
    (u, v, s), _ = run_work(world, occupied, buildings)
    assert len(u) > 200
    assert_pair_shape((u, v, s), world.n)
    assert occupied[u].all() and occupied[v].all()
    assert (world.household[u] != world.household[v]).all()


def test_work_strength_is_between_point_two_and_point_five() -> None:
    world, occupied = two_clusters()
    (_, _, s), _ = run_work(world, occupied, building_table([commercial(0, 0)]))
    assert s.min() >= np.float32(0.2) and s.max() <= np.float32(0.5)
    assert abs(float(s.mean()) - 0.35) < 0.02


def test_work_degree_per_occupied_resident_is_near_the_target() -> None:
    world, occupied = two_clusters()
    (u, _, _), _ = run_work(world, occupied, building_table([commercial(0, 0)]), degree=1.0)
    assert 0.8 <= 2 * len(u) / int(occupied.sum()) <= 1.1


def test_work_ties_pair_residents_who_chose_the_same_workplace_cell() -> None:
    world, occupied = two_clusters()
    buildings = building_table([commercial(0, 0), commercial(EAST, 0, bank=1)])
    (u, v, _), _ = run_work(world, occupied, buildings, scale_m=100.0)
    same_cluster = (world.x[u] > EAST / 2) == (world.x[v] > EAST / 2)
    assert same_cluster.mean() > 0.99


def test_work_ties_can_join_homes_far_apart_through_one_workplace() -> None:
    world, occupied = two_clusters()
    (u, v, _), _ = run_work(world, occupied, building_table([commercial(0, 0)]))
    distance = pair_distance(world.x, world.y, u, v)
    assert (distance > 10_000).mean() > 0.3


def test_work_ties_of_a_zero_degree_are_empty() -> None:
    world, occupied = two_clusters(100)
    (u, _, _), _ = run_work(world, occupied, building_table([commercial(0, 0)]), degree=0.0)
    assert len(u) == 0


def test_nobody_occupied_means_no_work_ties() -> None:
    world, occupied = two_clusters(100)
    (u, _, _), _ = run_work(world, np.zeros_like(occupied), building_table([commercial(0, 0)]))
    assert len(u) == 0


MIXED_BUILDINGS = [
    (0, 0, BuildingKind.COMMERCIAL, 200.0, 2, 0, 0),
    (600, 0, BuildingKind.INDUSTRIAL, 300.0, 1, 1, 1),
    (1200, 0, BuildingKind.CIVIC, 150.0, 5, 0, 0),
    (1800, 0, BuildingKind.UNKNOWN, 1500.0, 2, 1, 1),
    (2400, 0, BuildingKind.UNKNOWN, 1500.0, 5, 0, 0),
    (3000, 0, BuildingKind.UNKNOWN, 800.0, 1, 0, 0),
    (3600, 0, BuildingKind.RESIDENTIAL, 5000.0, 1, 0, 0),
    (4200, 0, BuildingKind.RELIGIOUS, 5000.0, 1, 0, 0),
    (4800, 0, BuildingKind.OTHER, 5000.0, 1, 0, 0),
    (5400, 0, BuildingKind.UNKNOWN, 1000.0, 3, 1, 0),
]


def test_workplace_summary_counts_follow_the_selection_rule() -> None:
    world, occupied = two_clusters(100)
    _, summary = run_work(
        world,
        occupied,
        building_table(MIXED_BUILDINGS),
    )
    assert summary["count"] == 5
    assert summary["by_source"] == {"osm_tagged": 3, "untagged_footprint": 2}
    assert summary["by_bank"] == {"west": 3, "east": 2}
    assert summary["by_district"] == {
        "Alpha": {"osm_tagged": 2, "untagged_footprint": 0},
        "Beta": {"osm_tagged": 1, "untagged_footprint": 2},
    }


def test_workplace_thresholds_come_from_the_arguments() -> None:
    world, occupied = two_clusters(100)
    table = building_table(MIXED_BUILDINGS)
    _, strict = run_work(world, occupied, table, min_area_m2=1600.0)
    _, tall = run_work(world, occupied, table, max_floors=5)
    assert strict["by_source"]["untagged_footprint"] == 0
    assert tall["by_source"]["untagged_footprint"] == 3


def test_workers_by_bank_follow_where_the_workplaces_are() -> None:
    world, occupied = two_clusters()
    east_only = building_table([commercial(EAST, 0, bank=1), commercial(EAST, 900, bank=1)])
    _, summary = run_work(world, occupied, east_only)
    workers = summary["workers_by_bank"]
    assert workers == {"west": 0, "east": int(occupied.sum())}


def test_a_workplace_at_exactly_the_limits_counts_and_one_beyond_does_not() -> None:
    table = building_table(
        [
            (0, 0, BuildingKind.UNKNOWN, 1000.0, 3, 0, 0),
            (0, 0, BuildingKind.UNKNOWN, 999.0, 3, 0, 0),
            (0, 0, BuildingKind.UNKNOWN, 1000.0, 4, 0, 0),
        ]
    )
    tagged, untagged = workplace_masks(table, 1000.0, 3)
    assert not tagged.any()
    assert untagged.tolist() == [True, False, False]


def test_a_tagged_workplace_counts_whatever_its_size() -> None:
    table = building_table([(0, 0, BuildingKind.CIVIC, 10.0, 40, 0, 0)])
    tagged, untagged = workplace_masks(table, 1000.0, 3)
    assert tagged.tolist() == [True]
    assert untagged.tolist() == [False]


def homes_at_the_origin(n: int = 4000) -> tuple[World, np.ndarray]:
    world = uniform_world(n, seed=5, extent_m=1.0)
    return world, np.ones(n, np.bool_)


FAR = 1e9


def test_cells_are_drawn_by_floor_area_not_footprint_area() -> None:
    world, occupied = homes_at_the_origin()
    table = building_table(
        [
            (-1000, 0, BuildingKind.COMMERCIAL, 1000.0, 4, 0, 0),
            (1000, 0, BuildingKind.COMMERCIAL, 2000.0, 1, 0, 1),
        ]
    )
    _, summary = run_work(world, occupied, table, scale_m=FAR)
    assert abs(summary["workers_by_bank"]["west"] / world.n - 2 / 3) < 0.04


def test_buildings_in_one_cell_are_one_workplace_on_the_bank_with_more_floor_area() -> None:
    world, occupied = homes_at_the_origin(500)
    east_heavy = [
        (100, 100, BuildingKind.COMMERCIAL, 1000.0, 1, 0, 0),
        (400, 100, BuildingKind.COMMERCIAL, 1000.0, 2, 0, 1),
    ]
    west_heavy = [
        (100, 100, BuildingKind.COMMERCIAL, 1000.0, 2, 0, 0),
        (400, 100, BuildingKind.COMMERCIAL, 1000.0, 1, 0, 1),
    ]
    _, east = run_work(world, occupied, building_table(east_heavy), cell_m=500.0)
    _, west = run_work(world, occupied, building_table(west_heavy), cell_m=500.0)
    assert east["workers_by_bank"] == {"west": 0, "east": world.n}
    assert west["workers_by_bank"] == {"west": world.n, "east": 0}


def test_a_cell_split_evenly_across_the_river_stays_on_the_west_bank() -> None:
    world, occupied = homes_at_the_origin(500)
    even = [
        (100, 100, BuildingKind.COMMERCIAL, 1000.0, 1, 0, 0),
        (400, 100, BuildingKind.COMMERCIAL, 1000.0, 1, 0, 1),
    ]
    _, summary = run_work(world, occupied, building_table(even), cell_m=500.0)
    assert summary["workers_by_bank"] == {"west": world.n, "east": 0}


def test_the_cell_size_decides_whether_two_buildings_share_coworkers() -> None:
    world, occupied = two_clusters()
    buildings = building_table([commercial(0, 0), commercial(EAST, 0, bank=1)])
    (u, v, _), _ = run_work(world, occupied, buildings, scale_m=100.0, cell_m=500.0)
    (mu, mv, _), _ = run_work(world, occupied, buildings, scale_m=100.0, cell_m=100_000.0)
    across = (world.x[u] > EAST / 2) != (world.x[v] > EAST / 2)
    merged_across = (world.x[mu] > EAST / 2) != (world.x[mv] > EAST / 2)
    assert across.mean() < 0.01
    assert 0.4 < merged_across.mean() < 0.6


def test_a_nearer_workplace_of_equal_floor_area_is_chosen_more_often() -> None:
    world, occupied = homes_at_the_origin()
    table = building_table(
        [
            (500, 0, BuildingKind.COMMERCIAL, 1000.0, 1, 0, 0),
            (3000, 0, BuildingKind.COMMERCIAL, 1000.0, 1, 0, 1),
        ]
    )
    _, near = run_work(world, occupied, table, scale_m=500.0)
    _, flat_decay = run_work(world, occupied, table, scale_m=FAR)
    assert near["workers_by_bank"]["west"] / world.n > 0.95
    assert abs(flat_decay["workers_by_bank"]["west"] / world.n - 0.5) < 0.04


def test_merge_keeps_the_lowest_kind_and_its_strength_for_a_repeated_pair() -> None:
    graph = merge(
        {
            HOUSEHOLD: triple([(0, 1, 1.0)]),
            NEIGHBOUR: triple([(0, 1, 0.3), (1, 2, 0.4)]),
            FRIEND: triple([(0, 1, 0.5), (0, 3, 0.6), (1, 2, 0.7)]),
            WORK: triple([(0, 3, 0.2), (2, 3, 0.3), (0, 1, 0.45)]),
        },
        6,
    )
    table = edges(graph)
    assert table == {
        (0, 1): (HOUSEHOLD, 1.0),
        (1, 0): (HOUSEHOLD, 1.0),
        (1, 2): (NEIGHBOUR, np.float32(0.4)),
        (2, 1): (NEIGHBOUR, np.float32(0.4)),
        (0, 3): (FRIEND, np.float32(0.6)),
        (3, 0): (FRIEND, np.float32(0.6)),
        (2, 3): (WORK, np.float32(0.3)),
        (3, 2): (WORK, np.float32(0.3)),
    }
    assert_csr_invariants(graph, 6)


def test_merge_gives_residents_without_ties_empty_rows() -> None:
    graph = build_graph([(1, 3, FRIEND, 0.5)], 6)
    assert graph.offsets.tolist() == [0, 0, 1, 1, 2, 2, 2]
    assert graph.targets.tolist() == [3, 1]
    assert graph.kind.tolist() == [FRIEND, FRIEND]


def test_merge_of_nothing_is_an_empty_graph_with_n_plus_one_offsets() -> None:
    graph = merge({}, 4)
    assert graph.offsets.tolist() == [0, 0, 0, 0, 0]
    assert len(graph.targets) == len(graph.kind) == len(graph.strength) == 0
    assert graph.targets.dtype == np.uint32 and graph.kind.dtype == np.uint8


def test_merge_sorts_each_row_by_target() -> None:
    graph = build_graph([(0, 4, FRIEND, 0.5), (0, 2, WORK, 0.3), (0, 3, NEIGHBOUR, 0.2)], 5)
    row = graph.targets[graph.offsets[0] : graph.offsets[1]]
    assert row.tolist() == [2, 3, 4]


def pair_sets(n: int) -> st.SearchStrategy[list[tuple[int, int, float]]]:
    pair = st.tuples(st.integers(0, n - 2), st.integers(1, n - 1)).filter(lambda p: p[0] < p[1])
    strength = st.floats(0.01, 1.0)
    return st.lists(st.tuples(pair, strength), max_size=40, unique_by=lambda e: e[0]).map(
        lambda rows: [(a, b, w) for (a, b), w in rows]
    )


@settings(deadline=None, max_examples=60)
@given(st.data(), st.integers(2, 30))
def test_merged_graph_always_satisfies_the_csr_invariants_and_kind_priority(
    data: st.DataObject, n: int
) -> None:
    per_kind = {k: data.draw(pair_sets(n)) for k in range(4)}
    graph = merge({k: triple(p) for k, p in per_kind.items()}, n)
    assert_csr_invariants(graph, n)
    expected: dict[tuple[int, int], tuple[int, float]] = {}
    for k in range(4):
        for a, b, w in per_kind[k]:
            expected.setdefault((a, b), (k, float(np.float32(w))))
    table = edges(graph)
    assert len(table) == 2 * len(expected)
    for (a, b), value in expected.items():
        assert table[(a, b)] == value and table[(b, a)] == value


@settings(deadline=None, max_examples=25)
@given(
    st.integers(8, 60),
    st.integers(0, 2**31 - 1),
    st.integers(1, 4),
    st.floats(0.0, 6.0),
)
def test_generated_ties_always_merge_into_a_valid_csr(
    n: int, seed: int, household_size: int, degree: float
) -> None:
    world = uniform_world(n, seed=seed, extent_m=800.0, household_size=household_size)
    spouse = np.full(n, NONE, np.uint32)
    occupied = np.random.default_rng(seed).random(n) < 0.6
    buildings = building_table([commercial(100, 100), commercial(700, 650, bank=1)])
    parts = {
        HOUSEHOLD: household_ties(world.household, spouse),
        NEIGHBOUR: neighbours(world, seed, degree=degree, candidates=4),
        FRIEND: friends(world, seed, degree=degree, candidates=4),
        WORK: run_work(world, occupied, buildings, seed, degree=degree, candidates=4)[0],
    }
    for t in parts.values():
        assert_pair_shape(t, n)
    graph = merge(parts, n)
    assert_csr_invariants(graph, n)
    members = [
        list(combinations(np.flatnonzero(world.household == h).tolist(), 2))
        for h in set(world.household.tolist())
    ]
    for pair in (p for group in members for p in group):
        assert edges(graph)[pair][0] == HOUSEHOLD
