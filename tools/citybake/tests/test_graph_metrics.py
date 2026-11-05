from typing import Any

import pytest
from graph_fixture import FRIEND, HOUSEHOLD, NEIGHBOUR, WORK, build_graph, graph_report

Tie = tuple[int, int, int, float]


def report_for(
    ties: list[Tie],
    n: int,
    bank: list[int] | None = None,
    segment: list[int] | None = None,
    x: list[float] | None = None,
) -> dict[str, Any]:
    return graph_report(
        build_graph(ties, n),
        bank or [0] * n,
        segment or [0] * n,
        x or list(range(n)),
        [0.0] * n,
    )


TRIANGLE: list[Tie] = [(0, 1, FRIEND, 0.5), (0, 2, FRIEND, 0.5), (1, 2, FRIEND, 0.5)]
STAR: list[Tie] = [(0, k, FRIEND, 0.5) for k in range(1, 5)]


def clustering(report: dict[str, Any], household_removed: bool = False) -> dict[str, float]:
    section = report["clustering"]
    return section["without_household"] if household_removed else section


def local(report: dict[str, Any], household_removed: bool = False) -> float:
    return clustering(report, household_removed)["average_local"]


def transitivity(report: dict[str, Any], household_removed: bool = False) -> float:
    return clustering(report, household_removed)["transitivity"]


def test_a_triangle_has_clustering_one() -> None:
    report = report_for(TRIANGLE, 3)
    assert local(report) == pytest.approx(1.0)
    assert transitivity(report) == pytest.approx(1.0)


def test_a_star_has_clustering_zero() -> None:
    report = report_for(STAR, 5)
    assert local(report) == pytest.approx(0.0)
    assert transitivity(report) == pytest.approx(0.0)


def test_clustering_averages_over_residents_with_two_or_more_ties() -> None:
    ties: list[Tie] = [
        (0, 1, HOUSEHOLD, 1.0),
        (0, 2, HOUSEHOLD, 1.0),
        (1, 2, HOUSEHOLD, 1.0),
        (2, 3, FRIEND, 0.5),
        (3, 4, FRIEND, 0.5),
    ]
    report = report_for(ties, 5)
    assert local(report) == pytest.approx((1 + 1 + 1 / 3 + 0) / 4, abs=1e-3)
    assert transitivity(report) == pytest.approx(0.5, abs=1e-3)


def test_clustering_without_household_ties_drops_the_household_clique() -> None:
    ties: list[Tie] = [
        (0, 1, HOUSEHOLD, 1.0),
        (0, 2, HOUSEHOLD, 1.0),
        (1, 2, HOUSEHOLD, 1.0),
        (2, 3, FRIEND, 0.5),
        (3, 4, FRIEND, 0.5),
    ]
    report = report_for(ties, 5)
    assert local(report, household_removed=True) == pytest.approx(0.0)
    assert transitivity(report, household_removed=True) == pytest.approx(0.0)


MIXED: list[Tie] = [
    (0, 1, HOUSEHOLD, 1.0),
    (2, 3, HOUSEHOLD, 1.0),
    (0, 3, FRIEND, 0.5),
    (1, 2, FRIEND, 0.5),
    (0, 2, FRIEND, 0.5),
    (0, 5, WORK, 0.3),
    (4, 5, NEIGHBOUR, 0.2),
]
BANK = [0, 0, 0, 1, 1, 1]


def test_cross_river_share_counts_undirected_ties_once() -> None:
    cross = report_for(MIXED, 6, bank=BANK)["cross_river"]
    assert cross["all"] == pytest.approx(3 / 7, abs=1e-3)
    assert cross["non_household"] == pytest.approx(0.4, abs=1e-3)


def test_cross_river_share_per_kind() -> None:
    cross = report_for(MIXED, 6, bank=BANK)["cross_river"]
    expected = {"household": 0.5, "neighbour": 0.0, "friend": 1 / 3, "work": 1.0}
    for kind, share in expected.items():
        assert cross["by_kind"][kind] == pytest.approx(share, abs=1e-3), kind


def test_homophily_compares_friend_same_segment_share_with_random_mixing() -> None:
    ties: list[Tie] = [
        (0, 1, FRIEND, 0.5),
        (0, 3, FRIEND, 0.5),
        (3, 4, FRIEND, 0.5),
        (2, 5, FRIEND, 0.5),
        (0, 2, HOUSEHOLD, 1.0),
        (1, 2, WORK, 0.3),
    ]
    homophily = report_for(ties, 6, segment=[0, 0, 0, 1, 1, 2])["homophily"]
    expected = (1 / 2) ** 2 + (1 / 3) ** 2 + (1 / 6) ** 2
    assert homophily["expected_random_mixing"] == pytest.approx(expected, abs=1e-3)
    # the two same-segment ties that are not friends would make this 4 / 6 if they were counted
    assert homophily["friend_same_segment_share"] == pytest.approx(0.5, abs=1e-3)


def test_random_mixing_expectation_follows_the_segment_shares() -> None:
    homophily = report_for(TRIANGLE, 10, segment=[0] * 9 + [1])["homophily"]
    assert homophily["expected_random_mixing"] == pytest.approx(0.9**2 + 0.1**2, abs=1e-3)


RING = [(i, (i + 1) % 10, FRIEND, 0.5) for i in range(9)] + [(0, 9, FRIEND, 0.5)]


def test_degree_percentiles_of_a_ring_are_all_two() -> None:
    degree = report_for(RING, 10)["degree"]
    for key in ("mean", "p10", "p50", "p90", "p99", "max"):
        assert degree[key] == pytest.approx(2.0), key
    assert degree["isolated_share"] == 0.0


def test_degree_mean_max_and_isolated_share_of_a_star_with_an_isolate() -> None:
    degree = report_for(STAR, 6)["degree"]
    assert degree["mean"] == pytest.approx(8 / 6, abs=1e-3)
    assert degree["max"] == 4
    assert degree["isolated_share"] == pytest.approx(1 / 6, abs=1e-3)
    assert degree["p10"] <= degree["p50"] <= degree["p90"] <= degree["p99"] <= degree["max"]


def test_mean_degree_by_kind_averages_each_kind_over_all_residents() -> None:
    ties: list[Tie] = [*STAR, (5, 6, HOUSEHOLD, 1.0), (4, 5, WORK, 0.3)]
    by_kind = report_for(ties, 8)["degree"]["mean_by_kind"]
    assert by_kind["friend"] == pytest.approx(8 / 8)
    assert by_kind["household"] == pytest.approx(2 / 8)
    assert by_kind["work"] == pytest.approx(2 / 8)
    assert by_kind["neighbour"] == pytest.approx(0.0)


def test_tie_distance_percentiles_are_reported_per_kind() -> None:
    ties: list[Tie] = [
        (0, 1, NEIGHBOUR, 0.3),
        (2, 3, NEIGHBOUR, 0.3),
        (0, 2, FRIEND, 0.5),
        (1, 3, FRIEND, 0.5),
    ]
    distance = report_for(ties, 4, x=[0.0, 100.0, 1000.0, 1100.0])["distance_m"]
    assert distance["neighbour"] == pytest.approx({"p50": 100.0, "p90": 100.0})
    assert distance["friend"] == pytest.approx({"p50": 1000.0, "p90": 1000.0})
    assert distance["work"] == {"p50": 0.0, "p90": 0.0}


def test_components_count_and_largest_share() -> None:
    ties: list[Tie] = [
        (0, 1, FRIEND, 0.5),
        (1, 2, FRIEND, 0.5),
        (0, 2, FRIEND, 0.5),
        (3, 4, FRIEND, 0.5),
        (4, 5, FRIEND, 0.5),
    ]
    components = report_for(ties, 7)["components"]
    assert components == {"count": 3, "largest_share": pytest.approx(3 / 7, abs=1e-3)}


def test_a_connected_graph_is_one_component_holding_everyone() -> None:
    components = report_for(RING, 10)["components"]
    assert components == {"count": 1, "largest_share": pytest.approx(1.0)}


def test_strength_distribution_per_kind() -> None:
    ties: list[Tie] = [
        (0, 1, FRIEND, 0.5),
        (1, 2, FRIEND, 0.5),
        (2, 3, WORK, 0.2),
        (3, 4, WORK, 0.4),
    ]
    strength = report_for(ties, 5)["strength"]
    for key in ("p10", "p50", "p90"):
        assert strength["friend"][key] == pytest.approx(0.5, abs=1e-3)
    assert strength["work"]["mean"] == pytest.approx(0.3, abs=1e-3)
    assert strength["work"]["p50"] == pytest.approx(0.3, abs=1e-3)


def test_per_resident_mean_strength_excludes_residents_without_ties() -> None:
    ties: list[Tie] = [(0, 1, HOUSEHOLD, 1.0), (1, 2, FRIEND, 0.5)]
    per_resident = report_for(ties, 4)["strength"]["per_resident_mean"]
    assert per_resident["mean"] == pytest.approx((1.0 + 0.75 + 0.5) / 3, abs=1e-3)
    assert per_resident["p50"] == pytest.approx(0.75, abs=1e-3)
    assert 0.5 - 1e-3 <= per_resident["p10"] <= per_resident["p50"] <= per_resident["p90"] <= 1.0


def test_report_has_the_documented_sections() -> None:
    report = report_for(MIXED, 6, bank=BANK)
    assert {"degree", "clustering", "cross_river", "homophily", "distance_m"} <= set(report)
    assert {"components", "strength"} <= set(report)
    assert {"mean", "p10", "p50", "p90", "p99", "max", "isolated_share", "mean_by_kind"} <= set(
        report["degree"]
    )
    assert {"p10", "p50", "p90", "mean"} == set(report["strength"]["per_resident_mean"])
