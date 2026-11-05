from dataclasses import replace

import pytest

from twl_citybake.config import GraphConfig, GraphTargets, load_city


def test_graph_defaults_are_the_documented_values() -> None:
    config = GraphConfig()
    assert (config.neighbour_degree, config.neighbour_scale_m) == (1.6, 250.0)
    assert (config.neighbour_candidates, config.friend_candidates) == (64, 16)
    assert (config.friend_scale_m, config.friend_homophily) == (3000.0, 4.0)
    assert (config.work_scale_m, config.work_cell_m) == (5000.0, 500.0)
    assert (config.workplace_min_area_m2, config.workplace_max_floors) == (1000.0, 3)
    assert tuple(config.targets.mean_degree) == (5.0, 8.0)
    assert tuple(config.targets.clustering) == (0.10, 0.50)
    assert tuple(config.targets.cross_river_non_household) == (0.05, 0.25)
    assert config.targets.largest_component_min == 0.95
    assert config.targets.isolated_max == 0.02


def test_ahmedabad_states_the_approved_graph_targets() -> None:
    targets = load_city("ahmedabad").graph.targets
    assert tuple(targets.mean_degree) == (5.0, 8.0)
    assert targets.largest_component_min == 0.95


INVALID = [
    {"neighbour_scale_m": 0.0},
    {"friend_scale_m": -1.0},
    {"work_scale_m": 0.0},
    {"neighbour_degree": -0.1},
    {"friend_degree": -1.0},
    {"work_degree": -1.0},
    {"neighbour_candidates": 1},
    {"friend_candidates": 1},
    {"workplace_candidates": 1},
    {"neighbour_candidates": 64.0},
    {"friend_candidates": 16.0},
    {"workplace_candidates": 16.0},
    {"workplace_max_floors": 3.0},
    {"workplace_max_floors": 0},
    {"targets": {"mean_degree": (8.0, 5.0)}},
    {"targets": {"clustering": (0.5, 0.1)}},
    {"targets": {"cross_river_non_household": (0.3, 0.1)}},
    {"targets": {"mean_degree": ("a", "b")}},
    {"targets": {"clustering": 0.5}},
    {"targets": {"clustering": (0.1, 0.2, 0.3)}},
]


@pytest.mark.parametrize("change", INVALID, ids=lambda c: next(iter(c)) + str(c))
def test_graph_config_rejects_invalid_values(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        targets = change.get("targets")
        if isinstance(targets, dict):
            change = {"targets": GraphTargets(**targets)}
        replace(GraphConfig(), **change)


def test_graph_config_is_frozen() -> None:
    with pytest.raises(AttributeError):
        GraphConfig().friend_degree = 9.0  # type: ignore[misc]
