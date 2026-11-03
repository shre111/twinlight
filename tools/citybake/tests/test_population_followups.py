import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from osm_fixture import BBOX
from population_fixture import CENSUS_COUNTS, CENSUS_SIZES, write_world
from test_population_stage import tiny_config

from twl_citybake.config import BBox, HouseholdConfig, load_city
from twl_citybake.population.stage import edge_people, run_population, write_population_layers
from twl_citybake.population.worldpop import PopGrid

SIZES = (1, 2, 3, 4, 5, 6, 8, 12, 17)
COUNTS = (37094, 97013, 176374, 310595, 229831, 143502, 158773, 18359, 4216)


def household(shrink: float) -> HouseholdConfig:
    return HouseholdConfig(source="t", sizes=SIZES, counts=COUNTS, size_shrink=shrink)


def mean_of(sizes: tuple[int, ...], counts: tuple[float, ...]) -> float:
    return sum(s * n for s, n in zip(sizes, counts, strict=True)) / sum(counts)


@pytest.mark.parametrize("shrink", [0.5, 0.8, 0.9, 0.99])
def test_shrunk_counts_hit_the_scaled_census_mean(shrink: float) -> None:
    cfg = household(shrink)
    got = mean_of(SIZES, cfg.shrunk_counts())
    assert got == pytest.approx(cfg.mean_size * shrink, rel=1e-9)


def test_shrink_of_one_returns_the_census_counts_as_floats() -> None:
    counts = household(1.0).shrunk_counts()
    assert counts == tuple(float(n) for n in COUNTS)
    assert all(isinstance(n, float) for n in counts)


def test_shrinking_moves_share_to_small_households_and_away_from_the_largest() -> None:
    base = np.asarray(household(1.0).shrunk_counts())
    shrunk = np.asarray(household(0.8).shrunk_counts())
    base, shrunk = base / base.sum(), shrunk / shrunk.sum()
    assert shrunk[0] > base[0]
    assert shrunk[-1] < base[-1]


def test_shrunk_counts_stay_positive_for_every_size() -> None:
    assert all(n > 0 for n in household(0.6).shrunk_counts())


@pytest.mark.parametrize("shrink", [0.0, -0.5, 1.01, 2.0])
def test_size_shrink_outside_zero_one_is_rejected(shrink: float) -> None:
    with pytest.raises(ValueError, match="size_shrink"):
        household(shrink)


def test_size_shrink_that_pushes_the_mean_below_the_smallest_size_is_rejected() -> None:
    mean = household(1.0).mean_size
    with pytest.raises(ValueError, match="size_shrink"):
        household(0.99 / mean)
    household(1.0 / mean)


def test_ahmedabad_config_carries_the_population_provenance() -> None:
    city = load_city("ahmedabad")
    assert city.seed == 20260101
    assert city.worldpop is not None
    assert city.worldpop.md5 == "89905eed942228829cf21eee28eb12df"
    assert city.population.adult_share_source
    assert city.households is not None
    assert city.households.size_shrink == 1.0


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("world")
    write_world(BBOX, root / "pop.tif", root / "layers", block=18)
    return root / "layers", root / "pop.tif"


def test_stage_with_shrink_synthesises_smaller_households(world: tuple[Path, Path]) -> None:
    layers, tif = world
    base = tiny_config()
    shrunk = replace(
        base,
        households=HouseholdConfig(
            source="t", sizes=CENSUS_SIZES, counts=CENSUS_COUNTS, size_shrink=0.8
        ),
    )
    a = run_population(layers, tif, base)
    b = run_population(layers, tif, shrunk)
    mean_a = a.report["households"]["people"] / a.report["households"]["count"]
    mean_b = b.report["households"]["people"] / b.report["households"]["count"]
    assert mean_b < mean_a
    assert b.report["districts"]["outside_1pct"] == []
    assert b.report["districts"]["max_abs_people_vs_placed"] == pytest.approx(0.0, abs=1e-6)
    assert b.report["capacity"]["buildings_over_capacity"] == 0
    assert b.report["population"]["household_size_shrink"] == 0.8


def grid2x2() -> PopGrid:
    return PopGrid(
        values=np.array([[100.0, 200.0], [300.0, 400.0]]), west=0.0, north=2.0, dlon=1.0, dlat=1.0
    )


def test_edge_people_is_zero_for_cells_fully_inside() -> None:
    bbox = BBox(0.0, 0.0, 2.0, 2.0)
    assert edge_people(grid2x2(), bbox, np.array([0.5, 1.5]), np.array([1.5, 0.5])) == 0.0


def test_edge_people_adds_half_a_cell_when_half_outside() -> None:
    bbox = BBox(0.5, 0.0, 2.0, 2.0)
    assert edge_people(grid2x2(), bbox, np.array([0.7]), np.array([1.5])) == pytest.approx(50.0)


def test_edge_people_counts_each_housed_cell_once_with_corner_overlap() -> None:
    bbox = BBox(0.5, 0.5, 2.0, 2.0)
    lon, lat = np.array([0.6, 0.7, 0.6]), np.array([0.6, 0.7, 1.5])
    expected = 300.0 * 0.75 + 100.0 * 0.5
    assert edge_people(grid2x2(), bbox, lon, lat) == pytest.approx(expected)


def test_edge_people_ignores_unhoused_cells() -> None:
    bbox = BBox(0.5, 0.0, 2.0, 2.0)
    assert edge_people(grid2x2(), bbox, np.array([1.5]), np.array([1.5])) == 0.0
    assert edge_people(grid2x2(), bbox, np.array([]), np.array([])) == 0.0


def test_report_and_layer_json_carry_the_followup_fields(
    tmp_path: Path, world: tuple[Path, Path]
) -> None:
    layers, tif = world
    out = tmp_path / "population"
    write_population_layers(run_population(layers, tif, tiny_config()), out)
    report = json.loads((out / "report.json").read_text())
    meta = json.loads((out / "layer.json").read_text())

    assert report["worldpop"]["edge_people_beyond_bbox"] >= 0
    assert report["population"] == {
        "adult_share": 0.69,
        "adult_share_source": "",
        "shopper_share": 0.27,
        "household_source": "test fixture",
        "household_size_shrink": 1.0,
        "census_mean_household_size": pytest.approx(mean_of(CENSUS_SIZES, CENSUS_COUNTS), abs=1e-3),
    }
    small = report["districts"]["outside_1pct_under_1000"]
    assert isinstance(small, list) and all(isinstance(n, str) for n in small)
    assert report["districts"]["max_abs_people_vs_placed"] == pytest.approx(0.0, abs=1e-6)

    assert meta["shopper_share"] == 0.27
    assert meta["household_size_shrink"] == 1.0
    assert meta["mean_weight"] == report["residents"]["weight"]
    assert meta["mean_weight"] > 0


@pytest.mark.parametrize(
    "districts",
    [
        [{"id": 1, "name": "West"}, {"id": 0, "name": "East"}],
        [{"id": 0, "name": "West"}, {"id": 2, "name": "East"}],
    ],
)
def test_districts_with_ids_not_running_zero_to_n_in_order_are_rejected(
    tmp_path: Path, world: tuple[Path, Path], districts: list[dict]
) -> None:
    layers, tif = world
    copy = tmp_path / "layers"
    copy.mkdir()
    for p in layers.iterdir():
        if p.name != "districts.json":
            (copy / p.name).symlink_to(p)
    (copy / "districts.json").write_text(json.dumps(districts))
    with pytest.raises(ValueError, match=r"0\.\.n-1"):
        run_population(copy, tif, tiny_config())


def test_shrinking_below_the_smallest_counted_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="size_shrink"):
        HouseholdConfig(source="t", sizes=(1, 2, 3), counts=(0, 5, 5), size_shrink=0.5)


def test_report_keeps_census_shares_beside_the_shrunk_target_shares(tmp_path: Path) -> None:
    layers, tif = tmp_path / "layers", tmp_path / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    city = tiny_config()
    assert city.households is not None
    shrunk = replace(city, households=replace(city.households, size_shrink=0.8))
    households = run_population(layers, tif, shrunk).report["households"]
    assert households["census_size_shares"] != households["target_size_shares"]
    assert sum(households["census_size_shares"].values()) == pytest.approx(1.0, abs=5e-3)
