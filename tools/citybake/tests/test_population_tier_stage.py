import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from osm_fixture import BBOX
from population_fixture import write_world
from test_population_stage import tiny_config

from twl_citybake.config import CityConfig, TierConfig
from twl_citybake.population.capacity import storeys
from twl_citybake.population.stage import (
    CENSUS_ATTRIBUTION,
    PopulationLayers,
    run_population,
    write_population_layers,
)
from twl_citybake.population.tier import building_tiers
from twl_citybake.store import read_table, write_table

TIER = TierConfig(
    low_share=0.38,
    low_share_source="low source",
    high_share=0.2,
    high_share_source="high source",
    attribution="Tier credit",
)


def with_tier(tier: TierConfig | None = TIER) -> CityConfig:
    return replace(tiny_config(), tier=tier)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("tier-world")
    layers, tif = root / "layers", root / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    buildings = read_table(layers / "buildings")
    n = len(buildings["x"])
    buildings["area_m2"] = (120.0 + 55.0 * (np.arange(n) * 7 % 11)).astype(np.float32)
    buildings["bank"] = (buildings["x"] > 0).astype(np.uint8)
    write_table(layers / "buildings", buildings)
    return layers, tif


@pytest.fixture(scope="module")
def plain(world: tuple[Path, Path]) -> PopulationLayers:
    return run_population(*world, with_tier(None))


@pytest.fixture(scope="module")
def tiered(world: tuple[Path, Path]) -> PopulationLayers:
    return run_population(*world, with_tier())


def building_tier(layers: PopulationLayers, world: tuple[Path, Path]) -> np.ndarray:
    buildings = read_table(world[0] / "buildings")
    persons = layers.building_population["persons"]
    per_building = np.bincount(layers.households["building"], minlength=len(persons))
    return building_tiers(
        buildings["area_m2"],
        storeys(buildings["height_m"]),
        persons,
        per_building,
        TIER.low_share,
        TIER.high_share,
    ).tier


def test_households_and_residents_carry_a_uint8_tier_of_their_building(
    tiered: PopulationLayers, world: tuple[Path, Path]
) -> None:
    expected = building_tier(tiered, world)
    hh = tiered.households
    assert hh["tier"].dtype == np.uint8
    assert hh["tier"].tolist() == expected[hh["building"]].tolist()
    residents = tiered.residents
    assert residents["tier"].dtype == np.uint8
    assert residents["tier"].tolist() == expected[residents["building"]].tolist()


def test_all_three_tiers_occur_among_households(tiered: PopulationLayers) -> None:
    assert set(np.unique(tiered.households["tier"]).tolist()) == {0, 1, 2}


def test_residents_of_one_household_share_its_tier(tiered: PopulationLayers) -> None:
    by_household = tiered.households["tier"][tiered.residents["household"]]
    assert by_household.tolist() == tiered.residents["tier"].tolist()


def test_tier_changes_no_existing_column(plain: PopulationLayers, tiered: PopulationLayers) -> None:
    for name in ("households", "residents", "building_population"):
        a, b = getattr(plain, name), getattr(tiered, name)
        assert set(a) == set(b)
        for column in a:
            if column != "tier":
                assert np.array_equal(a[column], b[column]), (name, column)


def test_tier_changes_no_other_report_entry(
    plain: PopulationLayers, tiered: PopulationLayers
) -> None:
    assert {k: v for k, v in tiered.report.items() if k != "tier"} == {
        k: v for k, v in plain.report.items() if k != "tier"
    }


def test_without_a_tier_every_household_and_resident_is_middle(plain: PopulationLayers) -> None:
    for table in (plain.households, plain.residents):
        assert table["tier"].dtype == np.uint8
        assert (table["tier"] == 1).all()
    assert plain.report["tier"] is None


def test_without_a_tier_layer_json_has_no_tier_credit_or_source(plain: PopulationLayers) -> None:
    assert plain.meta["source"]["tier"] is None
    assert plain.meta["attribution"][-1] == CENSUS_ATTRIBUTION


def test_report_names_the_asked_shares_sources_and_cuts(tiered: PopulationLayers) -> None:
    section = tiered.report["tier"]
    assert section["low_share"] == 0.38
    assert section["high_share"] == 0.2
    assert section["low_share_source"] == "low source"
    assert section["high_share_source"] == "high source"
    assert 0 < section["low_cut_m2_per_person"] <= section["high_cut_m2_per_person"]
    assert set(section["banks"]) == {"west", "east"}


def test_report_household_shares_are_realised_shares_of_the_tables(
    tiered: PopulationLayers,
) -> None:
    section = tiered.report["tier"]
    tier = tiered.households["tier"]
    expected = [float(np.mean(tier == t)) for t in (0, 1, 2)]
    assert section["households"] == pytest.approx(expected, abs=1e-3)
    assert section["households"][0] <= 0.38 + 1e-3
    assert section["households"][2] >= 0.2 - 1e-3


def test_report_people_shares_weigh_households_by_size(tiered: PopulationLayers) -> None:
    size = tiered.households["size"].astype(np.float64)
    tier = tiered.households["tier"]
    expected = [size[tier == t].sum() / size.sum() for t in (0, 1, 2)]
    assert tiered.report["tier"]["people"] == pytest.approx(expected, abs=1e-3)


def test_report_bank_shares_are_household_shares_within_each_bank(
    tiered: PopulationLayers, world: tuple[Path, Path]
) -> None:
    bank = read_table(world[0] / "buildings")["bank"][tiered.households["building"]]
    tier = tiered.households["tier"]
    for name, value in (("west", 0), ("east", 1)):
        in_bank = tier[bank == value]
        expected = [float(np.mean(in_bank == t)) for t in (0, 1, 2)]
        assert tiered.report["tier"]["banks"][name] == pytest.approx(expected, abs=1e-3)


def test_a_bank_without_households_reports_zeros(tmp_path: Path) -> None:
    layers, tif = tmp_path / "layers", tmp_path / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    buildings = read_table(layers / "buildings")
    buildings["area_m2"] = (120.0 + 55.0 * (np.arange(len(buildings["x"])) % 5)).astype(np.float32)
    write_table(layers / "buildings", buildings)
    section = run_population(layers, tif, with_tier()).report["tier"]
    assert section["banks"]["east"] == [0.0, 0.0, 0.0]
    assert sum(section["banks"]["west"]) == pytest.approx(1.0, abs=2e-3)


def test_a_floors_column_equal_to_the_height_storeys_changes_nothing(
    world: tuple[Path, Path], tiered: PopulationLayers, tmp_path: Path
) -> None:
    layers = tmp_path / "layers"
    buildings = read_table(world[0] / "buildings")
    buildings["floors"] = storeys(buildings["height_m"]).astype(np.uint8)
    write_table(layers / "buildings", buildings)
    (layers / "districts.json").write_text((world[0] / "districts.json").read_text())
    other = run_population(layers, world[1], with_tier())
    assert other.households["tier"].tolist() == tiered.households["tier"].tolist()


def test_layer_json_credits_the_tier_after_the_census_and_records_its_source(
    tiered: PopulationLayers, tmp_path: Path
) -> None:
    out = tmp_path / "population"
    write_population_layers(tiered, out)
    meta = json.loads((out / "layer.json").read_text())
    attribution = meta["attribution"]
    assert attribution.index("Tier credit") > attribution.index(CENSUS_ATTRIBUTION)
    assert meta["source"]["tier"] == asdict(TIER)


def test_written_tables_keep_the_tier_column_as_uint8(
    tiered: PopulationLayers, tmp_path: Path
) -> None:
    out = tmp_path / "population"
    write_population_layers(tiered, out)
    for name in ("households", "residents"):
        assert read_table(out / name)["tier"].dtype == np.uint8
    report = json.loads((out / "report.json").read_text())
    assert report["tier"]["households"] == tiered.report["tier"]["households"]
