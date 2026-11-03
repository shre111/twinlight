import dataclasses
from pathlib import Path

import numpy as np
import pytest
from osm_fixture import OsmBuilder, TinyCity, ring_area

from twl_citybake.config import CityConfig, DistrictConfig, RiverConfig
from twl_citybake.osm import heights
from twl_citybake.osm.extract import OsmLayers, extract_osm
from twl_citybake.osm.tags import HeightSource

SEEDS = {"Alpha": (-700.0, 700.0), "Beta": (700.0, -700.0), "Gamma": (-300.0, -800.0)}
DISTRICT_IDS = {"Alpha": 0, "Beta": 1, "Gamma": 2}


def building_row(layers: OsmLayers, osm_id: int, osm_type: int = 0) -> int:
    b = layers.buildings
    hits = np.flatnonzero((b["osm_type"] == osm_type) & (b["osm_id"] == osm_id))
    assert len(hits) == 1
    return int(hits[0])


def height_of(layers: OsmLayers, osm_id: int, osm_type: int = 0) -> float:
    return float(layers.buildings["height_m"][building_row(layers, osm_id, osm_type)])


def inside(point: tuple[float, float], rings: list[list[list[float]]]) -> bool:
    x, y = point
    hits = 0
    for ring in rings:
        pts = np.asarray(ring, np.float64)
        nxt = np.roll(pts, -1, axis=0)
        crosses = (pts[:, 1] > y) != (nxt[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = pts[:, 0] + (y - pts[:, 1]) * (nxt[:, 0] - pts[:, 0]) / (nxt[:, 1] - pts[:, 1])
        hits += int(np.sum(crosses & (x < at)))
    return hits % 2 == 1


def test_min_height_samples_is_twenty() -> None:
    assert heights.MIN_HEIGHT_SAMPLES == 20


def test_river_splits_the_clip_into_west_and_east_banks(layers: OsmLayers) -> None:
    b = layers.buildings

    assert layers.report["banks_split"] is True
    assert np.array_equal(b["bank"], (b["x"] > 0).astype(np.uint8))
    assert set(b["bank"].tolist()) == {0, 1}


def test_rivers_with_other_names_do_not_split_banks(layers: OsmLayers) -> None:
    b = layers.buildings
    i = building_row(layers, 1002)

    assert b["x"][i] == pytest.approx(-500, abs=0.1)
    assert b["bank"][i] == 0


def test_west_bank_share_is_reported(layers: OsmLayers) -> None:
    west = int(np.sum(layers.buildings["bank"] == 0))
    assert west == 30
    assert layers.report["bank_share_west"] == pytest.approx(30 / 61, abs=0.001)


def test_districts_come_from_place_nodes_with_duplicate_names_merged(layers: OsmLayers) -> None:
    assert [d["name"] for d in layers.districts] == ["Alpha", "Beta", "Gamma"]
    assert [d["id"] for d in layers.districts] == [0, 1, 2]


def test_unlisted_place_kinds_unnamed_and_outside_nodes_are_not_seeds(layers: OsmLayers) -> None:
    names = {d["name"] for d in layers.districts}
    assert names.isdisjoint({"Hamlet", "Outside", "gamma "})
    assert len(layers.districts) == 3


def test_duplicate_name_keeps_the_lowest_id_seed(layers: OsmLayers) -> None:
    gamma = layers.districts[2]

    assert tuple(gamma["seed"]) == pytest.approx(SEEDS["Gamma"], abs=0.1)
    assert gamma["bank"] == 0


def test_district_seeds_and_banks(layers: OsmLayers) -> None:
    for entry in layers.districts:
        assert tuple(entry["seed"]) == pytest.approx(SEEDS[entry["name"]], abs=0.1)
    assert [d["bank"] for d in layers.districts] == [0, 1, 0]


def test_districts_are_numbered_by_normalised_name(layers: OsmLayers) -> None:
    b = layers.buildings
    for osm_id, name in [(1003, "Alpha"), (1010, "Beta"), (1011, "Gamma"), (21_000, "Beta")]:
        assert b["district"][building_row(layers, osm_id)] == DISTRICT_IDS[name]


def test_each_building_belongs_to_the_nearest_seed_on_its_own_bank(layers: OsmLayers) -> None:
    b = layers.buildings
    seeds = layers.districts
    seed_xy = np.array([d["seed"] for d in seeds], np.float64)
    seed_bank = np.array([d["bank"] for d in seeds])
    for i in range(len(b["osm_id"])):
        own = np.flatnonzero(seed_bank == b["bank"][i])
        distance = np.hypot(seed_xy[own, 0] - b["x"][i], seed_xy[own, 1] - b["y"][i])
        assert b["district"][i] == own[int(np.argmin(distance))]


def test_building_across_the_river_ignores_a_closer_seed_on_the_other_bank(
    layers: OsmLayers,
) -> None:
    b = layers.buildings
    i = building_row(layers, 1010)
    gamma, beta = np.array(SEEDS["Gamma"]), np.array(SEEDS["Beta"])
    here = np.array([b["x"][i], b["y"][i]])

    assert np.linalg.norm(here - gamma) < np.linalg.norm(here - beta)
    assert b["district"][i] == DISTRICT_IDS["Beta"]


def test_every_building_has_a_valid_district_on_its_bank(layers: OsmLayers) -> None:
    b = layers.buildings
    banks = np.array([d["bank"] for d in layers.districts])

    assert np.all(b["district"] < len(layers.districts))
    assert np.array_equal(banks[b["district"]], b["bank"])


def test_district_building_counts_match_assignments(layers: OsmLayers) -> None:
    b = layers.buildings
    counts = [int(np.sum(b["district"] == i)) for i in range(len(layers.districts))]

    assert [d["buildings"] for d in layers.districts] == counts == [29, 31, 1]
    assert layers.report["districts"] == {
        "count": 3,
        "buildings_min": 1,
        "buildings_median": 29,
        "buildings_max": 31,
    }


def test_district_polygons_contain_their_seed_and_stay_on_their_bank(layers: OsmLayers) -> None:
    for entry in layers.districts:
        assert entry["polygon"]
        assert inside(tuple(entry["seed"]), entry["polygon"])
        xs = np.concatenate([np.asarray(ring, np.float64)[:, 0] for ring in entry["polygon"]])
        if entry["bank"] == 0:
            assert xs.max() <= 1.0
        else:
            assert xs.min() >= -1.0


def test_district_polygon_coordinates_are_rounded_to_a_tenth_of_a_metre(layers: OsmLayers) -> None:
    for entry in layers.districts:
        for ring in entry["polygon"]:
            scaled = np.asarray(ring, np.float64) * 10
            assert np.allclose(scaled, np.round(scaled), atol=1e-6)


def test_district_polygons_tile_their_bank(layers: OsmLayers, tiny_city: TinyCity) -> None:
    west = [d for d in layers.districts if d["bank"] == 0]
    area = sum(ring_area(d["polygon"][0]) for d in west)
    xmin, ymin, _, ymax = tiny_city.extent

    assert area == pytest.approx(-xmin * (ymax - ymin), rel=0.05)


def test_default_height_uses_district_and_kind_group_when_large_enough(layers: OsmLayers) -> None:
    assert height_of(layers, 1004) == pytest.approx(9.6, abs=1e-3)
    assert height_of(layers, 1008) == pytest.approx(19.2, abs=1e-3)


def test_default_height_falls_back_to_city_wide_kind_median(layers: OsmLayers) -> None:
    assert height_of(layers, 1011) == pytest.approx(14.4, abs=1e-3)


def test_default_height_falls_back_to_median_over_all_level_buildings(layers: OsmLayers) -> None:
    for osm_id in (1003, 1005, 1010):
        assert height_of(layers, osm_id) == pytest.approx(12.8, abs=1e-3)
    assert height_of(layers, 900, osm_type=1) == pytest.approx(12.8, abs=1e-3)


def test_default_height_changes_when_the_group_threshold_is_raised(
    monkeypatch: pytest.MonkeyPatch, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    monkeypatch.setattr(heights, "MIN_HEIGHT_SAMPLES", 26)

    raised = extract_osm(tiny_city.path, city_config)

    assert height_of(raised, 1004) == pytest.approx(14.4, abs=1e-3)


def test_default_height_is_two_levels_when_no_level_data_exists(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    builder = OsmBuilder(tiny_city.projection)
    builder.rect(1, -300, 200, 20, 20, {"building": "yes"})
    builder.rect(2, 300, -200, 20, 20, {"building": "apartments", "height": "unknow"})
    builder.rect(3, 100, 100, 20, 20, {"building": "retail"})
    builder.write(tmp_path / "bare.osm")

    bare = extract_osm(tmp_path / "bare.osm", city_config)

    assert np.allclose(bare.buildings["height_m"], 6.4, atol=1e-3)
    assert np.all(bare.buildings["height_source"] == HeightSource.DEFAULT)


def test_without_a_matching_river_every_bank_is_zero(
    tiny_city: TinyCity, city_config: CityConfig
) -> None:
    config = dataclasses.replace(city_config, river=RiverConfig(names=("Nowhere",)))

    flat = extract_osm(tiny_city.path, config)

    assert flat.report["banks_split"] is False
    assert flat.report["bank_share_west"] == 1.0
    assert np.all(flat.buildings["bank"] == 0)


def test_without_a_matching_river_each_building_takes_the_nearest_seed_overall(
    tiny_city: TinyCity, city_config: CityConfig
) -> None:
    config = dataclasses.replace(city_config, river=RiverConfig(names=("Nowhere",)))

    flat = extract_osm(tiny_city.path, config)

    seed_xy = np.array([d["seed"] for d in flat.districts], np.float64)
    b = flat.buildings
    for i in range(len(b["osm_id"])):
        distance = np.hypot(seed_xy[:, 0] - b["x"][i], seed_xy[:, 1] - b["y"][i])
        assert b["district"][i] == int(np.argmin(distance))


def test_without_seeds_a_single_city_district_holds_everything(
    tiny_city: TinyCity, city_config: CityConfig
) -> None:
    config = dataclasses.replace(city_config, districts=DistrictConfig(place_kinds=("none",)))

    single = extract_osm(tiny_city.path, config)

    assert [d["name"] for d in single.districts] == ["City"]
    assert single.districts[0]["id"] == 0
    assert single.districts[0]["buildings"] == 61
    assert np.all(single.buildings["district"] == 0)
    assert single.report["districts"]["count"] == 1


def test_bank_without_seeds_falls_back_to_the_nearest_seed_overall(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    builder = OsmBuilder(tiny_city.projection)
    builder.rect(1, 400, 400, 20, 20, {"building": "yes"})
    builder.rect(2, -400, 400, 20, 20, {"building": "yes"})
    builder.node(-100, 0, {"place": "suburb", "name": "Only"})
    builder.node(0, -1200, node_id=6001)
    builder.node(0, 1200, node_id=6002)
    builder.way(300, [6001, 6002], {"waterway": "river", "name": "Sabarmati"})
    builder.write(tmp_path / "lopsided.osm")

    lopsided = extract_osm(tmp_path / "lopsided.osm", city_config)

    assert lopsided.report["banks_split"] is True
    assert [d["name"] for d in lopsided.districts] == ["Only"]
    assert lopsided.buildings["district"].tolist() == [0, 0]


@pytest.mark.parametrize("ids", [(7001, 7002), (7002, 7001)])
def test_place_nodes_at_the_same_position_yield_one_district(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig, ids: tuple[int, int]
) -> None:
    builder = OsmBuilder(tiny_city.projection)
    builder.rect(1, -300, 200, 20, 20, {"building": "yes"})
    builder.rect(2, 300, -200, 20, 20, {"building": "yes"})
    builder.node(-100, 100, {"place": "suburb", "name": "Foo"}, node_id=ids[0])
    builder.node(-100, 100, {"place": "suburb", "name": "Bar"}, node_id=ids[1])
    builder.write(tmp_path / "twins.osm")

    twins = extract_osm(tmp_path / "twins.osm", city_config)

    assert len(twins.districts) == 1
    assert twins.districts[0]["buildings"] == 2
    assert twins.districts[0]["name"] == ("Foo" if ids[0] < ids[1] else "Bar")


def test_coincident_seeds_do_not_remove_distinct_neighbours(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    builder = OsmBuilder(tiny_city.projection)
    builder.rect(1, -300, 200, 20, 20, {"building": "yes"})
    builder.rect(2, 300, -200, 20, 20, {"building": "yes"})
    builder.node(-300, 100, {"place": "suburb", "name": "Foo"}, node_id=7001)
    builder.node(-300, 100, {"place": "suburb", "name": "Bar"}, node_id=7002)
    builder.node(300, -100, {"place": "suburb", "name": "Baz"}, node_id=7003)
    builder.write(tmp_path / "mixed.osm")

    mixed = extract_osm(tmp_path / "mixed.osm", city_config)

    assert sorted(d["name"] for d in mixed.districts) == ["Baz", "Foo"]
    assert all(d["buildings"] == 1 for d in mixed.districts)
