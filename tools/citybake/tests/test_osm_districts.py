import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from osm_fixture import OsmBuilder, TinyCity, ring_area

from twl_citybake.config import CityConfig, DistrictConfig, ExtraSeed, RiverConfig, load_city
from twl_citybake.osm import heights
from twl_citybake.osm.extract import OsmLayers, extract_osm, write_osm_layers
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
        "extra_seeds": 0,
        "merged": 0,
        "renamed": 0,
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


Place = tuple[int, float, float, dict[str, str]]


def build_osm(path: Path, tiny_city: TinyCity, places: Sequence[Place]) -> Path:
    builder = OsmBuilder(tiny_city.projection)
    for osm_id, x, y, tags in places:
        builder.node(x, y, tags, node_id=osm_id)
    builder.node(0, -1200, node_id=6001)
    builder.node(0, 1200, node_id=6002)
    builder.way(300, [6001, 6002], {"waterway": "river", "name": "Sabarmati"})
    for i, (x, y) in enumerate([(-30, 0), (-400, 50), (-790, 40), (-400, 790), (30, 0), (410, 0)]):
        builder.rect(1 + i, x, y, 10, 10, {"building": "yes"})
    builder.write(path)
    return path


def suburb(name: str, **extra: str) -> dict[str, str]:
    return {"place": "suburb", "name": name, **extra}


def four_seeds() -> list[Place]:
    return [
        (8001, -800, 0, suburb("Aa")),
        (8002, -400, 0, suburb("Bb")),
        (8003, -400, 800, suburb("Cc")),
        (8004, 400, 0, suburb("Dd")),
    ]


def bake(
    tmp_path: Path,
    tiny_city: TinyCity,
    city_config: CityConfig,
    places: Sequence[Place],
    **rules: Any,
) -> OsmLayers:
    path = build_osm(tmp_path / "d.osm", tiny_city, places)
    districts = dataclasses.replace(city_config.districts, **rules)
    return extract_osm(path, dataclasses.replace(city_config, districts=districts))


def names_by_building(layers: OsmLayers) -> dict[int, str]:
    b = layers.buildings
    return {
        int(osm_id): layers.districts[int(d)]["name"]
        for osm_id, d in zip(b["osm_id"], b["district"], strict=True)
    }


def test_district_entries_carry_name_local_equal_to_name_without_name_en(layers: OsmLayers) -> None:
    assert [d["name_local"] for d in layers.districts] == ["Alpha", "Beta", "Gamma"]
    assert set(layers.districts[0]) == {
        "id",
        "name",
        "name_local",
        "seed",
        "bank",
        "buildings",
        "polygon",
    }


def test_display_name_prefers_name_en_and_keeps_the_local_name(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    places = [
        (8001, -800, 0, suburb("લપકામણ", **{"name:en": "Lapkaman"})),
        (8002, 400, 0, suburb("Dd")),
    ]

    result = bake(tmp_path, tiny_city, city_config, places)

    assert [(d["name"], d["name_local"]) for d in result.districts] == [
        ("Dd", "Dd"),
        ("Lapkaman", "લપકામણ"),
    ]


@pytest.mark.parametrize("name_en", ["", "--", " "])
def test_name_en_without_alphanumerics_falls_back_to_name(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig, name_en: str
) -> None:
    places = [(8001, -800, 0, suburb("Santej", **{"name:en": name_en}))]

    result = bake(tmp_path, tiny_city, city_config, places)

    assert [(d["name"], d["name_local"]) for d in result.districts] == [("Santej", "Santej")]


@pytest.mark.parametrize(("gujarati_id", "latin_id"), [(8001, 8002), (8002, 8001)])
def test_name_en_dedupes_against_a_latin_named_seed_and_lowest_id_wins(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig, gujarati_id: int, latin_id: int
) -> None:
    places = [
        (gujarati_id, -800, 0, suburb("લપકામણ", **{"name:en": "Lapkaman"})),
        (latin_id, -400, 0, suburb("Lapkaman")),
    ]

    result = bake(tmp_path, tiny_city, city_config, places)

    assert len(result.districts) == 1
    (district,) = result.districts
    won_gujarati = gujarati_id < latin_id
    assert district["name"] == "Lapkaman"
    assert district["name_local"] == ("લપકામણ" if won_gujarati else "Lapkaman")
    expected = (-800, 0) if won_gujarati else (-400, 0)
    assert tuple(district["seed"]) == pytest.approx(expected, abs=0.1)


def test_extra_seed_at_a_plain_node_becomes_a_district(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    places = [*four_seeds(), (9001, -30, 300, {})]

    result = bake(tmp_path, tiny_city, city_config, places, extra_seeds=(ExtraSeed("Vatva", 9001),))

    vatva = next(d for d in result.districts if d["name"] == "Vatva")
    assert vatva["name_local"] == "Vatva"
    assert vatva["bank"] == 0
    assert tuple(vatva["seed"]) == pytest.approx((-30, 300), abs=0.1)
    assert vatva["buildings"] >= 1
    assert names_by_building(result)[1] == "Vatva"
    assert result.report["districts"]["extra_seeds"] == 1
    assert result.report["districts"]["count"] == 5


def test_extra_seed_replaces_a_same_named_place_even_with_a_lower_id(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    places = [
        (100, -700, -600, suburb("vatva ")),
        (9001, -30, 300, {}),
        (8004, 400, 0, suburb("Dd")),
    ]

    result = bake(tmp_path, tiny_city, city_config, places, extra_seeds=(ExtraSeed("Vatva", 9001),))

    vatva = [d for d in result.districts if d["name"] == "Vatva"]
    assert len(result.districts) == 2
    assert len(vatva) == 1
    assert vatva[0]["name_local"] == "Vatva"
    assert tuple(vatva[0]["seed"]) == pytest.approx((-30, 300), abs=0.1)


def test_extra_seed_node_missing_from_the_file_raises_naming_the_node(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    with pytest.raises(ValueError, match="777777"):
        bake(
            tmp_path,
            tiny_city,
            city_config,
            four_seeds(),
            extra_seeds=(ExtraSeed("Ghost", 777777),),
        )


def test_merge_relabels_the_merged_seed_buildings_to_the_target(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    before = bake(tmp_path, tiny_city, city_config, four_seeds())
    assert names_by_building(before)[1] == "Bb"
    assert names_by_building(before)[2] == "Bb"

    merged = bake(tmp_path, tiny_city, city_config, four_seeds(), merge={"Bb": "Aa"})

    assert [d["name"] for d in merged.districts] == ["Aa", "Cc", "Dd"]
    assert [d["id"] for d in merged.districts] == [0, 1, 2]
    names = names_by_building(merged)
    assert [names[i] for i in range(1, 7)] == ["Aa", "Aa", "Aa", "Cc", "Dd", "Dd"]
    assert merged.report["districts"]["merged"] == 1
    assert sum(d["buildings"] for d in merged.districts) == 6


def test_merged_buildings_go_to_the_target_even_when_another_seed_is_nearer(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    merged = bake(tmp_path, tiny_city, city_config, four_seeds(), merge={"Bb": "Cc"})

    names = names_by_building(merged)
    assert [names[i] for i in range(1, 7)] == ["Cc", "Cc", "Aa", "Cc", "Dd", "Dd"]
    assert [d["name"] for d in merged.districts] == ["Aa", "Cc", "Dd"]


def test_merge_target_covers_both_cells_and_keeps_its_own_seed_and_local_name(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    plain = bake(tmp_path, tiny_city, city_config, four_seeds())
    merged = bake(tmp_path, tiny_city, city_config, four_seeds(), merge={"Bb": "Cc"})

    target = next(d for d in merged.districts if d["name"] == "Cc")
    original = next(d for d in plain.districts if d["name"] == "Cc")
    assert tuple(target["seed"]) == tuple(original["seed"])
    assert target["name_local"] == "Cc"
    assert target["bank"] == 0
    assert inside((-400, 0), target["polygon"])
    assert inside((-400, 800), target["polygon"])
    assert inside((-30, 0), target["polygon"])
    area = sum(ring_area(r) for r in target["polygon"])
    parts = sum(ring_area(d["polygon"][0]) for d in plain.districts if d["name"] in ("Bb", "Cc"))
    assert area == pytest.approx(parts, rel=0.01)


def test_merge_matches_names_by_normalised_form(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    result = bake(tmp_path, tiny_city, city_config, four_seeds(), merge={" b-b": "AA "})

    assert [d["name"] for d in result.districts] == ["Aa", "Cc", "Dd"]


def test_merge_applies_after_dedupe_of_cross_script_names(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    places = [
        (8001, -800, 0, suburb("Aa")),
        (8002, -400, 0, suburb("બીબી", **{"name:en": "Bb"})),
        (8003, -400, 800, suburb("Bb")),
    ]

    result = bake(tmp_path, tiny_city, city_config, places, merge={"Bb": "Aa"})

    assert [d["name"] for d in result.districts] == ["Aa"]
    assert result.report["districts"]["merged"] == 1


def test_rename_changes_the_display_name_and_renumbers_by_final_name(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    result = bake(tmp_path, tiny_city, city_config, four_seeds(), rename={"cc": "Aardvark"})

    assert [d["name"] for d in result.districts] == ["Aa", "Aardvark", "Bb", "Dd"]
    assert [d["name_local"] for d in result.districts] == ["Aa", "Cc", "Bb", "Dd"]
    assert [d["id"] for d in result.districts] == [0, 1, 2, 3]
    assert names_by_building(result)[4] == "Aardvark"
    assert result.districts[1]["bank"] == 0
    assert result.report["districts"]["renamed"] == 1


def test_rename_applies_after_merge(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    result = bake(
        tmp_path,
        tiny_city,
        city_config,
        four_seeds(),
        merge={"Bb": "Aa"},
        rename={"Aa": "Zed"},
    )

    assert [d["name"] for d in result.districts] == ["Cc", "Dd", "Zed"]
    assert names_by_building(result)[1] == "Zed"
    assert (result.report["districts"]["merged"], result.report["districts"]["renamed"]) == (1, 1)


@pytest.mark.parametrize(
    ("rules", "culprit"),
    [
        ({"merge": {"Nowhere": "Aa"}}, "Nowhere"),
        ({"merge": {"Bb": "Elsewhere"}}, "Elsewhere"),
        ({"rename": {"Typo": "Aa2"}}, "Typo"),
    ],
)
def test_unmatched_merge_or_rename_keys_and_targets_raise_naming_the_culprit(
    tmp_path: Path,
    tiny_city: TinyCity,
    city_config: CityConfig,
    rules: dict[str, Any],
    culprit: str,
) -> None:
    with pytest.raises(ValueError, match=culprit):
        bake(tmp_path, tiny_city, city_config, four_seeds(), **rules)


@pytest.mark.parametrize("target", ["Aa", "a-a", "AA "])
def test_rename_colliding_with_another_final_name_raises(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig, target: str
) -> None:
    with pytest.raises(ValueError):
        bake(tmp_path, tiny_city, city_config, four_seeds(), rename={"Cc": target})


def test_ahmedabad_config_carries_the_district_rules() -> None:
    districts = load_city("ahmedabad").districts

    assert districts.extra_seeds == (ExtraSeed("Vatva", 4398457021),)
    assert dict(districts.merge) == {"Koteshwar Village": "Koteshwar"}
    assert dict(districts.rename) == {
        "Ramol -Jantanagar": "Ramol",
        "Jantanagar -Ramol": "Jantanagar",
        "vejalpur": "Vejalpur",
        "DaniLimda": "Danilimda",
        "Narol gam": "Narol",
    }


def test_district_rules_default_to_empty() -> None:
    plain = DistrictConfig()

    assert plain.extra_seeds == ()
    assert not plain.merge
    assert not plain.rename


def test_districts_json_carries_final_name_and_local_name(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    places = [
        (8001, -800, 0, suburb("લપકામણ", **{"name:en": "Lapkaman"})),
        (8004, 400, 0, suburb("dd")),
    ]
    result = bake(tmp_path, tiny_city, city_config, places, rename={"dd": "Dd"})

    write_osm_layers(result, tmp_path / "out")

    entries = json.loads((tmp_path / "out" / "districts.json").read_text(encoding="utf-8"))
    assert [(e["name"], e["name_local"]) for e in entries] == [
        ("Dd", "dd"),
        ("Lapkaman", "લપકામણ"),
    ]


def test_rules_given_in_any_order_write_byte_identical_layers(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    places = [*four_seeds(), (9001, -30, 300, {}), (9002, 30, 300, {})]
    merge_a: Mapping[str, str] = {"Bb": "Aa", "Dd": "Eee"}
    merge_b: Mapping[str, str] = dict(reversed(list(merge_a.items())))
    rename_a: Mapping[str, str] = {"Cc": "Cz", "Aa": "Ay"}
    rename_b: Mapping[str, str] = dict(reversed(list(rename_a.items())))
    extras = (ExtraSeed("Vatva", 9001), ExtraSeed("Eee", 9002))
    runs = [
        (merge_a, rename_a, extras),
        (merge_b, rename_b, extras[::-1]),
        (merge_a, rename_a, extras),
    ]
    outputs: list[dict[str, bytes]] = []
    for i, (merge, rename, seeds) in enumerate(runs):
        result = bake(
            tmp_path, tiny_city, city_config, places, extra_seeds=seeds, merge=merge, rename=rename
        )
        out = tmp_path / f"run{i}"
        write_osm_layers(result, out)
        outputs.append(
            {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
        )

    assert outputs[0] == outputs[1] == outputs[2]
    assert b"Vatva" in outputs[0]["districts.json"]
