import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from osm_fixture import TinyCity

from twl_citybake.cli import main
from twl_citybake.config import BBox, CityConfig, DistrictConfig, load_city
from twl_citybake.osm.clip import clip_extract
from twl_citybake.osm.extract import OsmLayers, extract_osm, write_osm_layers
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table

TABLES = ["buildings", "building_rings", "road_nodes", "road_edges", "road_geom", "water"]
JSON_FILES = ["districts.json", "report.json", "layer.json"]
AHMEDABAD_BBOX = BBox(72.45, 22.92, 72.72, 23.14)


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture(scope="module")
def written(layers: OsmLayers, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("layers") / "out"
    write_osm_layers(layers, out)
    return out


def test_layer_directory_layout(written: Path) -> None:
    assert sorted(p.name for p in written.iterdir()) == sorted(TABLES + JSON_FILES)
    for table in TABLES:
        assert (written / table).is_dir()
        assert (written / table / "_table.json").is_file()


def test_building_table_round_trips_through_disk(layers: OsmLayers, written: Path) -> None:
    loaded = read_table(written / "buildings")

    assert list(loaded) == list(layers.buildings)
    for name, column in layers.buildings.items():
        assert loaded[name].dtype == column.dtype
        np.testing.assert_array_equal(loaded[name], column)


def test_road_tables_round_trip_through_disk(layers: OsmLayers, written: Path) -> None:
    for table, columns in [("road_nodes", layers.road_nodes), ("road_edges", layers.road_edges)]:
        loaded = read_table(written / table)
        for name, column in columns.items():
            np.testing.assert_array_equal(loaded[name], column)


def test_ring_geometry_and_water_arrays_are_stored_as_npy(layers: OsmLayers, written: Path) -> None:
    stored = {
        "building_rings": layers.rings,
        "road_geom": layers.road_geom,
        "water": layers.water,
    }
    for table, columns in stored.items():
        for name, column in columns.items():
            array = np.load(written / table / f"{name}.npy", allow_pickle=False)
            assert array.dtype == column.dtype
            np.testing.assert_array_equal(array, column)


def test_report_json_matches_the_in_memory_report(layers: OsmLayers, written: Path) -> None:
    assert json.loads((written / "report.json").read_text()) == layers.report


def test_report_has_the_specified_keys(layers: OsmLayers) -> None:
    report = layers.report
    assert set(report) == {
        "buildings",
        "roads",
        "water",
        "banks_split",
        "bank_share_west",
        "districts",
        "coverage",
    }
    assert set(report["buildings"]) == {
        "count",
        "skipped_invalid",
        "by_kind",
        "by_height_source",
        "height_m",
        "levels_histogram",
        "footprint_m2",
    }
    assert set(report["buildings"]["height_m"]) == {"p5", "p25", "p50", "p75", "p95", "max"}
    assert set(report["buildings"]["footprint_m2"]) == {"p5", "p50", "p95"}
    assert set(report["roads"]) == {
        "nodes",
        "edges",
        "length_km_by_class",
        "components",
        "largest_component_share",
    }
    assert set(report["water"]) == {"river_area_km2", "lakes", "lake_area_km2"}
    assert set(report["districts"]) == {
        "count",
        "buildings_min",
        "buildings_median",
        "buildings_max",
        "extra_seeds",
        "merged",
        "renamed",
    }
    assert {"cell_m", "cells_in_bbox", "cells_without_buildings"} <= set(report["coverage"])


def test_report_exact_values_on_the_fixture(layers: OsmLayers) -> None:
    buildings = layers.report["buildings"]
    by_kind = {k.lower(): v for k, v in buildings["by_kind"].items() if v}
    by_source = {k.lower(): v for k, v in buildings["by_height_source"].items() if v}

    assert buildings["count"] == 61
    assert buildings["skipped_invalid"] == 1
    assert by_kind == {"unknown": 6, "residential": 53, "commercial": 2}
    assert by_source == {"height": 3, "levels": 51, "default": 7}
    assert buildings["height_m"]["p50"] == pytest.approx(12.8, abs=0.001)
    assert buildings["height_m"]["max"] == pytest.approx(19.2, abs=0.001)
    assert (
        buildings["height_m"]["p5"] <= buildings["height_m"]["p50"] <= buildings["height_m"]["p95"]
    )
    assert buildings["footprint_m2"]["p50"] == pytest.approx(400, abs=1)
    assert layers.report["roads"]["nodes"] == 13
    assert layers.report["roads"]["edges"] == 11
    assert layers.report["water"]["lakes"] == 2
    assert layers.report["banks_split"] is True


def test_report_floats_are_rounded_to_three_decimals(layers: OsmLayers) -> None:
    def floats(value: object) -> list[float]:
        if isinstance(value, float):
            return [value]
        if isinstance(value, dict):
            return [f for v in value.values() for f in floats(v)]
        return []

    found = floats(layers.report)
    assert found
    assert all(round(f, 3) == f for f in found)


def test_coverage_counts_one_kilometre_cells_inside_the_bbox(layers: OsmLayers) -> None:
    coverage = layers.report["coverage"]

    assert coverage["cell_m"] == 1000
    assert coverage["cells_in_bbox"] == 4
    assert coverage["cells_without_buildings"] == 0


def test_coverage_reports_cells_with_no_buildings(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    from osm_fixture import OsmBuilder

    builder = OsmBuilder(tiny_city.projection)
    builder.rect(1, -400, 400, 20, 20, {"building": "yes"})
    builder.write(tmp_path / "sparse.osm")

    sparse = extract_osm(tmp_path / "sparse.osm", city_config)

    assert sparse.report["coverage"]["cells_in_bbox"] == 4
    assert sparse.report["coverage"]["cells_without_buildings"] == 3


def test_layer_json_carries_attribution_and_source(written: Path, city_config: CityConfig) -> None:
    layer = json.loads((written / "layer.json").read_text(encoding="utf-8"))
    proj4 = LocalProjection.for_bbox(city_config.bbox).proj4

    assert layer["city"] == "tinyville"
    assert layer["proj4"] == proj4
    assert layer["attribution"] == "© OpenStreetMap contributors, ODbL 1.0"
    assert layer["source"] == {"url": city_config.osm.url, "md5": city_config.osm.md5}
    bbox = layer["bbox"]
    values = sorted(bbox.values() if isinstance(bbox, dict) else bbox)
    assert values == pytest.approx(
        sorted(
            [
                city_config.bbox.west,
                city_config.bbox.south,
                city_config.bbox.east,
                city_config.bbox.north,
            ]
        )
    )


def test_districts_json_matches_in_memory_districts(layers: OsmLayers, written: Path) -> None:
    assert json.loads((written / "districts.json").read_text(encoding="utf-8")) == layers.districts


def test_extract_and_write_twice_gives_identical_bytes(
    tmp_path: Path, tiny_city: TinyCity, city_config: CityConfig
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    write_osm_layers(extract_osm(tiny_city.path, city_config), first)
    write_osm_layers(extract_osm(tiny_city.path, city_config), second)

    assert tree_bytes(first) == tree_bytes(second)
    assert tree_bytes(first)


def test_cli_osm_bakes_layers_from_the_fetched_extract(
    tmp_path: Path,
    tiny_city: TinyCity,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ahmedabad = load_city("ahmedabad")
    plain = dataclasses.replace(
        ahmedabad, districts=DistrictConfig(place_kinds=ahmedabad.districts.place_kinds)
    )
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: plain)
    stage = tmp_path / "cache" / "citybake" / "ahmedabad" / "osm"
    stage.mkdir(parents=True)
    clip_extract(tiny_city.path, stage / "ahmedabad.osm.pbf", AHMEDABAD_BBOX)

    assert main(["--data-dir", str(tmp_path), "osm", "ahmedabad"]) == 0

    layers_dir = stage / "layers"
    report = json.loads((layers_dir / "report.json").read_text())
    assert report["buildings"]["count"] == 62
    assert report["buildings"]["skipped_invalid"] == 1
    assert sorted(p.name for p in layers_dir.iterdir()) == sorted(TABLES + JSON_FILES)
    assert capsys.readouterr().out.strip()
