import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_type_hints

import numpy as np
import pytest
from footprints_fixture import (
    G29_AT,
    GNONE_AT,
    M_NONE_AT,
    M_STRIP_AT,
    HeightGrid,
    empty_height_grid,
    no_network,  # noqa: F401
    tiny_cache_rows,
    write_vida_cache,
)
from osm_fixture import BBOX, TinyCity
from population_fixture import write_world
from scipy.spatial import KDTree
from test_population_stage import tiny_config as tiny_population_config

from twl_citybake.cli import main
from twl_citybake.config import CityConfig, DataDirs
from twl_citybake.footprints.stage import build_buildings, write_building_layers
from twl_citybake.osm.clip import clip_extract
from twl_citybake.osm.extract import OsmLayers, extract_osm, write_osm_layers
from twl_citybake.osm.tags import BuildingKind, HeightSource
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table, table_digest, write_json

VIDA_FILE = "vida.parquet"
HEIGHTS_FILE = "heights.tif"

OSM, GOOGLE, MICROSOFT = 0, 1, 2
NON_OSM_TYPE = 255
RASTER = 3

EXPECTED_ATTRIBUTION = [
    "© OpenStreetMap contributors, ODbL 1.0",
    "Google-Microsoft-OSM Open Buildings combined by VIDA, ODbL 1.0",
    "Google Open Buildings v3, CC BY 4.0 or ODbL 1.0",
    "Microsoft Global ML Building Footprints (ODbL 1.0, via VIDA's 2024-08-27 snapshot)",
    "Google Open Buildings 2.5D Temporal (Sirko et al. 2023, arXiv:2310.11622), CC BY 4.0 or "
    "ODbL 1.0, contains modified Copernicus Sentinel data",
]

NEW_CENTRES = [GNONE_AT, G29_AT, M_NONE_AT, M_STRIP_AT]
NEW_SOURCES = [GOOGLE, GOOGLE, MICROSOFT, MICROSOFT]


def footprints_config(base: CityConfig) -> CityConfig:
    hint = get_type_hints(CityConfig)["footprints"]
    section = next(arg for arg in get_args(hint) if arg is not type(None))
    return replace(
        base,
        footprints=section(
            vida_url="https://example.invalid/vida.parquet",
            vida_etag='"test-etag"',
            vida_snapshot="2024-08-27",
            heights_manifest="https://example.invalid/manifest.json",
            overlap_drop=0.3,
            raster_min_height_m=1.0,
            raster_cell_m=4.0,
        ),
    )


@dataclass(frozen=True)
class World:
    config: CityConfig
    osm_layers: Path
    osm_pbf: Path
    cache: Path
    heights: Path
    baseline: OsmLayers
    out: Path


def row_of(table: dict[str, np.ndarray], way_id: int) -> int:
    return int(np.flatnonzero((table["osm_id"] == way_id) & (table["osm_type"] == 0))[0])


def paint_heights(base: dict[str, np.ndarray], proj: LocalProjection) -> HeightGrid:
    grid = empty_height_grid(proj)

    def at(way_id: int) -> tuple[float, float]:
        i = row_of(base, way_id)
        return float(base["x"][i]), float(base["y"][i])

    grid.paint(*at(1001), [40.0] * 9)
    grid.paint(*at(1002), [25.6] * 9)
    grid.paint(*at(20000), [9.6] * 9)
    grid.paint(*at(1003), [0.4] * 5 + [9.5] * 4)
    grid.paint(*GNONE_AT, [7.25] * 6 + [30.0] * 3)
    grid.paint(*M_NONE_AT, [0.4] * 9)
    return grid


@pytest.fixture(scope="module")
def world(
    tmp_path_factory: pytest.TempPathFactory,
    tiny_city: TinyCity,
    city_config: CityConfig,
    layers: OsmLayers,
) -> World:
    root = tmp_path_factory.mktemp("footprints")
    config = footprints_config(city_config)
    osm_layers = root / "osm" / "layers"
    write_osm_layers(layers, osm_layers)
    cache = write_vida_cache(root / VIDA_FILE, tiny_cache_rows(), tiny_city.projection)
    heights = paint_heights(layers.buildings, tiny_city.projection).write(root / HEIGHTS_FILE)
    built = build_buildings(osm_layers, tiny_city.path, cache, heights, config)
    out = root / "buildings" / "layers"
    write_building_layers(built, out)
    return World(config, osm_layers, tiny_city.path, cache, heights, layers, out)


@pytest.fixture(scope="module")
def table(world: World) -> dict[str, np.ndarray]:
    return read_table(world.out / "buildings")


@pytest.fixture(scope="module")
def report(world: World) -> dict[str, Any]:
    return json.loads((world.out / "report.json").read_text())


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def walk(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, (*path, key))


def find_dict(report: dict[str, Any], key: str, required: set[str]) -> dict[str, Any]:
    hits = [
        node
        for path, node in walk(report)
        if path and path[-1] == key and isinstance(node, dict) and required <= set(node)
    ]
    assert len(hits) == 1, f"expected one {key!r} entry with {sorted(required)}, found {len(hits)}"
    return hits[0]


def n_osm(world: World) -> int:
    return len(world.baseline.buildings["osm_id"])


def test_the_written_layers_hold_the_building_tables_and_reports(world: World) -> None:
    names = {p.name for p in world.out.iterdir()}
    assert {"buildings", "building_rings", "report.json", "layer.json"} <= names
    assert (world.out / "buildings" / "_table.json").is_file()


def test_the_table_has_the_osm_columns_plus_source_and_confidence(
    world: World, table: dict[str, np.ndarray]
) -> None:
    base = world.baseline.buildings
    assert set(table) == set(base) | {"source", "confidence", "raster_height_m"}
    for name, column in base.items():
        assert table[name].dtype == column.dtype, name
    assert table["source"].dtype == np.uint8
    assert table["confidence"].dtype == np.float32
    assert table["raster_height_m"].dtype == np.float32
    assert len({len(c) for c in table.values()}) == 1


def test_raster_height_keeps_the_raw_median_and_is_nan_without_raster_support(
    table: dict[str, np.ndarray],
) -> None:
    from_raster = table["height_source"] == 3
    np.testing.assert_array_equal(
        table["raster_height_m"][from_raster], table["height_m"][from_raster]
    )
    assert np.isnan(table["raster_height_m"][table["height_source"] == 2]).all()


def test_osm_rows_come_first_and_are_identical_to_the_osm_stage_except_defaulted_heights(
    world: World, table: dict[str, np.ndarray]
) -> None:
    base, n = world.baseline.buildings, n_osm(world)
    for name, column in base.items():
        if name not in {"height_m", "height_source"}:
            np.testing.assert_array_equal(table[name][:n], column, err_msg=name)

    tagged = np.isin(base["height_source"], [HeightSource.HEIGHT, HeightSource.LEVELS])
    np.testing.assert_array_equal(table["height_source"][:n][tagged], base["height_source"][tagged])
    np.testing.assert_array_equal(table["height_m"][:n][tagged], base["height_m"][tagged])
    assert np.isin(table["height_source"][:n][~tagged], [HeightSource.DEFAULT, RASTER]).all()


def test_osm_rows_are_marked_source_osm_with_full_confidence(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    assert (table["source"][:n] == OSM).all()
    assert (table["confidence"][:n] == 1.0).all()


def test_only_the_non_overlapping_footprints_are_appended_google_before_microsoft(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    assert len(table["osm_id"]) == n + len(NEW_CENTRES)
    assert table["source"][n:].tolist() == NEW_SOURCES
    centres = np.column_stack([table["x"][n:], table["y"][n:]])
    np.testing.assert_allclose(centres, np.asarray(NEW_CENTRES), atol=0.05)


def test_a_microsoft_footprint_over_a_dropped_google_one_is_kept(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    strip = np.flatnonzero(table["source"][n:] == MICROSOFT)[-1] + n
    assert (table["x"][strip], table["y"][strip]) == pytest.approx(M_STRIP_AT, abs=0.05)


def test_footprints_overlapping_osm_beyond_thirty_percent_are_absent(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    for x, y in [(-286.2, -700.0), (-700.0, 500.0), (-600.0, 500.0), (-500.0, -500.0)]:
        near = np.hypot(table["x"][n:] - x, table["y"][n:] - y) < 2
        assert not near.any(), (x, y)


def test_new_rows_have_no_osm_identity_and_unknown_kind(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    assert (table["osm_id"][n:] == -1).all()
    assert (table["osm_type"][n:] == NON_OSM_TYPE).all()
    assert (table["kind"][n:] == BuildingKind.UNKNOWN).all()
    assert (table["osm_id"][:n] >= 0).all()


def test_new_rows_carry_footprint_geometry_and_source_confidence(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    np.testing.assert_allclose(table["area_m2"][n:], [400, 400, 400, 100], rtol=0.01)
    assert (np.abs(table["w"][n:] - 20.0) < 0.5).sum() >= 3
    assert table["rot"].dtype == np.float32
    google, microsoft = table["confidence"][n:][:2], table["confidence"][n:][2:]
    np.testing.assert_allclose(google, [0.91, 0.85], rtol=1e-6)
    assert np.isnan(microsoft).all()


def test_every_row_has_a_closed_ring_around_its_footprint(
    world: World, table: dict[str, np.ndarray]
) -> None:
    rings = {n: np.load(world.out / "building_rings" / f"{n}.npy") for n in ("offsets", "xy")}
    offsets, xy = rings["offsets"].astype(np.int64), rings["xy"]
    assert len(offsets) == len(table["x"]) + 1
    base = np.load(world.osm_layers / "building_rings" / "offsets.npy").astype(np.int64)
    base_xy = np.load(world.osm_layers / "building_rings" / "xy.npy")
    np.testing.assert_array_equal(offsets[: len(base)], base)
    np.testing.assert_array_equal(xy[: base[-1]], base_xy)
    for i in range(n_osm(world), len(table["x"])):
        ring = xy[offsets[i] : offsets[i + 1]]
        np.testing.assert_array_equal(ring[0], ring[-1])
        half = float(np.sqrt(table["area_m2"][i])) / 2
        assert ring[:, 0].min() == pytest.approx(table["x"][i] - half, abs=0.5)
        assert ring[:, 1].max() == pytest.approx(table["y"][i] + half, abs=0.5)


def test_raster_is_height_source_three_and_leaves_osm_tagged_heights_alone(
    world: World, table: dict[str, np.ndarray]
) -> None:
    assert HeightSource["RASTER"] == RASTER
    tagged = {1001: (15.0, HeightSource.HEIGHT), 1002: (12.8, HeightSource.LEVELS)}
    tagged[20000] = (9.6, HeightSource.LEVELS)
    for way_id, (height, source) in tagged.items():
        i = row_of(table, way_id)
        assert table["height_m"][i] == pytest.approx(height, abs=1e-4), way_id
        assert table["height_source"][i] == source, way_id


def test_untagged_osm_buildings_take_the_raster_median_ignoring_ground_pixels(
    table: dict[str, np.ndarray],
) -> None:
    i = row_of(table, 1003)
    assert table["height_source"][i] == RASTER
    assert table["height_m"][i] == pytest.approx(9.5)


def test_new_footprints_use_the_raster_median_else_the_default_chain(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    source, height = table["height_source"][n:], table["height_m"][n:]
    assert source.tolist() == [
        RASTER,
        HeightSource.DEFAULT,
        HeightSource.DEFAULT,
        HeightSource.DEFAULT,
    ]
    assert height[0] == pytest.approx(7.25)
    assert (height[1:] > 0).all() and np.isfinite(height).all()


def test_untagged_osm_buildings_without_raster_pixels_keep_a_default_height(
    table: dict[str, np.ndarray],
) -> None:
    i = row_of(table, 1004)
    assert table["height_source"][i] == HeightSource.DEFAULT
    assert table["height_m"][i] > 0


def bank_and_district(world: World, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    districts = json.loads((world.osm_layers / "districts.json").read_text())
    seeds = np.asarray([d["seed"] for d in districts])
    seed_bank = np.asarray([d["bank"] for d in districts])
    bank = (x > 0).astype(np.uint8)
    district = np.zeros(len(x), np.int64)
    for b in (0, 1):
        members = np.flatnonzero(seed_bank == b)
        on_bank = np.flatnonzero(bank == b)
        if len(on_bank):
            _, nearest = KDTree(seeds[members]).query(np.column_stack([x[on_bank], y[on_bank]]))
            district[on_bank] = members[nearest]
    return bank, district


def test_the_bank_and_district_oracle_reproduces_the_osm_stage(world: World) -> None:
    base = world.baseline.buildings
    bank, district = bank_and_district(world, base["x"], base["y"])
    np.testing.assert_array_equal(bank, base["bank"])
    np.testing.assert_array_equal(district, base["district"])


def test_new_footprints_get_the_nearest_seed_on_their_own_bank(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    bank, district = bank_and_district(world, table["x"][n:], table["y"][n:])
    assert set(bank.tolist()) == {0, 1}
    np.testing.assert_array_equal(table["bank"][n:], bank)
    np.testing.assert_array_equal(table["district"][n:], district)


def test_a_new_footprint_nearer_to_a_seed_on_the_other_bank_still_takes_its_own_banks_seed(
    world: World, table: dict[str, np.ndarray]
) -> None:
    n = n_osm(world)
    districts = json.loads((world.osm_layers / "districts.json").read_text())
    names = {d["id"]: d["name"] for d in districts}
    east_south = n + 1
    assert (table["x"][east_south], table["y"][east_south]) == pytest.approx(G29_AT, abs=0.05)
    assert names[int(table["district"][east_south])] == "Beta"


def test_the_report_counts_each_source_through_the_dedupe_tiers(report: dict[str, Any]) -> None:
    expected = {"google": (5, 1, 2, 2), "microsoft": (5, 1, 2, 2)}
    for name, (fetched, invalid, dropped, kept) in expected.items():
        counts = find_dict(report, name, {"fetched", "invalid", "kept"})
        dropped_key = next(k for k in counts if k.startswith("dropped"))
        assert counts["fetched"] == fetched, name
        assert counts["invalid"] == invalid, name
        assert counts[dropped_key] == dropped, name
        assert counts["kept"] == kept, name
        assert counts["fetched"] == counts["invalid"] + counts[dropped_key] + counts["kept"]


def test_the_report_height_sources_match_the_table(
    report: dict[str, Any], table: dict[str, np.ndarray]
) -> None:
    sources = find_dict(report, "by_height_source", {"HEIGHT", "LEVELS", "DEFAULT", "RASTER"})
    counts = np.bincount(table["height_source"], minlength=4)
    assert sources == {s.name: int(counts[s]) for s in HeightSource}
    assert sources["RASTER"] == 2


def test_the_report_checks_the_raster_against_levels_tagged_osm_buildings(
    report: dict[str, Any],
) -> None:
    check = next(
        node
        for _, node in walk(report)
        if isinstance(node, dict) and "n" in node and any("ratio" in k for k in node)
    )
    ratio = next(v for k, v in check.items() if "ratio" in k)
    error = next(v for k, v in check.items() if "error" in k)
    assert check["n"] == 2
    assert ratio == pytest.approx(1.5, rel=1e-2)
    assert error == pytest.approx(6.4, rel=1e-2)


def test_the_report_gives_the_source_mix_per_district(
    world: World, report: dict[str, Any], table: dict[str, np.ndarray]
) -> None:
    names = [d["name"] for d in json.loads((world.osm_layers / "districts.json").read_text())]
    candidates = [
        node
        for _, node in walk(report)
        if isinstance(node, dict)
        and node
        and all(
            isinstance(v, dict) and {"osm", "google", "microsoft"} <= set(v) for v in node.values()
        )
    ]
    assert len(candidates) == 1
    mix = candidates[0]
    assert set(mix) == set(names)
    for district, name in enumerate(names):
        rows = table["district"] == district
        for code, key in ((OSM, "osm"), (GOOGLE, "google"), (MICROSOFT, "microsoft")):
            assert mix[name][key] == int((rows & (table["source"] == code)).sum()), (name, key)
    assert sum(sum(m[k] for k in ("osm", "google", "microsoft")) for m in mix.values()) == len(
        table["source"]
    )


def test_the_coverage_report_cannot_gain_cells_without_buildings(
    world: World, report: dict[str, Any]
) -> None:
    now = find_dict(report, "coverage", {"cells_without_buildings", "cells_in_bbox", "cell_m"})
    before = world.baseline.report["coverage"]
    assert now["cells_in_bbox"] == before["cells_in_bbox"]
    assert now["cells_without_buildings"] <= before["cells_without_buildings"]


def test_layer_json_credits_every_source_with_its_licence(world: World) -> None:
    meta = json.loads((world.out / "layer.json").read_text())
    assert meta["city"] == "tinyville"
    assert meta["attribution"] == EXPECTED_ATTRIBUTION


def test_building_layers_are_byte_identical_across_runs(world: World, tmp_path: Path) -> None:
    again = build_buildings(
        world.osm_layers, world.osm_pbf, world.cache, world.heights, world.config
    )
    write_building_layers(again, tmp_path / "again")
    assert tree_bytes(tmp_path / "again") == tree_bytes(world.out)
    assert tree_bytes(world.out)


def test_the_footprints_command_bakes_from_pre_placed_caches_without_fetching(
    world: World,
    tiny_city: TinyCity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_network: None,  # noqa: F811
) -> None:
    config = world.config
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: config)
    dirs = DataDirs(tmp_path)
    osm_stage = dirs.stage(config.slug, "osm")
    pbf = osm_stage / f"{config.slug}.osm.pbf"
    clip_extract(tiny_city.path, pbf, BBOX)
    write_osm_layers(extract_osm(pbf, config), osm_stage / "layers")
    cache_dir = dirs.stage(config.slug, "footprints")
    cache_dir.mkdir(parents=True)
    shutil.copy(world.cache, cache_dir / VIDA_FILE)
    shutil.copy(world.heights, cache_dir / HEIGHTS_FILE)

    assert main(["--data-dir", str(tmp_path), "footprints", config.slug]) == 0

    out = dirs.stage(config.slug, "buildings") / "layers"
    expected = build_buildings(
        osm_stage / "layers", pbf, cache_dir / VIDA_FILE, cache_dir / HEIGHTS_FILE, config
    )
    write_building_layers(expected, tmp_path / "expected")
    assert tree_bytes(out) == tree_bytes(tmp_path / "expected")
    assert capsys.readouterr().out.strip()
    assert sorted(p.name for p in cache_dir.iterdir()) == sorted([VIDA_FILE, HEIGHTS_FILE])


def test_the_population_command_reads_the_combined_buildings_when_that_stage_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    city = tiny_population_config()
    assert city.worldpop is not None
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    dirs = DataDirs(tmp_path)
    write_world(
        city.bbox,
        dirs.raw / "worldpop" / city.worldpop.filename,
        dirs.stage(city.slug, "osm") / "layers",
        block=18,
    )
    combined = dirs.stage(city.slug, "buildings") / "layers"
    expected = write_world(city.bbox, tmp_path / "unused.tif", combined, block=10)
    osm_digest = table_digest(dirs.stage(city.slug, "osm") / "layers" / "buildings")
    write_json(combined / "layer.json", {"osm_buildings_sha256": osm_digest})
    assert expected["buildings"] != len(
        read_table(dirs.stage(city.slug, "osm") / "layers" / "buildings")["x"]
    )

    assert main(["--data-dir", str(tmp_path), "population", city.slug]) == 0

    persons = read_table(dirs.stage(city.slug, "population") / "building_population")["persons"]
    assert len(persons) == int(expected["buildings"])


def test_the_population_command_refuses_combined_buildings_older_than_the_osm_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    city = tiny_population_config()
    assert city.worldpop is not None
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    dirs = DataDirs(tmp_path)
    write_world(
        city.bbox,
        dirs.raw / "worldpop" / city.worldpop.filename,
        dirs.stage(city.slug, "osm") / "layers",
        block=18,
    )
    combined = dirs.stage(city.slug, "buildings") / "layers"
    write_world(city.bbox, tmp_path / "unused.tif", combined, block=10)
    write_json(combined / "layer.json", {"osm_buildings_sha256": "0" * 64})

    with pytest.raises(SystemExit, match="predates the OSM layers"):
        main(["--data-dir", str(tmp_path), "population", city.slug])
