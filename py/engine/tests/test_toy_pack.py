import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from blake3 import blake3

from twl_engine.fixtures import ToyCity, build_toy_city, write_pack
from twl_engine.fixtures.pack import SEGMENTS, TILE_LEVEL, TOY_TIER, pack_floors, toy_contents
from twl_pack import NO_PERSONA, PACK_VERSION, EncodedPack, PackReader, encode_pack

SMALL = 5_000


def pack_files(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=SMALL)


@pytest.fixture(scope="module")
def pack(city: ToyCity, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("pack_a")
    return out, write_pack(city, out)


@pytest.fixture(scope="module")
def reader(pack: tuple[Path, str]) -> PackReader:
    return PackReader(pack[0])


@pytest.fixture(scope="module")
def encoded(city: ToyCity) -> EncodedPack:
    return encode_pack(toy_contents(city))


@pytest.fixture(scope="module")
def half_step(reader: PackReader) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = reader.manifest["bbox"]
    return (max_x - min_x) / 65535 / 2 + 1e-3, (max_y - min_y) / 65535 / 2 + 1e-3


def test_same_seed_writes_byte_identical_packs(
    pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_a, hash_a = pack
    out_b = tmp_path_factory.mktemp("pack_b")
    assert write_pack(build_toy_city(seed=49, residents=SMALL), out_b) == hash_a
    assert pack_files(out_a) == pack_files(out_b)


def test_different_seed_writes_a_different_pack_hash(
    pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    other = build_toy_city(seed=7, residents=SMALL)
    assert write_pack(other, tmp_path_factory.mktemp("pack_seed7")) != pack[1]


def test_changing_one_column_changes_the_pack_hash(
    city: ToyCity, pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    taller = dataclasses.replace(city.buildings, height=city.buildings.height + np.float32(1))
    changed = dataclasses.replace(city, buildings=taller)
    assert write_pack(changed, tmp_path_factory.mktemp("pack_taller")) != pack[1]


def test_returned_pack_hash_is_blake3_of_manifest_bytes(pack: tuple[Path, str]) -> None:
    out, pack_hash = pack
    assert pack_hash == blake3((out / "manifest.json").read_bytes()).hexdigest()


def test_pack_directory_holds_the_toy_files_and_verifies(
    pack: tuple[Path, str], reader: PackReader
) -> None:
    reader.verify()
    assert set(pack_files(pack[0])) == {
        "buildings.bin",
        "districts.json",
        "graph.bin",
        "manifest.json",
        "residents.bin",
        "residents_engine.bin",
        "river.bin",
        "roads.bin",
        "stores.bin",
    }


def test_manifest_describes_a_toy_pack_of_this_city(
    pack: tuple[Path, str], city: ToyCity, reader: PackReader
) -> None:
    manifest = json.loads((pack[0] / "manifest.json").read_bytes())
    assert manifest == reader.manifest
    assert manifest["format"] == "twinlight-pack"
    assert manifest["version"] == PACK_VERSION == 2
    assert manifest["kind"] == "toy"
    assert manifest["name"] == "toy"
    assert manifest["seed"] == 49
    assert manifest["crs"] == "local"
    assert manifest["timezone"] == city.timezone == "Asia/Kolkata"
    assert manifest["weekend_days"] == list(city.weekend_days) == [6, 7]
    assert manifest["segments"] == list(SEGMENTS)
    assert manifest["bands"] == []
    assert manifest["attributions"] == []
    assert manifest["counts"] == {
        "bands": 0,
        "buildingTileLevel": TILE_LEVEL,
        "buildings": city.buildings.x.size,
        "detailBuildings": 0,
        "hexes": 0,
        "residents": SMALL,
        "riverPoints": city.river.x.size,
        "riverPolygons": city.river.kind.size,
        "riverRings": city.river.offsets.size - 1,
        "roadPoints": city.roads.x.size,
        "roads": city.roads.kind.size,
        "stores": 14,
        "ties": city.ties.targets.size,
    }


def test_manifest_bbox_encloses_buildings_roads_residents_and_stores(
    reader: PackReader, city: ToyCity
) -> None:
    min_x, min_y, max_x, max_y = reader.manifest["bbox"]
    for part in (city.buildings, city.roads, city.residents, city.stores):
        assert min_x <= part.x.min() and part.x.max() <= max_x
        assert min_y <= part.y.min() and part.y.max() <= max_y


def test_residents_keep_their_order_position_and_segment(
    reader: PackReader, city: ToyCity, half_step: tuple[float, float]
) -> None:
    residents, source = reader.residents(), city.residents
    np.testing.assert_allclose(reader.x_m(residents.qx), source.x, atol=half_step[0], rtol=0)
    np.testing.assert_allclose(reader.y_m(residents.qy), source.y, atol=half_step[1], rtol=0)
    np.testing.assert_array_equal(residents.segment, source.segment)
    np.testing.assert_array_equal(residents.floor, pack_floors(source.z))


def test_each_resident_points_at_the_same_building_in_pack_order(
    reader: PackReader, encoded: EncodedPack, city: ToyCity, half_step: tuple[float, float]
) -> None:
    home_in_pack = reader.residents().building
    np.testing.assert_array_equal(encoded.building_order[home_in_pack], city.residents.building)
    buildings, home = reader.buildings(), city.residents.building
    x, y = reader.x_m(buildings.qx[home_in_pack]), reader.y_m(buildings.qy[home_in_pack])
    np.testing.assert_allclose(x, city.buildings.x[home], atol=half_step[0], rtol=0)
    np.testing.assert_allclose(y, city.buildings.y[home], atol=half_step[1], rtol=0)


def test_no_resident_stands_above_their_roof(reader: PackReader) -> None:
    residents, buildings = reader.residents(), reader.buildings()
    assert (residents.z_m <= buildings.height_m[residents.building]).all()


def test_buildings_keep_their_size_rotation_and_district(
    reader: PackReader, encoded: EncodedPack, city: ToyCity
) -> None:
    buildings, source, order = reader.buildings(), city.buildings, encoded.building_order
    assert sorted(order.tolist()) == list(range(source.x.size))
    for decoded, original in (
        (buildings.width_m, source.width),
        (buildings.depth_m, source.depth),
        (buildings.height_m, source.height),
    ):
        half_code_step = np.sqrt(original[order].astype(np.float64)) / 12 + 1 / 576
        assert (np.abs(decoded - original[order]) <= half_code_step + 1e-3).all()
    turned = np.abs(
        np.mod(source.rotation[order].astype(np.float64), np.pi) - buildings.rotation_rad
    )
    assert (np.minimum(turned, np.pi - turned) <= np.pi / 255 / 2 + 1e-6).all()
    np.testing.assert_array_equal(buildings.district, source.district[order])
    assert buildings.tile_start is not None and buildings.tile_start.size == 4**TILE_LEVEL + 1


def test_roads_keep_their_polylines_and_class(
    reader: PackReader, city: ToyCity, half_step: tuple[float, float]
) -> None:
    roads, source = reader.roads(), city.roads
    np.testing.assert_array_equal(roads.offsets, source.offsets)
    np.testing.assert_array_equal(roads.road_class, source.kind)
    np.testing.assert_allclose(reader.x_m(roads.qx), source.x, atol=half_step[0], rtol=0)
    np.testing.assert_allclose(reader.y_m(roads.qy), source.y, atol=half_step[1], rtol=0)


def test_engine_columns_are_the_toy_constants(reader: PackReader) -> None:
    engine = reader.residents_engine()
    assert (engine.hex == 0).all()
    assert (engine.persona_ref == NO_PERSONA).all()
    assert (engine.weight == 1).all()
    assert (engine.tier == TOY_TIER).all()


def test_stores_river_and_graph_are_exact(reader: PackReader, city: ToyCity) -> None:
    stores, river, graph = reader.stores(), reader.river(), reader.graph()
    for got, want in (
        (stores.x, city.stores.x),
        (stores.y, city.stores.y),
        (stores.radius_m, city.stores.radius_m),
        (stores.district, city.stores.district),
        (river.offsets, city.river.offsets),
        (river.x, city.river.x),
        (river.y, city.river.y),
        (river.kind, city.river.kind),
        (graph.offsets, city.ties.offsets),
        (graph.targets, city.ties.targets),
        (graph.kind, city.ties.kind),
        (graph.strength, city.ties.strength),
    ):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(river.rings, np.ones(city.river.kind.size))


def test_districts_match_the_city(reader: PackReader, city: ToyCity) -> None:
    assert reader.districts() == [dataclasses.asdict(d) for d in city.districts]


@pytest.mark.parametrize(
    ("z", "floor"),
    [(1.0, 0), (1.6, 0), (2.4, 0), (4.79, 0), (4.8, 1), (5.7, 1), (9.0, 2), (12.3, 3)],
)
def test_pack_floor_is_the_highest_at_or_below_the_height(z: float, floor: int) -> None:
    assert pack_floors(np.array([z], dtype=np.float32))[0] == floor


def test_cli_prints_the_same_hash_as_write_pack(
    pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    out = tmp_path_factory.mktemp("pack_cli")
    command = [sys.executable, "-m", "twl_engine.fixtures", "--seed", "49"]
    command += ["--residents", str(SMALL), "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines()[-1].strip() == pack[1]
    assert pack_files(out) == pack_files(pack[0])
