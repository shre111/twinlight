import dataclasses
import json
import math
from pathlib import Path

import blake3
import numpy as np
import pytest
import zstandard

from twl_pack.format import PackError
from twl_pack.reader import PackReader
from twl_pack.writer import encode_pack, write_pack

COMPRESS = zstandard.ZstdCompressor(level=19, write_checksum=False, write_content_size=True)


def written(contents, directory: Path):
    encoded = encode_pack(contents)
    write_pack(encoded, directory)
    return encoded


def edit_manifest(directory: Path, edit) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    edit(manifest)
    path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def swap_raw(directory: Path, name: str, raw: bytes, *, relist: bool) -> None:
    (directory / name).write_bytes(COMPRESS.compress(raw))
    if relist:
        edit_manifest(
            directory, lambda m: m["files"].__setitem__(name, blake3.blake3(raw).hexdigest())
        )


@pytest.fixture
def pack_dir(city, tmp_path):
    written(city, tmp_path)
    return tmp_path


def test_reader_exposes_manifest_hash_counts_bbox(city, tmp_path):
    encoded = written(city, tmp_path)
    r = PackReader(tmp_path)
    assert r.pack_hash == encoded.pack_hash
    assert r.manifest == json.loads(encoded.manifest)
    assert r.counts["buildings"] == 6
    assert r.bbox == (0, 0, 65535, 65535)
    assert isinstance(r.bbox, tuple)


def test_buildings_round_trip(pack_dir):
    b = PackReader(pack_dir).buildings()
    assert b.tile_start is not None
    assert b.tile_start.tolist() == [0, 4, 5, 6, 6]
    assert b.code.tolist() == [0, 1, 1, 2, 0x55555555, 0xAAAAAAAA]
    assert b.qx.tolist() == [0, 1, 1, 0, 65535, 0]
    assert b.qy.tolist() == [0, 0, 0, 1, 0, 65535]
    assert b.width.tolist() == [120, 48, 72, 60, 24, 36]
    assert b.depth.tolist() == [24, 60, 36, 48, 120, 72]
    assert b.height.tolist() == [93, 66, 85, 76, 38, 54]
    assert b.rotation.tolist() == [191, 85, 0, 170, 0, 64]
    assert b.district.tolist() == [0, 0, 1, 1, 0, 1]


def test_buildings_metres_within_quantisation_tolerance(pack_dir):
    b = PackReader(pack_dir).buildings()
    assert b.width_m.tolist() == [100.0, 16.0, 36.0, 25.0, 4.0, 9.0]
    assert b.depth_m.tolist() == [4.0, 25.0, 9.0, 16.0, 100.0, 36.0]
    heights = [60.0, 30.0, 50.0, 40.0, 10.0, 20.0]
    # a code step near h metres is about h * (1/ (2 * code)) metres
    assert b.height_m == pytest.approx(heights, rel=0.02)
    rotations = [-math.pi / 4 % math.pi, math.pi / 3, 0.0, 2 * math.pi / 3, 0.0, math.pi / 4]
    assert b.rotation_rad == pytest.approx(rotations, abs=math.pi / 255)


def test_detail_round_trip(pack_dir):
    d = PackReader(pack_dir).detail()
    assert d is not None
    assert d.tile_start is None
    assert d.code.tolist() == [12, 51]
    assert d.qx.tolist() == [2, 5]
    assert d.width.tolist() == [36, 24]
    assert d.district.tolist() == [0, 1]


def test_roads_round_trip(pack_dir):
    r = PackReader(pack_dir).roads()
    assert r.points.tolist() == [3, 1]
    assert r.road_class.tolist() == [0, 2]
    assert r.qx.tolist() == [65535, 0, 40000, 7]
    assert r.qy.tolist() == [0, 0, 0, 9]
    assert r.offsets.dtype == np.int64
    assert r.offsets.tolist() == [0, 3, 4]


def test_residents_round_trip(pack_dir):
    r = PackReader(pack_dir).residents()
    assert r.qx.tolist() == [10, 65535, 0]
    assert r.qy.tolist() == [20, 0, 65535]
    assert r.floor.tolist() == [0, 3, 255]
    assert r.building.tolist() == [4, 0, 1]
    assert r.segment.tolist() == [0, 1, 1]
    assert r.z_m == pytest.approx([1.6, 11.2, 817.6])


def test_residents_engine_stores_graph_round_trip(pack_dir):
    reader = PackReader(pack_dir)
    e = reader.residents_engine()
    assert e.hex.tolist() == [1, 0, 1]
    assert e.persona_ref.tolist() == [5, 0xFFFFFFFF, 7]
    assert e.weight.tolist() == [1.0, 0.5, 2.0]
    s = reader.stores()
    assert s.x.tolist() == [100.5, 300.0]
    assert s.y.tolist() == [200.25, 400.0]
    assert s.hex.tolist() == [0, 1]
    assert s.radius_m.tolist() == [1500.0, 2000.5]
    assert s.district.tolist() == [0, 1]
    g = reader.graph()
    assert g.offsets.tolist() == [0, 2, 3, 3]
    assert g.targets.tolist() == [1, 2, 0]
    assert g.kind.tolist() == [0, 2, 3]
    assert g.strength.tolist() == [0.5, 1.0, 0.25]


def test_river_round_trip(pack_dir):
    r = PackReader(pack_dir).river()
    assert r.offsets.tolist() == [0, 4, 8, 11]
    assert r.x.tolist() == [0, 10, 10, 0, 2, 2, 8, 8, 20, 30, 25]
    assert r.y.tolist() == [0, 0, 10, 10, 2, 8, 8, 2, 20, 20, 30]
    assert r.kind.tolist() == [0, 1]
    assert r.rings.tolist() == [2, 1]


def test_hexes_and_travel_shape_and_order(pack_dir):
    reader = PackReader(pack_dir)
    hexes, t = reader.hexes(), reader.travel()
    assert hexes is not None
    assert t is not None
    assert hexes.tolist() == [0x8828308281FFFFF, 0x8828308283FFFFF]
    assert t.shape == (2, 2, 2)
    assert t[0, 1, 0] == 7  # flat index store * 4 + hex * 2 + band
    assert t[1, 0, 1] == 10
    assert t.ravel().tolist() == [5, 6, 7, 8, 9, 10, 11, 12]


def test_districts_round_trip_with_non_ascii(pack_dir, city):
    assert PackReader(pack_dir).districts() == city.districts


def test_metres_from_quantised_use_bbox(toy, tmp_path):
    written(toy, tmp_path)
    reader = PackReader(tmp_path)
    r = reader.residents()
    step = 550 / 65535
    assert reader.x_m(r.qx) == pytest.approx(toy.residents.x, abs=step)
    assert reader.y_m(r.qy) == pytest.approx(toy.residents.y, abs=step)
    assert reader.x_m(np.array([0, 65535], dtype=np.uint16)).tolist() == [-300.0, 250.0]


def test_toy_has_no_cells_and_no_detail(toy, tmp_path):
    written(toy, tmp_path)
    reader = PackReader(tmp_path)
    assert reader.hexes() is None
    assert reader.travel() is None
    assert reader.detail() is None
    assert reader.residents_engine().persona_ref.tolist() == [0xFFFFFFFF] * 4
    assert reader.residents_engine().weight.tolist() == [1.0] * 4
    reader.verify()


def test_toy_river_survives_outside_bbox(toy, tmp_path):
    written(toy, tmp_path)
    assert PackReader(tmp_path).river().x.tolist() == [-6000, 6000, 6000, -6000]


def test_detail_is_none_when_the_pack_has_no_detail_file(city, tmp_path):
    written(dataclasses.replace(city, detail=None), tmp_path)
    assert not (tmp_path / "buildings_detail.bin").exists()
    reader = PackReader(tmp_path)
    assert reader.detail() is None
    reader.verify()


def test_position_deltas_past_the_largest_code_are_caught(city, tmp_path):
    raw = bytearray(written(city, tmp_path).raw["buildings.bin"])
    tile_table = 24  # 5 entries of 4 bytes, padded to 8
    raw[tile_table : tile_table + 6 * 4] = b"\xff" * (6 * 4)
    swap_raw(tmp_path, "buildings.bin", bytes(raw), relist=True)
    with pytest.raises(PackError, match="position_delta"):
        PackReader(tmp_path).buildings()


def test_verify_needs_the_detail_file_listed(pack_dir):
    edit_manifest(pack_dir, lambda m: m["files"].pop("buildings_detail.bin"))
    with pytest.raises(PackError, match=r"buildings_detail\.bin"):
        PackReader(pack_dir).verify()


def test_verify_passes_on_a_good_pack(pack_dir):
    PackReader(pack_dir).verify()


@pytest.mark.parametrize("name", ["buildings.bin", "roads.bin", "districts.json", "travel.bin"])
def test_flipped_byte_in_an_object_is_caught(city, tmp_path, name):
    encoded = written(city, tmp_path)
    raw = bytearray(encoded.raw[name])
    raw[0] ^= 0x01
    swap_raw(tmp_path, name, bytes(raw), relist=False)
    with pytest.raises(PackError):
        PackReader(tmp_path).verify()


def test_flipped_byte_in_compressed_bytes_is_caught(city, tmp_path):
    written(city, tmp_path)
    path = tmp_path / "residents.bin"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(PackError):
        PackReader(tmp_path).verify()
    with pytest.raises(PackError):
        PackReader(tmp_path).residents()


@pytest.mark.parametrize("method", ["buildings", "roads", "residents", "stores", "river", "graph"])
def test_truncated_object_is_caught_by_the_accessor(city, tmp_path, method):
    encoded = written(city, tmp_path)
    name = {
        "buildings": "buildings.bin",
        "roads": "roads.bin",
        "residents": "residents.bin",
        "stores": "stores.bin",
        "river": "river.bin",
        "graph": "graph.bin",
    }[method]
    (tmp_path / name).write_bytes(encoded.objects[name][:-4])
    with pytest.raises(PackError):
        getattr(PackReader(tmp_path), method)()


def test_truncated_object_is_caught_by_verify(pack_dir):
    path = pack_dir / "graph.bin"
    path.write_bytes(path.read_bytes()[:5])
    with pytest.raises(PackError):
        PackReader(pack_dir).verify()


@pytest.mark.parametrize(
    ("name", "method"),
    [("residents.bin", "residents"), ("districts.json", "districts"), ("travel.bin", "travel")],
)
def test_missing_listed_file_is_caught(pack_dir, name, method):
    (pack_dir / name).unlink()
    reader = PackReader(pack_dir)
    with pytest.raises(PackError):
        reader.verify()
    with pytest.raises(PackError):
        getattr(reader, method)()


def test_nonzero_padding_is_caught(city, tmp_path):
    encoded = written(city, tmp_path)
    raw = bytearray(encoded.raw["roads.bin"])
    raw[4] = 1  # points column is 4 bytes, the next 4 are padding
    swap_raw(tmp_path, "roads.bin", bytes(raw), relist=True)
    reader = PackReader(tmp_path)
    with pytest.raises(PackError):
        reader.roads()
    with pytest.raises(PackError):
        reader.verify()


def test_nonzero_padding_on_the_last_column_is_caught(city, tmp_path):
    encoded = written(city, tmp_path)
    raw = bytearray(encoded.raw["graph.bin"])
    raw[-1] = 9
    swap_raw(tmp_path, "graph.bin", bytes(raw), relist=True)
    with pytest.raises(PackError):
        PackReader(tmp_path).graph()


def test_length_that_disagrees_with_counts_is_caught(city, tmp_path):
    encoded = written(city, tmp_path)
    swap_raw(tmp_path, "stores.bin", encoded.raw["stores.bin"] + bytes(8), relist=True)
    with pytest.raises(PackError):
        PackReader(tmp_path).stores()


def test_frame_without_content_size_is_caught(city, tmp_path):
    encoded = written(city, tmp_path)
    no_size = zstandard.ZstdCompressor(level=19, write_content_size=False)
    (tmp_path / "hexes.bin").write_bytes(no_size.compress(encoded.raw["hexes.bin"]))
    assert zstandard.frame_content_size((tmp_path / "hexes.bin").read_bytes()) == -1
    reader = PackReader(tmp_path)
    with pytest.raises(PackError):
        reader.hexes()
    with pytest.raises(PackError):
        reader.verify()


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.__setitem__("format", "tape"),
        lambda m: m.__setitem__("version", 1),
        lambda m: m.__setitem__("version", 3),
        lambda m: m.__setitem__("compression", "br"),
        lambda m: m.__setitem__("compression", "gzip"),
        lambda m: m.pop("counts"),
        lambda m: m.pop("files"),
        lambda m: m.pop("bbox"),
    ],
    ids=[
        "wrong format",
        "version 1",
        "version 3",
        "compression br",
        "compression gzip",
        "missing counts",
        "missing files",
        "missing bbox",
    ],
)
def test_bad_manifest_is_refused_on_open(pack_dir, edit):
    edit_manifest(pack_dir, edit)
    with pytest.raises(PackError):
        PackReader(pack_dir)


def test_pack_hash_follows_the_manifest_bytes_on_disk(pack_dir):
    reader = PackReader(pack_dir)
    assert reader.pack_hash == blake3.blake3((pack_dir / "manifest.json").read_bytes()).hexdigest()
