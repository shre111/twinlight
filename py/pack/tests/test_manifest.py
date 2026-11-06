import dataclasses
import json

import blake3
import numpy as np
import pytest
import zstandard

from twl_pack.writer import EncodedPack, encode_pack, write_pack

KEYS = {
    "format",
    "version",
    "kind",
    "name",
    "crs",
    "bbox",
    "counts",
    "segments",
    "bands",
    "compression",
    "files",
    "sizes",
    "attributions",
}
FIRST_LOAD = [
    "buildings.bin",
    "districts.json",
    "residents.bin",
    "river.bin",
    "roads.bin",
    "stores.bin",
]
ENGINE = ["graph.bin", "hexes.bin", "residents_engine.bin", "travel.bin"]


def manifest_of(contents) -> dict:
    return json.loads(encode_pack(contents).manifest)


def test_city_manifest_has_exactly_the_documented_keys(city):
    assert set(manifest_of(city)) == KEYS


def test_toy_manifest_adds_only_seed(toy):
    m = manifest_of(toy)
    assert set(m) == KEYS | {"seed"}
    assert m["seed"] == 7
    assert m["kind"] == "toy"
    assert m["crs"] == "local"


def test_fixed_fields(city):
    m = manifest_of(city)
    assert m["format"] == "twinlight-pack"
    assert m["version"] == 2
    assert m["compression"] == "zstd"
    assert m["segments"] == ["young", "old"]
    assert m["bands"] == [
        {"name": "morning", "start_minute": 360},
        {"name": "evening", "start_minute": 1080},
    ]
    assert m["name"] == "hand"


def test_manifest_bytes_are_sorted_compact_and_untimestamped(city):
    raw = encode_pack(city).manifest
    m = json.loads(raw)
    assert raw == json.dumps(m, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert b"time" not in raw.lower() and b"date" not in raw.lower()


def test_pack_hash_is_blake3_of_manifest_bytes(city):
    e = encode_pack(city)
    assert e.pack_hash == blake3.blake3(e.manifest).hexdigest()


def test_counts_are_complete_and_exact(city):
    assert manifest_of(city)["counts"] == {
        "buildings": 6,
        "detailBuildings": 2,
        "buildingTileLevel": 1,
        "residents": 3,
        "ties": 3,
        "stores": 2,
        "roads": 2,
        "roadPoints": 4,
        "riverPolygons": 2,
        "riverRings": 3,
        "riverPoints": 11,
        "hexes": 2,
        "bands": 2,
    }


def test_toy_counts_have_zero_hexes_bands_and_detail(toy):
    c = manifest_of(toy)["counts"]
    assert (c["hexes"], c["bands"], c["detailBuildings"]) == (0, 0, 0)
    assert c["buildingTileLevel"] == 2
    assert c["riverRings"] == 1 and c["riverPolygons"] == 1 and c["riverPoints"] == 4


def test_files_hash_uncompressed_bytes(city):
    e = encode_pack(city)
    files = json.loads(e.manifest)["files"]
    assert set(files) == set(e.raw)
    for name, raw in e.raw.items():
        assert files[name] == blake3.blake3(raw).hexdigest()
    assert "manifest.json" not in files


def test_sizes_count_compressed_bytes_by_group(city):
    e = encode_pack(city)
    sizes = json.loads(e.manifest)["sizes"]
    assert sizes["first_load"] == {
        "files": FIRST_LOAD,
        "bytes": sum(len(e.objects[f]) for f in FIRST_LOAD),
        "budget": 6_000_000,
    }
    assert sizes["engine"] == {
        "files": ENGINE,
        "bytes": sum(len(e.objects[f]) for f in ENGINE),
        "budget": 10_000_000,
    }
    assert sizes["detail"] == {
        "files": ["buildings_detail.bin"],
        "bytes": len(e.objects["buildings_detail.bin"]),
    }


def test_sizes_for_toy_list_only_present_files(toy):
    e = encode_pack(toy)
    sizes = json.loads(e.manifest)["sizes"]
    assert sizes["engine"]["files"] == ["graph.bin", "residents_engine.bin"]
    assert sizes["detail"] == {"files": [], "bytes": 0}


def test_attributions_sorted_and_unique(city):
    assert manifest_of(city)["attributions"] == ["Census", "OSM"]


def test_manifest_changes_when_any_content_changes(city):
    nudged = dataclasses.replace(
        city, stores=dataclasses.replace(city.stores, radius_m=np.array([1500.0, 2001.0]))
    )
    assert encode_pack(nudged).pack_hash != encode_pack(city).pack_hash


def shifted_base(city):
    """Every position in 100..200 m, so one point can set each bbox edge."""
    n = len(city.buildings.x)
    b = dataclasses.replace(city.buildings, x=np.full(n, 150.0), y=np.full(n, 160.0))
    d = dataclasses.replace(city.detail, x=np.full(2, 150.0), y=np.full(2, 160.0))
    r = dataclasses.replace(city.roads, x=np.full(4, 150.0), y=np.full(4, 160.0))
    res = dataclasses.replace(city.residents, x=np.full(3, 150.0), y=np.full(3, 160.0))
    st = dataclasses.replace(city.stores, x=np.full(2, 150.0), y=np.full(2, 160.0))
    return dataclasses.replace(city, buildings=b, detail=d, roads=r, residents=res, stores=st)


def with_point(contents, source, x, y):
    col = getattr(contents, source)
    xs, ys = col.x.copy(), col.y.copy()
    xs[0], ys[0] = x, y
    return dataclasses.replace(contents, **{source: dataclasses.replace(col, x=xs, y=ys)})


SOURCES = ["buildings", "detail", "roads", "residents", "stores"]


@pytest.mark.parametrize("source", SOURCES)
def test_bbox_floors_minima_and_ceils_maxima_from_every_source(city, source):
    base = shifted_base(city)
    assert manifest_of(base)["bbox"] == [150, 160, 151, 161]
    low = manifest_of(with_point(base, source, -5.5, -6.5))["bbox"]
    assert low == [-6, -7, 150, 160]
    high = manifest_of(with_point(base, source, 301.25, 302.75))["bbox"]
    assert high == [150, 160, 302, 303]


def test_bbox_ignores_the_river(city):
    base = shifted_base(city)
    wide = dataclasses.replace(
        base,
        river=dataclasses.replace(base.river, x=base.river.x + 1e4, y=base.river.y - 1e4),
    )
    assert manifest_of(wide)["bbox"] == manifest_of(base)["bbox"]


def test_bbox_entries_are_whole_numbers(city):
    for v in manifest_of(shifted_base(city))["bbox"]:
        assert isinstance(v, int)


def test_bbox_adds_one_to_max_on_a_zero_width_axis(city):
    n = len(city.buildings.x)
    base = shifted_base(city)
    pinned = dataclasses.replace(
        base, buildings=dataclasses.replace(base.buildings, x=np.full(n, 150.0))
    )
    assert manifest_of(pinned)["bbox"] == [150, 160, 151, 161]
    skewed = with_point(pinned, "stores", 150.0, 170.5)
    assert manifest_of(skewed)["bbox"] == [150, 160, 151, 171]


def test_encoded_pack_is_frozen(city):
    e = encode_pack(city)
    assert isinstance(e, EncodedPack)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.pack_hash = "x"  # type: ignore[misc]


def test_write_pack_writes_objects_and_manifest_and_leaves_other_files(city, tmp_path):
    (tmp_path / "keep.txt").write_text("stay")
    e = encode_pack(city)
    write_pack(e, tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == e.manifest
    for name, obj in e.objects.items():
        assert (tmp_path / name).read_bytes() == obj
    assert (tmp_path / "keep.txt").read_text() == "stay"
    assert {p.name for p in tmp_path.iterdir()} == {*e.objects, "manifest.json", "keep.txt"}


def test_each_object_is_one_zstd_frame_with_content_size_and_no_checksum(city, toy):
    for contents in (city, toy):
        e = encode_pack(contents)
        for name, obj in e.objects.items():
            params = zstandard.get_frame_parameters(obj)
            assert params.content_size == len(e.raw[name]), name
            assert not params.has_checksum, name
            assert zstandard.frame_content_size(obj) == len(e.raw[name])
            assert zstandard.ZstdDecompressor().decompress(obj) == e.raw[name]


def test_objects_equal_direct_level_19_compression(city):
    e = encode_pack(city)
    c = zstandard.ZstdCompressor(level=19, write_checksum=False, write_content_size=True, threads=0)
    for name, raw in e.raw.items():
        assert e.objects[name] == c.compress(raw), name
