import dataclasses
import json

import numpy as np
import pytest

from twl_pack.writer import encode_pack


def pad(b: bytes) -> bytes:
    return b + bytes(-len(b) % 8)


def col(values, dtype) -> bytes:
    return pad(np.asarray(values, dtype=dtype).tobytes())


def planes(values, width) -> bytes:
    a = np.asarray(values, dtype=f"<u{width}")
    return pad(a.view(np.uint8).reshape(len(a), width).T.tobytes())


def expected_files() -> dict[str, bytes]:
    # Pack order of the six buildings is input rows [5, 2, 4, 3, 0, 1]; Morton codes
    # 0, 1, 1, 2, 0x55555555, 0xAAAAAAAA give deltas 0, 1, 0, 1, 0x55555553, 0x55555555.
    # Tile level 1 bins by code >> 30: tiles 0,0,0,0,1,2.
    buildings = (
        col([0, 4, 5, 6, 6], "<u4")
        + planes([0, 1, 0, 1, 0x55555553, 0x55555555], 4)
        + col([120, 48, 72, 60, 24, 36], "u1")
        + col([24, 60, 36, 48, 120, 72], "u1")
        + col([93, 66, 85, 76, 38, 54], "u1")
        + col([191, 85, 0, 170, 0, 64], "u1")
        + col([0, 0, 1, 1, 0, 1], "u1")
    )
    # Detail rows (5,5) and (2,2) have codes 51 and 12, so they swap: deltas 12, 39.
    detail = (
        planes([12, 39], 4)
        + col([36, 24], "u1")
        + col([36, 24], "u1")
        + col([36, 24], "u1")
        + col([0, 0], "u1")
        + col([0, 1], "u1")
    )
    # Road x column 65535, 0, 40000, 7 -> deltas wrap to 1, 2, 51071 (0xC77F), 51086 (0xC78E).
    roads = (
        col([3, 1], "<u2")
        + col([0, 2], "u1")
        + pad(bytes([0x01, 0x02, 0x7F, 0x8E, 0x00, 0x00, 0xC7, 0xC7]))
        + pad(bytes([0, 0, 0, 18, 0, 0, 0, 0]))
    )
    residents = (
        col([10, 65535, 0], "<u2")
        + col([20, 0, 65535], "<u2")
        + col([0, 3, 255], "u1")
        + col([4, 0, 1], "<u4")
        + col([0, 1, 1], "u1")
    )
    engine = (
        col([1, 0, 1], "<u4")
        + col([5, 0xFFFFFFFF, 7], "<u4")
        + col([1.0, 0.5, 2.0], "<f4")
        + col([2, 0, 1], "<u1")
    )
    stores = (
        col([100.5, 300.0], "<f4")
        + col([200.25, 400.0], "<f4")
        + col([0, 1], "<u4")
        + col([1500.0, 2000.5], "<f4")
        + col([0, 1], "<u2")
    )
    river = (
        col([0, 4, 8, 11], "<u4")
        + col([0, 10, 10, 0, 2, 2, 8, 8, 20, 30, 25], "<f4")
        + col([0, 0, 10, 10, 2, 8, 8, 2, 20, 20, 30], "<f4")
        + col([0, 1], "u1")
        + col([2, 1], "<u2")
    )
    graph = (
        col([0, 2, 3, 3], "<u4")
        + col([1, 2, 0], "<u4")
        + col([0, 2, 3], "u1")
        + col([0.5, 1.0, 0.25], "<f4")
    )
    return {
        "buildings.bin": buildings,
        "buildings_detail.bin": detail,
        "roads.bin": roads,
        "residents.bin": residents,
        "residents_engine.bin": engine,
        "stores.bin": stores,
        "river.bin": river,
        "graph.bin": graph,
        "hexes.bin": col([0x8828308281FFFFF, 0x8828308283FFFFF], "<u8"),
        "travel.bin": col([5, 6, 7, 8, 9, 10, 11, 12], "<u2"),
    }


@pytest.fixture
def encoded(city):
    return encode_pack(city)


@pytest.mark.parametrize("name", sorted(expected_files()))
def test_uncompressed_bytes_match_hand_layout(encoded, name):
    assert encoded.raw[name] == expected_files()[name]


def test_object_set_is_exactly_the_listed_files(encoded):
    assert set(encoded.raw) == {*expected_files(), "districts.json"}
    assert set(encoded.objects) == set(encoded.raw)


def test_districts_json_passes_dicts_through_compact_and_sorted(encoded, city):
    want = json.dumps(
        city.districts, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert encoded.raw["districts.json"] == want
    assert json.loads(want) == city.districts


def test_building_order_is_stable_on_morton_ties(encoded):
    assert encoded.building_order.dtype == np.int64
    assert encoded.building_order.tolist() == [5, 2, 4, 3, 0, 1]
    assert encoded.detail_order.dtype == np.int64
    assert encoded.detail_order.tolist() == [1, 0]


def test_residents_building_is_remapped_to_pack_order(encoded):
    raw = encoded.raw["residents.bin"]
    start = 8 + 8 + 8  # x and y (3 u16 -> 8), floor (3 u8 -> 8)
    building = np.frombuffer(raw, dtype="<u4", count=3, offset=start)
    assert building.tolist() == [4, 0, 1]


def test_tile_start_is_first_column_of_buildings_and_absent_from_detail(encoded):
    tile_start = np.frombuffer(encoded.raw["buildings.bin"], dtype="<u4", count=5)
    assert tile_start.tolist() == [0, 4, 5, 6, 6]
    assert len(encoded.raw["buildings.bin"]) == 24 + 24 + 5 * 8
    assert len(encoded.raw["buildings_detail.bin"]) == 8 + 5 * 8


@pytest.mark.parametrize("name", sorted(expected_files()))
def test_every_column_ends_on_an_eight_byte_boundary(encoded, name):
    assert len(encoded.raw[name]) % 8 == 0


def test_last_column_padding_is_zero(encoded):
    assert encoded.raw["buildings.bin"][-8:] == bytes([0, 0, 1, 1, 0, 1, 0, 0])
    assert encoded.raw["stores.bin"][-8:] == bytes([0, 0, 1, 0, 0, 0, 0, 0])


def test_detail_none_and_zero_rows_both_omit_the_file(city):
    none = encode_pack(dataclasses.replace(city, detail=None))
    empty_detail = dataclasses.replace(
        city.detail, **{f.name: np.zeros(0) for f in dataclasses.fields(city.detail)}
    )
    empty = encode_pack(dataclasses.replace(city, detail=empty_detail))
    for e in (none, empty):
        assert "buildings_detail.bin" not in e.raw
        assert "buildings_detail.bin" not in e.objects
        assert e.detail_order.size == 0
        assert json.loads(e.manifest)["counts"]["detailBuildings"] == 0


def test_toy_pack_has_no_cell_or_detail_files(toy):
    out = encode_pack(toy)
    assert "hexes.bin" not in out.raw
    assert "travel.bin" not in out.raw
    assert "buildings_detail.bin" not in out.raw
    assert "graph.bin" in out.raw and "residents_engine.bin" in out.raw


def test_clipped_width_is_counted_and_stored_as_255(city):
    wide = dataclasses.replace(
        city.buildings, width=np.array([500.0, 9, 16, 25, 36, 100]), height=np.full(6, 600.0)
    )
    out = encode_pack(dataclasses.replace(city, buildings=wide))
    assert out.clipped == {"width": 1, "depth": 0, "height": 6}
    # input row 0 sorts to pack row 4; its width plane starts after tile_start and deltas
    width = np.frombuffer(out.raw["buildings.bin"], dtype="u1", count=6, offset=24 + 24)
    assert width[4] == 255


def test_clipped_sums_buildings_and_detail(city):
    detail = dataclasses.replace(city.detail, depth=np.array([500.0, 500.0]))
    out = encode_pack(dataclasses.replace(city, detail=detail))
    assert out.clipped == {"width": 0, "depth": 2, "height": 0}


def test_input_order_of_buildings_does_not_change_the_file(city):
    perm = np.array([3, 1, 5, 0, 2, 4])
    b = city.buildings
    shuffled = dataclasses.replace(
        city,
        buildings=dataclasses.replace(
            b,
            x=b.x[perm],
            y=b.y[perm],
            width=b.width[perm],
            depth=b.depth[perm],
            height=b.height[perm],
            rotation=b.rotation[perm],
            district=b.district[perm],
        ),
        residents=dataclasses.replace(
            city.residents, building=np.argsort(perm)[city.residents.building]
        ),
    )
    base = encode_pack(city)
    other = encode_pack(shuffled)
    # this permutation keeps rows 2 and 4, which tie on Morton code, in the same relative order
    assert other.raw == base.raw
    assert other.pack_hash == base.pack_hash
