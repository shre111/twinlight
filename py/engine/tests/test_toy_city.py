from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city

PROTO_BUILDINGS = 7253
PROTO_TIES_PER_RESIDENT = 4.5
DISTRICT_NAMES = (
    "Ellisbridge",
    "Paldi",
    "Navrangpura",
    "Naranpura",
    "Chandkheda",
    "Ghatlodia",
    "Vastrapur",
    "Bodakdev",
    "Thaltej",
    "Satellite",
    "Shahibaug",
    "Maninagar",
)
SMALL = 5_000
GROUPS = ("buildings", "roads", "river", "residents", "ties", "stores")


def field_names(obj: object) -> list[str]:
    if dataclasses.is_dataclass(obj):
        return [f.name for f in dataclasses.fields(obj)]
    fields = getattr(obj, "_fields", None)
    if fields is not None:
        return list(fields)
    return list(vars(obj))


def city_arrays(city: ToyCity) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for group in GROUPS:
        obj = getattr(city, group)
        for name in field_names(obj):
            out[f"{group}.{name}"] = np.asarray(getattr(obj, name))
    return out


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=SMALL)


@pytest.fixture(scope="module")
def full_city() -> ToyCity:
    return build_toy_city()


def test_same_seed_gives_identical_arrays_and_dtypes(city: ToyCity) -> None:
    a, b = city_arrays(city), city_arrays(build_toy_city(seed=49, residents=SMALL))
    assert a.keys() == b.keys()
    assert len(a) >= 31
    for key in a:
        assert a[key].dtype == b[key].dtype, key
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_different_seed_gives_different_city(city: ToyCity) -> None:
    other = build_toy_city(seed=50, residents=SMALL)
    assert other.seed == 50
    assert city.seed == 49
    assert not np.array_equal(city.residents.x, other.residents.x)
    assert not np.array_equal(city.ties.targets, other.ties.targets)


def test_array_dtypes_match_the_pack_spec(city: ToyCity) -> None:
    expected = {
        "buildings.x": np.float32,
        "buildings.y": np.float32,
        "buildings.width": np.float32,
        "buildings.depth": np.float32,
        "buildings.height": np.float32,
        "buildings.rotation": np.float32,
        "buildings.district": np.uint16,
        "roads.offsets": np.uint32,
        "roads.x": np.float32,
        "roads.y": np.float32,
        "roads.kind": np.uint8,
        "river.offsets": np.uint32,
        "river.x": np.float32,
        "river.y": np.float32,
        "river.kind": np.uint8,
        "residents.x": np.float32,
        "residents.y": np.float32,
        "residents.z": np.float32,
        "residents.building": np.uint32,
        "residents.hex": np.uint32,
        "residents.segment": np.uint16,
        "residents.persona_ref": np.uint32,
        "residents.weight": np.float32,
        "ties.offsets": np.uint32,
        "ties.targets": np.uint32,
        "ties.kind": np.uint8,
        "ties.strength": np.float32,
        "stores.x": np.float32,
        "stores.y": np.float32,
        "stores.hex": np.uint32,
        "stores.radius_m": np.float32,
        "stores.district": np.uint16,
    }
    arrays = city_arrays(city)
    for key, dtype in expected.items():
        assert arrays[key].dtype == np.dtype(dtype), key
        assert arrays[key].ndim == 1, key


def test_full_size_counts_are_close_to_the_prototype(full_city: ToyCity) -> None:
    n = len(full_city.residents.x)
    assert n == 48_000
    assert len(full_city.buildings.x) == pytest.approx(PROTO_BUILDINGS, rel=0.03)
    ties_per_resident = len(full_city.ties.targets) / n
    assert ties_per_resident == pytest.approx(PROTO_TIES_PER_RESIDENT, rel=0.03)
    assert len(full_city.stores.x) == 14


def test_default_seed_is_49(full_city: ToyCity) -> None:
    assert full_city.seed == 49


def test_column_lengths_are_consistent(city: ToyCity) -> None:
    nb, nr = len(city.buildings.x), len(city.residents.x)
    assert nr == SMALL
    for name in ("y", "width", "depth", "height", "rotation", "district"):
        assert len(getattr(city.buildings, name)) == nb
    for name in ("y", "z", "building", "hex", "segment", "persona_ref", "weight"):
        assert len(getattr(city.residents, name)) == nr
    m = len(city.ties.targets)
    assert len(city.ties.kind) == m
    assert len(city.ties.strength) == m
    ns = len(city.stores.x)
    for name in ("y", "hex", "radius_m", "district"):
        assert len(getattr(city.stores, name)) == ns


def test_tie_graph_is_valid_csr_without_self_ties(city: ToyCity) -> None:
    t, n = city.ties, SMALL
    assert len(t.offsets) == n + 1
    assert t.offsets[0] == 0
    assert np.all(np.diff(t.offsets.astype(np.int64)) >= 0)
    assert t.offsets[-1] == len(t.targets)
    assert len(t.targets) > 0
    assert int(t.targets.max()) < n
    degrees = np.diff(t.offsets.astype(np.int64))
    owners = np.repeat(np.arange(n, dtype=np.int64), degrees)
    assert not np.any(owners == t.targets.astype(np.int64))


def test_tie_kinds_and_strengths_use_the_toy_values(city: ToyCity) -> None:
    assert set(np.unique(city.ties.kind).tolist()) == {1, 2}
    assert np.all(city.ties.strength == 1.0)


def test_each_resident_has_at_most_six_ties_and_almost_all_have_three(city: ToyCity) -> None:
    degrees = np.diff(city.ties.offsets.astype(np.int64))
    assert degrees.min() >= 1
    assert degrees.max() <= 6
    assert np.mean(degrees >= 3) > 0.995


def test_one_tie_in_ten_is_long_range(full_city: ToyCity) -> None:
    share_random = float(np.mean(full_city.ties.kind == 2))
    assert share_random == pytest.approx(0.1, abs=0.01)


def test_toy_resident_constants(city: ToyCity) -> None:
    r = city.residents
    assert np.all(r.hex == 0)
    assert np.all(r.persona_ref == 0xFFFFFFFF)
    assert np.all(r.weight == 1.0)
    assert set(np.unique(r.segment).tolist()) == {0, 1, 2, 3, 4}


def test_resident_building_indexes_are_valid_and_heights_below_roof(city: ToyCity) -> None:
    r, b = city.residents, city.buildings
    assert int(r.building.max()) < len(b.x)
    assert np.all(r.z >= 2.4 - 1e-4)
    assert np.all(r.z <= b.height[r.building] + 1e-4)


def test_resident_floors_follow_the_2_4_plus_3_3_ladder(city: ToyCity) -> None:
    z = city.residents.z.astype(np.float64)
    roof = city.buildings.height[city.residents.building].astype(np.float64)
    k = (z - 2.4) / 3.3
    on_ladder = np.abs(k - np.round(k)) < 1e-3
    capped = np.abs(z - (roof - 1.2)) < 1e-3
    assert np.all(on_ladder | capped)


def test_residents_stand_just_outside_their_building_in_pack_axes(city: ToyCity) -> None:
    r, b = city.residents, city.buildings
    i = r.building
    dx = r.x.astype(np.float64) - b.x[i]
    dy = r.y.astype(np.float64) - b.y[i]
    rot = b.rotation[i].astype(np.float64)
    c, s = np.cos(rot), np.sin(rot)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    half_w = b.width[i] / 2
    half_d = b.depth[i] / 2
    assert np.all(np.abs(lx) <= half_w + 1.0)
    assert np.all(np.abs(ly) <= half_d + 1.0)
    outside = (np.abs(lx) >= half_w + 0.7) | (np.abs(ly) >= half_d + 0.7)
    assert np.all(outside)


def test_rotations_are_varied_radians(city: ToyCity) -> None:
    b = city.buildings
    assert np.abs(b.rotation).max() < np.pi + 1e-3
    assert np.std(b.rotation) > 0.01


def test_districts_are_the_prototype_twelve(city: ToyCity) -> None:
    assert len(city.districts) == 12
    assert tuple(d.name for d in city.districts) == DISTRICT_NAMES
    assert tuple(d.id for d in city.districts) == tuple(range(12))
    assert int(city.buildings.district.max()) < 12
    assert int(city.stores.district.max()) < 12
    assert len(np.unique(city.buildings.district)) == 12


def test_district_axes_follow_x_east_y_north(city: ToyCity) -> None:
    by_name = {d.name: d for d in city.districts}
    assert by_name["Chandkheda"].y == pytest.approx(2650)
    assert by_name["Chandkheda"].x == pytest.approx(-250)
    assert by_name["Maninagar"].y == pytest.approx(-1350)
    assert by_name["Maninagar"].x == pytest.approx(1850)
    assert by_name["Maninagar"].east is True
    assert by_name["Ellisbridge"].east is False
    assert by_name["Shahibaug"].east is True


def test_buildings_lie_north_and_south_of_the_centre_by_district(city: ToyCity) -> None:
    ids = {d.name: d.id for d in city.districts}
    north = city.buildings.district == ids["Chandkheda"]
    south = city.buildings.district == ids["Maninagar"]
    assert float(np.mean(city.buildings.y[north])) > 1500
    assert float(np.mean(city.buildings.y[south])) < 0


def test_river_polygons_span_the_city_north_to_south(city: ToyCity) -> None:
    rv = city.river
    assert len(rv.offsets) >= 2
    assert rv.offsets[0] == 0
    assert np.all(np.diff(rv.offsets.astype(np.int64)) > 2)
    assert rv.offsets[-1] == len(rv.x) == len(rv.y)
    assert len(rv.kind) == len(rv.offsets) - 1
    assert np.all(rv.kind == 0)
    assert float(rv.y.max()) - float(rv.y.min()) > 5000
    assert float(rv.y.max()) > 0 > float(rv.y.min())


def test_roads_offsets_are_valid_and_cover_all_three_kinds(city: ToyCity) -> None:
    rd = city.roads
    n = len(rd.kind)
    assert len(rd.offsets) == n + 1
    assert rd.offsets[0] == 0
    assert np.all(np.diff(rd.offsets.astype(np.int64)) >= 2)
    assert rd.offsets[-1] == len(rd.x) == len(rd.y)
    assert set(np.unique(rd.kind).tolist()) == {0, 1, 2}


def test_streets_outnumber_arterials(city: ToyCity) -> None:
    kinds = city.roads.kind
    assert int(np.sum(kinds == 0)) > int(np.sum(kinds == 1))


def test_stores_are_sorted_by_x_with_toy_values(city: ToyCity) -> None:
    s = city.stores
    assert len(s.x) == 14
    assert np.all(np.diff(s.x) >= 0)
    assert np.all(s.radius_m == 900.0)
    assert np.all(s.hex == 0)


def test_stores_are_near_residents_and_spread_over_the_city(city: ToyCity) -> None:
    s, r = city.stores, city.residents
    rx, ry = r.x.astype(np.float64), r.y.astype(np.float64)
    for k in range(len(s.x)):
        assert np.hypot(rx - s.x[k], ry - s.y[k]).min() < 400
    assert float(s.x.max() - s.x.min()) > 3000
    assert float(s.y.max() - s.y.min()) > 3000


def test_building_geometry_is_plausible(city: ToyCity) -> None:
    b = city.buildings
    assert np.all(b.height >= 6 - 1e-4)
    assert np.all(b.height <= 120 + 1e-4)
    assert np.all(b.width > 0)
    assert np.all(b.depth > 0)
    assert np.all(np.abs(b.x) < 5000)
    assert np.all(np.abs(b.y) < 5000)
