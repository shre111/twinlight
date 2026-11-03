import math

import numpy as np
import pytest
from osm_fixture import ring_area

from twl_citybake.osm.extract import OsmLayers
from twl_citybake.osm.tags import BuildingKind, HeightSource

GRID_IDS = [20_000 + n for n in range(25)] + [21_000 + n for n in range(25)]
SINGLE_IDS = [1001, 1002, 1003, 1004, 1005, 1008, 1010, 1011, 1012, 1013]
EXPECTED_WAY_IDS = sorted(GRID_IDS + SINGLE_IDS)


def row(layers: OsmLayers, osm_id: int, osm_type: int = 0) -> int:
    b = layers.buildings
    hits = np.flatnonzero((b["osm_type"] == osm_type) & (b["osm_id"] == osm_id))
    assert len(hits) == 1, f"building {osm_type}/{osm_id} not found once"
    return int(hits[0])


def test_building_columns_have_the_specified_dtypes(layers: OsmLayers) -> None:
    expected = {
        "osm_type": np.uint8,
        "osm_id": np.int64,
        "x": np.float64,
        "y": np.float64,
        "w": np.float32,
        "d": np.float32,
        "rot": np.float32,
        "area_m2": np.float32,
        "height_m": np.float32,
        "height_source": np.uint8,
        "kind": np.uint8,
        "district": np.uint16,
        "bank": np.uint8,
    }
    assert {name: layers.buildings[name].dtype.type for name in expected} == expected
    assert len({len(col) for col in layers.buildings.values()}) == 1


def test_selection_keeps_buildings_inside_and_drops_the_rest(layers: OsmLayers) -> None:
    b = layers.buildings
    ways = b["osm_id"][b["osm_type"] == 0].tolist()
    relations = b["osm_id"][b["osm_type"] == 1].tolist()

    assert sorted(ways) == EXPECTED_WAY_IDS
    assert relations == [900]
    assert len(b["osm_id"]) == 61


def test_building_no_is_not_a_building(layers: OsmLayers) -> None:
    assert 1006 not in layers.buildings["osm_id"].tolist()


def test_building_with_centroid_outside_the_clip_is_dropped(layers: OsmLayers) -> None:
    assert 1007 not in layers.buildings["osm_id"].tolist()


def test_self_intersecting_footprint_is_skipped_and_counted(layers: OsmLayers) -> None:
    assert 1009 not in layers.buildings["osm_id"].tolist()
    assert layers.report["buildings"]["skipped_invalid"] == 1


def test_multipolygon_hole_reduces_area_but_not_centroid(layers: OsmLayers) -> None:
    b = layers.buildings
    i = row(layers, 900, osm_type=1)

    assert b["area_m2"][i] == pytest.approx(100 * 100 - 40 * 40, abs=1.0)
    assert b["x"][i] == pytest.approx(350, abs=0.1)
    assert b["y"][i] == pytest.approx(-450, abs=0.1)
    assert (b["w"][i], b["d"][i]) == pytest.approx((100, 100), abs=0.5)


def test_multipolygon_ring_is_the_exterior_without_the_hole(layers: OsmLayers) -> None:
    offsets = layers.rings["offsets"]
    i = row(layers, 900, osm_type=1)
    ring = layers.rings["xy"][offsets[i] : offsets[i + 1]]

    assert ring_area(ring) == pytest.approx(100 * 100, abs=1.0)
    assert ring[:, 0].min() == pytest.approx(300, abs=0.1)
    assert ring[:, 0].max() == pytest.approx(400, abs=0.1)


def test_tagged_height_buildings_take_height_and_building_height(layers: OsmLayers) -> None:
    b = layers.buildings
    for osm_id, height in [(1001, 15.0), (1012, 8.0), (1013, 12.0)]:
        i = row(layers, osm_id)
        assert b["height_m"][i] == pytest.approx(height, abs=1e-3)
        assert b["height_source"][i] == HeightSource.HEIGHT


def test_levels_buildings_get_levels_times_three_point_two(layers: OsmLayers) -> None:
    b = layers.buildings
    i = row(layers, 1002)
    assert b["height_m"][i] == pytest.approx(12.8, abs=1e-3)
    assert b["height_source"][i] == HeightSource.LEVELS

    for osm_id, height in [(20_000, 9.6), (21_000, 19.2)]:
        j = row(layers, osm_id)
        assert b["height_m"][j] == pytest.approx(height, abs=1e-3)
        assert b["height_source"][j] == HeightSource.LEVELS


def test_untagged_buildings_use_default_source_with_positive_height(layers: OsmLayers) -> None:
    b = layers.buildings
    default_ids = [1003, 1004, 1005, 1008, 1010, 1011]
    for osm_id in default_ids:
        i = row(layers, osm_id)
        assert b["height_source"][i] == HeightSource.DEFAULT
        assert b["height_m"][i] > 0
    assert b["height_source"][row(layers, 900, osm_type=1)] == HeightSource.DEFAULT


def test_every_building_has_positive_height_and_known_source_and_kind(layers: OsmLayers) -> None:
    b = layers.buildings
    assert np.all(b["height_m"] > 0)
    assert set(b["height_source"].tolist()) <= {int(s) for s in HeightSource}
    assert set(b["kind"].tolist()) <= {int(k) for k in BuildingKind}


def test_kinds_follow_building_tags(layers: OsmLayers) -> None:
    b = layers.buildings
    expected = {
        1001: BuildingKind.COMMERCIAL,
        1002: BuildingKind.UNKNOWN,
        1004: BuildingKind.RESIDENTIAL,
        1005: BuildingKind.COMMERCIAL,
        1008: BuildingKind.RESIDENTIAL,
        20_000: BuildingKind.RESIDENTIAL,
    }
    for osm_id, kind in expected.items():
        assert b["kind"][row(layers, osm_id)] == kind
    assert b["kind"][row(layers, 900, osm_type=1)] == BuildingKind.UNKNOWN


def test_box_sides_are_ordered_and_rotation_is_in_half_turn(layers: OsmLayers) -> None:
    b = layers.buildings
    assert np.all(b["w"] >= b["d"])
    assert np.all(b["d"] > 0)
    assert np.all(b["rot"] >= 0)
    assert np.all(b["rot"] < math.pi)


def test_box_covers_the_footprint_area(layers: OsmLayers) -> None:
    b = layers.buildings
    box = b["w"].astype(np.float64) * b["d"]
    assert np.all(b["area_m2"] > 0)
    assert np.all(box >= b["area_m2"] * (1 - 1e-4))


def test_axis_aligned_square_has_its_exact_geometry(layers: OsmLayers) -> None:
    b = layers.buildings
    i = row(layers, 1002)

    assert (b["x"][i], b["y"][i]) == pytest.approx((-500, 300), abs=0.05)
    assert (b["w"][i], b["d"][i]) == pytest.approx((20, 20), abs=0.1)
    assert b["area_m2"][i] == pytest.approx(400, abs=1.0)


def test_rotated_rectangle_reports_long_side_angle_from_east(layers: OsmLayers) -> None:
    b = layers.buildings
    i = row(layers, 1008)

    assert (b["w"][i], b["d"][i]) == pytest.approx((30, 12), abs=0.1)
    assert b["rot"][i] == pytest.approx(math.radians(120), abs=0.01)
    assert b["area_m2"][i] == pytest.approx(360, abs=1.0)
    assert (b["x"][i], b["y"][i]) == pytest.approx((200, -300), abs=0.05)


def test_buildings_are_sorted_by_osm_type_then_id(layers: OsmLayers) -> None:
    b = layers.buildings
    keys = list(zip(b["osm_type"].tolist(), b["osm_id"].tolist(), strict=True))
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_ring_offsets_are_monotone_and_cover_all_vertices(layers: OsmLayers) -> None:
    offsets = layers.rings["offsets"]
    xy = layers.rings["xy"]

    assert offsets.dtype == np.uint32
    assert xy.dtype == np.float32
    assert xy.ndim == 2 and xy.shape[1] == 2
    assert len(offsets) == len(layers.buildings["osm_id"]) + 1
    assert offsets[0] == 0
    assert np.all(np.diff(offsets.astype(np.int64)) >= 3)
    assert offsets[-1] == len(xy)


def test_each_ring_matches_its_building_footprint(layers: OsmLayers) -> None:
    b = layers.buildings
    offsets = layers.rings["offsets"]
    xy = layers.rings["xy"].astype(np.float64)
    for i in np.flatnonzero(b["osm_type"] == 0)[::7]:
        ring = xy[offsets[i] : offsets[i + 1]]
        assert ring_area(ring) == pytest.approx(float(b["area_m2"][i]), rel=0.01)
        assert ring[:, 0].min() <= b["x"][i] <= ring[:, 0].max()
        assert ring[:, 1].min() <= b["y"][i] <= ring[:, 1].max()
