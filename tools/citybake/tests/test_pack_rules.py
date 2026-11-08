import json
from typing import Any

import numpy as np
import pytest
import shapely
from pack_shapes import closed, square, square_ring, water_table_of

from twl_citybake.osm.tags import RoadClass
from twl_citybake.pack import rules
from twl_citybake.pack.rules import PackStageError

ARTERIAL = (RoadClass.MAJOR, RoadClass.PRIMARY, RoadClass.SECONDARY)


def polygons_of(*shapes: shapely.Polygon) -> np.ndarray:
    return np.array(shapes, dtype=object)


def mask(polygons: np.ndarray, area: list[float], min_area: float = 30.0, within: float = 3.0):
    return rules.isolated_small(polygons, np.array(area, np.float32), min_area, within).tolist()


def test_footprint_polygons_rebuild_one_polygon_per_ring() -> None:
    ring_a, ring_b = square_ring(0, 0, 5), square_ring(100, 0, 10)
    offsets = np.array([0, len(ring_a), len(ring_a) + len(ring_b)], np.uint32)
    xy = np.array([*ring_a, *ring_b], np.float32)
    polygons = rules.footprint_polygons(offsets, xy)
    assert [shapely.area(p) for p in polygons] == pytest.approx([25.0, 100.0])


def test_a_small_footprint_alone_is_isolated() -> None:
    assert mask(polygons_of(square(0, 0, 5)), [25.0]) == [True]


@pytest.mark.parametrize(
    ("gap", "isolated"),
    [(2.9, False), (3.0, False), (3.1, True), (0.0, False)],
    ids=["2.9m", "exactly-3m", "3.1m", "touching"],
)
def test_a_small_footprint_is_isolated_only_when_its_nearest_neighbour_is_beyond_the_gap(
    gap: float, isolated: bool
) -> None:
    large = square(2.5 + gap + 5.0, 0, 10)
    polygons = polygons_of(square(0, 0, 5), large)
    assert mask(polygons, [25.0, 100.0]) == [isolated, False]


@pytest.mark.parametrize(("gap", "isolated"), [(2.9, False), (3.0, False), (3.1, True)])
def test_a_small_neighbour_counts_as_a_neighbour(gap: float, isolated: bool) -> None:
    polygons = polygons_of(square(0, 0, 5), square(5.0 + gap, 0, 5))
    assert mask(polygons, [25.0, 25.0]) == [isolated, isolated]


def test_two_small_footprints_close_to_each_other_are_both_kept() -> None:
    polygons = polygons_of(square(0, 0, 4), square(6.0, 0, 4))
    assert mask(polygons, [16.0, 16.0]) == [False, False]


def test_a_neighbour_in_the_other_direction_counts_too() -> None:
    polygons = polygons_of(square(0, 0, 5), square(0, 7.9, 5))
    assert mask(polygons, [25.0, 25.0]) == [False, False]


@pytest.mark.parametrize(
    ("area", "isolated"), [(29.999, True), (30.0, False), (30.001, False), (5000.0, False)]
)
def test_a_footprint_at_or_over_min_area_is_never_isolated_however_alone(
    area: float, isolated: bool
) -> None:
    assert mask(polygons_of(square(0, 0, 5)), [area]) == [isolated]


def test_a_footprint_is_not_its_own_neighbour() -> None:
    polygons = polygons_of(square(0, 0, 5), square(500, 500, 5))
    assert mask(polygons, [25.0, 25.0]) == [True, True]


def test_the_mask_follows_input_order_and_leaves_large_footprints_unflagged() -> None:
    polygons = polygons_of(
        square(0, 0, 20),
        square(100, 0, 5),
        square(300, 0, 5),
        square(305.5, 0, 5),
        square(600, 0, 12),
    )
    area = [400.0, 25.0, 25.0, 25.0, 144.0]
    assert mask(polygons, area) == [False, True, False, False, False]


def test_min_area_and_gap_are_parameters() -> None:
    polygons = polygons_of(square(0, 0, 5), square(15.0, 0, 5))
    assert mask(polygons, [25.0, 25.0], min_area=20.0, within=3.0) == [False, False]
    assert mask(polygons, [25.0, 25.0], min_area=30.0, within=3.0) == [True, True]
    assert mask(polygons, [25.0, 25.0], min_area=30.0, within=10.0) == [False, False]


LAKE = [(0, 0), (100, 0), (100, 100), (0, 100)]
HOLE = [(40, 40), (40, 60), (60, 60), (60, 40)]
ISLAND_POND = [(200, 0), (300, 0), (300, 100), (200, 100)]


def water_geometry() -> shapely.Geometry:
    table = water_table_of([(1, [closed(LAKE), closed(HOLE)]), (0, [closed(ISLAND_POND)])])
    return rules.water_geometry(table)


def test_a_point_in_a_lake_hole_is_not_in_water() -> None:
    assert rules.in_water(water_geometry(), np.array([50.0]), np.array([50.0])).tolist() == [False]


def test_a_point_in_the_lake_ring_around_the_hole_is_in_water() -> None:
    assert rules.in_water(water_geometry(), np.array([10.0]), np.array([50.0])).tolist() == [True]


def test_a_point_in_a_second_polygon_is_in_water() -> None:
    assert rules.in_water(water_geometry(), np.array([250.0]), np.array([50.0])).tolist() == [True]


def test_a_point_outside_every_polygon_is_not_in_water() -> None:
    x, y = np.array([150.0, -5.0, 250.0]), np.array([50.0, 50.0, 500.0])
    assert rules.in_water(water_geometry(), x, y).tolist() == [False, False, False]


def test_in_water_returns_one_flag_per_point_in_order() -> None:
    x = np.array([10.0, 50.0, 250.0, 150.0])
    y = np.array([10.0, 50.0, 50.0, 50.0])
    flags = rules.in_water(water_geometry(), x, y)
    assert flags.dtype == np.bool_
    assert flags.tolist() == [True, False, True, False]


def test_water_geometry_area_is_exterior_minus_hole_plus_other_polygons() -> None:
    assert shapely.area(water_geometry()) == pytest.approx(100 * 100 - 20 * 20 + 100 * 100)


@pytest.mark.parametrize(
    ("z", "floor"), [(1.6, 0), (4.8, 1), (8.0, 2), (24.0, 7), (1.605, 0), (1.595, 0)]
)
def test_floors_from_z_inverts_the_mid_floor_height(z: float, floor: int) -> None:
    assert rules.floors_from_z(np.array([z])).tolist() == [floor]


def test_floors_from_z_returns_uint8() -> None:
    assert rules.floors_from_z(np.array([1.6, 4.8])).dtype == np.uint8


def test_floors_from_z_rounds_an_f32_mid_floor_height_correctly() -> None:
    z = np.array([(f + 0.5) * 3.2 for f in range(0, 40)], np.float32)
    assert rules.floors_from_z(z).tolist() == list(range(40))


@pytest.mark.parametrize("z", [3.2, 0.0, 1.62, 1.58, 4.0])
def test_floors_from_z_raises_for_a_z_between_floors(z: float) -> None:
    with pytest.raises(PackStageError):
        rules.floors_from_z(np.array([z]))


def test_floors_from_z_raises_when_any_one_z_is_between_floors() -> None:
    with pytest.raises(PackStageError):
        rules.floors_from_z(np.array([1.6, 4.8, 3.2]))


@pytest.mark.parametrize(
    ("height", "floor"),
    [(3.0, 0), (4.8, 1), (4.79, 0), (1.0, 0), (0.0, 0), (6.4, 1), (8.0, 2), (9.6, 2), (12.8, 3)],
)
def test_top_floor_is_the_highest_floor_whose_mid_height_fits(height: float, floor: int) -> None:
    assert rules.top_floor(np.array([height])).tolist() == [floor]


def test_top_floor_works_on_an_f32_height_array() -> None:
    assert rules.top_floor(np.array([6.4, 9.6, 3.2], np.float32)).tolist() == [1, 2, 0]


BX = np.array([0.0, 10.0, 100.0, 40.0, 14.0])
BY = np.array([0.0, 0.0, 100.0, 0.0, 2.0])
BH = np.array([9.6, 6.4, 3.2, 12.8, 3.2], np.float32)
BKEPT = np.array([True, True, False, True, False])


def rehome(rows: list[tuple[int, float, float, int, bool]], kept: np.ndarray = BKEPT) -> Any:
    return rules.rehome_residents(
        np.array([r[0] for r in rows], np.uint32),
        np.array([r[3] for r in rows], np.uint8),
        np.array([r[1] for r in rows], np.float64),
        np.array([r[2] for r in rows], np.float64),
        np.array([r[4] for r in rows], np.bool_),
        kept,
        BX,
        BY,
        BH,
    )


def test_a_kept_home_outside_water_changes_nothing() -> None:
    out = rehome([(0, 3.0, 4.0, 2, False)])
    assert (out.building.tolist(), out.x.tolist(), out.y.tolist(), out.floor.tolist()) == (
        [0],
        [3.0],
        [4.0],
        [2],
    )
    assert (out.repointed, out.moved, out.floors_lowered) == (0, 0, 0)
    assert len(out.repoint_distance_m) == 0


def test_a_dropped_home_outside_water_repoints_to_the_kept_building_nearest_the_home_centre() -> (
    None
):
    # the resident at (15, 5) is nearest building 1, the home at (100, 100) is nearest building 3
    out = rehome([(2, 15.0, 5.0, 1, False)])
    assert out.building.tolist() == [3]
    assert (out.x.tolist(), out.y.tolist(), out.floor.tolist()) == ([15.0], [5.0], [1])
    assert (out.repointed, out.moved, out.floors_lowered) == (1, 0, 0)
    assert out.repoint_distance_m.tolist() == pytest.approx([np.hypot(60.0, 100.0)])


def test_a_resident_in_water_with_a_kept_home_moves_to_the_home_centre() -> None:
    out = rehome([(1, 50.0, 50.0, 1, True)])
    assert out.building.tolist() == [1]
    assert (out.x.tolist(), out.y.tolist(), out.floor.tolist()) == ([10.0], [0.0], [1])
    assert (out.repointed, out.moved, out.floors_lowered) == (0, 1, 0)
    assert len(out.repoint_distance_m) == 0


def test_a_resident_in_water_with_a_dropped_home_moves_to_the_nearest_kept_centre() -> None:
    out = rehome([(2, 50.0, 50.0, 0, True)])
    assert out.building.tolist() == [3]
    assert (out.x.tolist(), out.y.tolist(), out.floor.tolist()) == ([40.0], [0.0], [0])
    assert (out.repointed, out.moved, out.floors_lowered) == (1, 1, 0)


def test_a_floor_that_fits_the_building_is_kept_in_water() -> None:
    # building 0 is 9.6 m, so floors 0 to 2 fit
    out = rehome([(0, 50.0, 50.0, 2, True)])
    assert (out.floor.tolist(), out.floors_lowered) == ([2], 0)


def test_a_floor_too_high_for_the_building_is_lowered_to_the_highest_that_fits() -> None:
    # building 1 is 6.4 m, so only floors 0 and 1 fit
    out = rehome([(1, 50.0, 50.0, 3, True)])
    assert (out.floor.tolist(), out.floors_lowered) == ([1], 1)


def test_the_floor_limit_comes_from_the_building_moved_to_not_the_dropped_home() -> None:
    # home 2 is 3.2 m; the kept building 3 is 12.8 m, so floor 3 fits and floor 5 becomes 3
    out = rehome([(2, 50.0, 50.0, 5, True), (2, 50.0, 50.0, 3, True)])
    assert out.floor.tolist() == [3, 3]
    assert out.floors_lowered == 1


def test_a_resident_not_in_water_repointed_to_a_lower_building_keeps_their_floor() -> None:
    # building 4 is dropped; its nearest kept building 1 has only floors 0 and 1
    out = rehome([(4, 0.0, 9.0, 3, False)])
    assert out.building.tolist() == [1]
    assert (out.floor.tolist(), out.floors_lowered) == ([3], 0)
    assert (out.x.tolist(), out.y.tolist()) == ([0.0], [9.0])


def test_counts_distances_and_order_over_a_mixed_set_of_residents() -> None:
    rows = [
        (0, 3.0, 4.0, 2, False),
        (2, 15.0, 5.0, 1, False),
        (1, 50.0, 50.0, 3, True),
        (2, 50.0, 50.0, 5, True),
        (4, 0.0, 9.0, 3, False),
        (0, 1.0, 1.0, 2, True),
    ]
    out = rehome(rows)
    assert out.building.tolist() == [0, 3, 1, 3, 1, 0]
    assert out.x.tolist() == [3.0, 15.0, 10.0, 40.0, 0.0, 0.0]
    assert out.y.tolist() == [4.0, 5.0, 0.0, 0.0, 9.0, 0.0]
    assert out.floor.tolist() == [2, 1, 1, 3, 3, 2]
    assert (out.repointed, out.moved, out.floors_lowered) == (3, 3, 2)
    assert out.repoint_distance_m.dtype == np.float64
    assert out.repoint_distance_m.tolist() == pytest.approx(
        [np.hypot(60.0, 100.0), np.hypot(60.0, 100.0), np.hypot(4.0, 2.0)]
    )


def test_every_resident_is_kept_in_the_same_order_and_ends_in_a_kept_building() -> None:
    rows = [(i % 5, float(i), float(-i), i % 3, i % 2 == 0) for i in range(40)]
    out = rehome(rows)
    for column in (out.building, out.x, out.y, out.floor):
        assert len(column) == len(rows)
    assert BKEPT[out.building].all()


def test_a_kept_home_never_changes_building_even_in_water() -> None:
    rows = [(h, 5.0, 5.0, 0, w) for h in (0, 1, 3) for w in (True, False)]
    assert rehome(rows).building.tolist() == [h for h, *_ in rows]


def test_rehoming_with_no_kept_building_raises() -> None:
    with pytest.raises(PackStageError):
        rehome([(0, 0.0, 0.0, 0, False)], kept=np.zeros(5, np.bool_))


def test_rehomed_is_frozen() -> None:
    out = rehome([(0, 3.0, 4.0, 2, False)])
    with pytest.raises(AttributeError):
        out.moved = 3


def test_a_bridge_is_class_two_whatever_the_road_class() -> None:
    road_class = np.arange(6, dtype=np.uint8)
    out = rules.pack_road_class(road_class, np.ones(6, np.uint8), ARTERIAL)
    assert out.tolist() == [2] * 6


def test_arterial_classes_are_one_and_the_rest_zero() -> None:
    road_class = np.array([0, 1, 2, 3, 4, 5, 0, 4], np.uint8)
    bridge = np.array([0, 0, 0, 0, 0, 0, 1, 1], np.uint8)
    out = rules.pack_road_class(road_class, bridge, ARTERIAL)
    assert out.tolist() == [1, 1, 1, 0, 0, 0, 2, 2]
    assert out.dtype == np.uint8


def test_the_arterial_set_is_a_parameter() -> None:
    road_class = np.array([0, 1, 2, 3], np.uint8)
    out = rules.pack_road_class(road_class, np.zeros(4, np.uint8), (RoadClass.TERTIARY,))
    assert out.tolist() == [0, 0, 0, 1]


def test_river_columns_strip_the_closing_point_of_each_ring_and_rebuild_offsets() -> None:
    river = closed([(0, 0), (10, 0), (10, 10), (0, 10)])
    lake = closed([(100, 100), (200, 100), (200, 200), (100, 200)])
    hole = closed([(120, 120), (120, 180), (180, 180), (180, 120)])
    table = water_table_of([(0, [river]), (1, [lake, hole])])
    out = rules.river_columns(table)
    assert out.offsets.tolist() == [0, 4, 8, 12]
    assert out.rings.tolist() == [1, 2]
    assert out.kind.tolist() == [0, 1]
    assert out.x.tolist() == [0, 10, 10, 0, 100, 200, 200, 100, 120, 120, 180, 180]
    assert out.y.tolist() == [0, 0, 10, 10, 100, 100, 200, 200, 120, 180, 180, 120]


def test_river_columns_leave_a_ring_that_is_already_open_alone() -> None:
    closed_ring = closed([(0, 0), (10, 0), (10, 10), (0, 10)])
    open_ring = [(100, 0), (110, 0), (110, 10), (100, 10)]
    out = rules.river_columns(water_table_of([(0, [closed_ring]), (1, [open_ring])]))
    assert out.offsets.tolist() == [0, 4, 8]
    assert out.x.tolist() == [0, 10, 10, 0, 100, 110, 110, 100]


def test_river_columns_keep_holes_with_their_polygon_and_do_not_reorient() -> None:
    lake = closed([(0, 0), (100, 0), (100, 100), (0, 100)])
    hole = closed([(40, 40), (40, 60), (60, 60), (60, 40)])
    out = rules.river_columns(water_table_of([(1, [lake, hole]), (1, [lake])]))
    assert out.rings.tolist() == [2, 1]
    assert out.offsets.tolist() == [0, 4, 8, 12]
    assert out.x[4:8].tolist() == [40, 40, 60, 60]
    assert out.y[4:8].tolist() == [40, 60, 60, 40]


def test_river_columns_dtypes_are_pack_compatible() -> None:
    out = rules.river_columns(water_table_of([(0, [closed([(0, 0), (10, 0), (10, 10), (0, 10)])])]))
    assert out.x.dtype == np.float32
    assert out.y.dtype == np.float32


def district(**changes: Any) -> dict[str, Any]:
    base = {
        "id": 3,
        "name": "Kālupur",
        "name_local": "કાલુપુર",
        "seed": [10.5, -20.5],
        "bank": 1,
        "buildings": 99,
        "polygon": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]]],
    }
    return {**base, **changes}


def test_pack_districts_keep_exactly_the_documented_keys() -> None:
    (out,) = rules.pack_districts([district()])
    assert sorted(out) == ["east", "id", "name", "name_local", "polygons", "x", "y"]
    assert "bank" not in out and "buildings" not in out and "seed" not in out
    assert "polygon" not in out


def test_pack_districts_take_the_position_from_the_seed_and_east_from_the_bank() -> None:
    west, east = rules.pack_districts([district(bank=0), district(bank=1)])
    assert (west["x"], west["y"], west["east"]) == (10.5, -20.5, False)
    assert (east["x"], east["y"], east["east"]) == (10.5, -20.5, True)
    assert isinstance(west["east"], bool)
    assert isinstance(east["east"], bool)


def test_pack_districts_polygons_are_open_rings_and_open_rings_stay_open() -> None:
    parts = [
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]],
        [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0]],
    ]
    (out,) = rules.pack_districts([district(polygon=parts)])
    assert out["polygons"] == [
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
        [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0]],
    ]


def test_pack_districts_preserve_non_ascii_names_and_survive_json() -> None:
    out = rules.pack_districts([district()])
    assert out[0]["name"] == "Kālupur"
    assert out[0]["name_local"] == "કાલુપુર"
    assert json.loads(json.dumps(out, ensure_ascii=False)) == out


def test_pack_districts_keep_id_order() -> None:
    out = rules.pack_districts([district(id=0), district(id=1), district(id=2)])
    assert [d["id"] for d in out] == [0, 1, 2]


def test_attributions_of_metas_without_the_key_is_empty() -> None:
    assert rules.attributions({}, {"city": "x"}) == []


def test_attributions_accept_a_string_and_a_list_and_flatten_in_order() -> None:
    metas = [{"attribution": "A"}, {"attribution": ["B", "C"]}, {}, {"attribution": ["D"]}]
    assert rules.attributions(*metas) == ["A", "B", "C", "D"]


def test_attributions_with_no_metas_is_empty() -> None:
    assert rules.attributions() == []
