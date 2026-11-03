import numpy as np
import pytest

from twl_engine.fixtures.city import CityLayout, build_layout
from twl_engine.fixtures.geometry import (
    DISTRICT_TABLE,
    DISTRICT_X,
    DISTRICT_Z,
    district_at,
    river_half_width,
    river_x,
    toy_districts,
)
from twl_engine.fixtures.types import ROAD_ARTERIAL, ROAD_BRIDGE, ROAD_STREET
from twl_engine.rng import spawn_streams


@pytest.fixture(scope="module")
def layout() -> CityLayout:
    return build_layout(spawn_streams(49, ["city"])["city"])


def test_only_shahibaug_and_maninagar_are_east_of_the_river() -> None:
    assert [d.name for d in toy_districts() if d.east] == ["Shahibaug", "Maninagar"]


def test_each_district_centre_belongs_to_its_own_district() -> None:
    np.testing.assert_array_equal(
        district_at(DISTRICT_X, DISTRICT_Z), np.arange(len(DISTRICT_TABLE))
    )


def test_districts_use_pack_axes_with_north_positive() -> None:
    chandkheda = next(d for d in toy_districts() if d.name == "Chandkheda")
    assert (chandkheda.id, chandkheda.x, chandkheda.y) == (4, -250.0, 2650.0)


def test_river_is_one_counter_clockwise_ring_around_the_centreline(layout: CityLayout) -> None:
    river = layout.river
    assert river.offsets.tolist() == [0, river.x.size]
    x, y = river.x.astype(np.float64), river.y.astype(np.float64)
    shoelace = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2
    assert shoelace > 0
    half = x.size // 2
    np.testing.assert_allclose((x[:half] + x[half:][::-1]) / 2, river_x(-y[:half]), atol=1e-3)


def test_buildings_keep_clear_of_the_river(layout: CityLayout) -> None:
    b = layout.buildings
    x, z = b.x.astype(np.float64), -b.y.astype(np.float64)
    assert np.all(np.abs(x - river_x(z)) >= river_half_width(z) + 22)


def test_buildings_stay_in_their_districts_side_of_the_river(layout: CityLayout) -> None:
    b = layout.buildings
    east_bank = b.x > river_x(-b.y.astype(np.float64))
    east_district = np.array([d.east for d in toy_districts()])[b.district]
    np.testing.assert_array_equal(east_bank, east_district)


def test_roads_have_every_kind_and_two_point_streets(layout: CityLayout) -> None:
    roads = layout.roads
    lengths = np.diff(roads.offsets.astype(np.int64))
    assert roads.offsets[-1] == roads.x.size == roads.y.size
    assert set(np.unique(roads.kind).tolist()) == {ROAD_STREET, ROAD_ARTERIAL, ROAD_BRIDGE}
    assert np.all(lengths[roads.kind == ROAD_STREET] == 2)
    assert np.all(lengths[roads.kind == ROAD_BRIDGE] == 2)
    assert np.all(lengths >= 2)


def test_bridges_cross_the_river(layout: CityLayout) -> None:
    roads = layout.roads
    starts = roads.offsets[:-1][roads.kind == ROAD_BRIDGE]
    west, east = roads.x[starts], roads.x[starts + 1]
    z = -roads.y[starts].astype(np.float64)
    assert np.all(west < river_x(z) - river_half_width(z))
    assert np.all(east > river_x(z) + river_half_width(z))
