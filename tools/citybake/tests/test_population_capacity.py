import numpy as np

from twl_citybake.osm.tags import BuildingKind
from twl_citybake.population.capacity import residential_capacity

SHARE = {
    BuildingKind.UNKNOWN: 0.9,
    BuildingKind.RESIDENTIAL: 1.0,
    BuildingKind.COMMERCIAL: 0.2,
    BuildingKind.INDUSTRIAL: 0.05,
    BuildingKind.CIVIC: 0.1,
    BuildingKind.RELIGIOUS: 0.0,
    BuildingKind.OTHER: 0.0,
}


def capacity(area: list[float], height: list[float], kind: list[BuildingKind]) -> list[int]:
    out = residential_capacity(
        np.asarray(area, np.float64),
        np.asarray(height, np.float64),
        np.asarray(kind, np.uint8),
        SHARE,
        4.0,
    )
    return out.tolist()


def test_capacity_is_floor_of_area_times_storeys_times_share_over_floor_per_person() -> None:
    assert capacity([100.0], [9.6], [BuildingKind.RESIDENTIAL]) == [75]
    assert capacity([101.0], [9.6], [BuildingKind.RESIDENTIAL]) == [75]
    assert capacity([100.0], [9.6], [BuildingKind.COMMERCIAL]) == [15]


def test_storeys_are_height_over_3_2_rounded() -> None:
    heights = [3.2, 4.7, 5.0, 6.4, 12.8]
    got = capacity([40.0] * 5, heights, [BuildingKind.RESIDENTIAL] * 5)
    assert got == [10, 10, 20, 20, 40]


def test_buildings_lower_than_one_storey_count_as_one_storey() -> None:
    got = capacity([80.0, 80.0], [0.0, 1.0], [BuildingKind.RESIDENTIAL] * 2)
    assert got == [20, 20]


def test_kinds_with_zero_share_have_no_capacity() -> None:
    got = capacity([500.0] * 2, [20.0] * 2, [BuildingKind.RELIGIOUS, BuildingKind.OTHER])
    assert got == [0, 0]


def test_each_kind_uses_its_own_share() -> None:
    kinds = list(BuildingKind)
    got = capacity([321.0] * len(kinds), [3.2] * len(kinds), kinds)
    assert got == [int(321 * SHARE[k] / 4.0) for k in kinds]


def test_capacity_is_int64_and_never_negative() -> None:
    out = residential_capacity(
        np.array([0.0, 0.3, 1000.0]),
        np.array([3.0, 3.0, 30.0]),
        np.array([1, 2, 3], np.uint8),
        SHARE,
        4.0,
    )
    assert out.dtype == np.int64
    assert (out >= 0).all()
    assert out[0] == 0


def test_denser_floor_allowance_raises_capacity() -> None:
    args = (np.array([200.0]), np.array([6.4]), np.array([1], np.uint8), SHARE)
    assert residential_capacity(*args, 2.0)[0] == 2 * residential_capacity(*args, 4.0)[0]
