import numpy as np
import pytest

from twl_citybake.osm.heights import (
    FALLBACK_LEVELS,
    FOOTPRINT_BANDS_M2,
    MIN_HEIGHT_SAMPLES,
    default_levels,
    final_heights,
)
from twl_citybake.osm.tags import LEVEL_HEIGHT_M, BuildingKind, HeightSource

RES = int(BuildingKind.RESIDENTIAL)
COM = int(BuildingKind.COMMERCIAL)
CIVIC = int(BuildingKind.CIVIC)
N = MIN_HEIGHT_SAMPLES


def half_rounded(value: float) -> float:
    return float(np.round(value * 2) / 2)


class Rows:
    def __init__(self) -> None:
        self.district: list[int] = []
        self.kind: list[int] = []
        self.area: list[float] = []
        self.levels: list[float] = []

    def sampled(self, count: int, district: int, kind: int, area: float, levels: float) -> "Rows":
        self._add(count, district, kind, area, levels)
        return self

    def query(self, district: int, kind: int, area: float) -> int:
        self._add(1, district, kind, area, float("nan"))
        return len(self.levels) - 1

    def _add(self, count: int, district: int, kind: int, area: float, levels: float) -> None:
        self.district += [district] * count
        self.kind += [kind] * count
        self.area += [area] * count
        self.levels += [levels] * count

    def solve(self) -> np.ndarray:
        return default_levels(
            np.array(self.district, np.uint16),
            np.array(self.kind, np.uint8),
            np.array(self.area, np.float64),
            np.array(self.levels, np.float64),
        )


def test_footprint_bands_cut_at_250_and_700() -> None:
    assert FOOTPRINT_BANDS_M2 == (250.0, 700.0)


def test_no_level_data_gives_two_levels() -> None:
    rows = Rows()
    rows.query(0, RES, 100.0)
    rows.query(1, COM, 900.0)

    assert FALLBACK_LEVELS == 2.0
    assert np.all(rows.solve() == 2.0)


def test_city_median_applies_even_with_tiny_groups() -> None:
    rows = Rows().sampled(1, 0, RES, 100.0, 3.0).sampled(1, 0, COM, 100.0, 5.0)
    rows.sampled(1, 0, COM, 100.0, 7.0)
    q = rows.query(4, CIVIC, 300.0)

    assert rows.solve()[q] == 5.0


def test_kind_group_overrides_city_median() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 4.0).sampled(N, 0, COM, 100.0, 10.0)
    q_res = rows.query(7, RES, 800.0)
    q_com = rows.query(7, COM, 800.0)
    q_other = rows.query(7, CIVIC, 800.0)

    out = rows.solve()
    assert out[q_res] == 4.0
    assert out[q_com] == 10.0
    assert out[q_other] == 7.0


def test_kind_group_just_under_threshold_is_ignored() -> None:
    rows = Rows().sampled(N - 1, 0, RES, 100.0, 4.0).sampled(N, 0, COM, 100.0, 10.0)
    q = rows.query(7, RES, 800.0)

    city = half_rounded(float(np.median([4.0] * (N - 1) + [10.0] * N)))
    assert rows.solve()[q] == city
    assert rows.solve()[q] != 4.0


def test_kind_group_at_threshold_is_used() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 4.0).sampled(50, 0, COM, 100.0, 10.0)
    q = rows.query(7, RES, 800.0)

    assert rows.solve()[q] == 4.0


def test_district_kind_group_overrides_kind_group() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 2.0).sampled(N, 1, RES, 300.0, 6.0)
    q_one = rows.query(1, RES, 800.0)
    q_none = rows.query(2, RES, 800.0)

    out = rows.solve()
    assert out[q_one] == 6.0
    assert out[q_none] == 4.0


def test_district_kind_group_just_under_threshold_falls_back_to_kind() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 2.0).sampled(N - 1, 1, RES, 300.0, 6.0)
    q = rows.query(1, RES, 800.0)

    assert rows.solve()[q] == half_rounded(float(np.median([2.0] * N + [6.0] * (N - 1))))


def test_kind_band_group_overrides_kind_group() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 2.0).sampled(N, 0, RES, 800.0, 8.0)
    q_small = rows.query(5, RES, 100.0)
    q_large = rows.query(5, RES, 800.0)
    q_mid = rows.query(5, RES, 400.0)

    out = rows.solve()
    assert out[q_small] == 2.0
    assert out[q_large] == 8.0
    assert out[q_mid] == 5.0


def test_kind_band_group_just_under_threshold_falls_back_to_kind() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 2.0).sampled(N - 1, 0, RES, 800.0, 8.0)
    q = rows.query(5, RES, 800.0)

    assert rows.solve()[q] == half_rounded(float(np.median([2.0] * N + [8.0] * (N - 1))))
    assert rows.solve()[q] != 8.0


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        (0.0, 2.0),
        (249.9, 2.0),
        (250.0, 4.0),
        (699.9, 4.0),
        (700.0, 8.0),
        (50_000.0, 8.0),
    ],
)
def test_band_cut_points_belong_to_the_upper_band(area: float, expected: float) -> None:
    rows = (
        Rows()
        .sampled(N, 0, RES, 100.0, 2.0)
        .sampled(N, 0, RES, 400.0, 4.0)
        .sampled(N, 0, RES, 900.0, 8.0)
    )
    q = rows.query(9, RES, area)

    assert rows.solve()[q] == expected


def test_samples_exactly_on_a_cut_count_in_the_upper_band() -> None:
    rows = Rows().sampled(N, 0, RES, 250.0, 6.0).sampled(N, 0, RES, 100.0, 2.0)
    q_at = rows.query(9, RES, 250.0)
    q_below = rows.query(9, RES, 249.9)

    out = rows.solve()
    assert out[q_at] == 6.0
    assert out[q_below] == 2.0


def test_district_kind_band_group_is_the_finest_override() -> None:
    rows = (
        Rows()
        .sampled(N, 0, RES, 100.0, 3.0)
        .sampled(N, 0, RES, 400.0, 6.0)
        .sampled(N, 1, RES, 100.0, 2.0)
    )
    q_finest = rows.query(0, RES, 100.0)
    q_district_kind = rows.query(0, RES, 800.0)
    q_other_district = rows.query(1, RES, 100.0)
    q_kind_band = rows.query(2, RES, 100.0)

    out = rows.solve()
    assert out[q_finest] == 3.0
    assert out[q_district_kind] == 4.5
    assert out[q_other_district] == 2.0
    assert out[q_kind_band] == 2.5


def test_district_kind_band_group_just_under_threshold_falls_back() -> None:
    rows = (
        Rows()
        .sampled(N - 1, 0, RES, 100.0, 3.0)
        .sampled(N, 0, RES, 400.0, 6.0)
        .sampled(N, 1, RES, 100.0, 2.0)
    )
    q = rows.query(0, RES, 100.0)

    kind_band = half_rounded(float(np.median([3.0] * (N - 1) + [2.0] * N)))
    assert rows.solve()[q] == kind_band
    assert rows.solve()[q] != 3.0


def test_groups_do_not_mix_kinds() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 3.0).sampled(N, 0, COM, 100.0, 9.0)
    q_res = rows.query(0, RES, 100.0)
    q_com = rows.query(0, COM, 100.0)

    out = rows.solve()
    assert out[q_res] == 3.0
    assert out[q_com] == 9.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3.2, 3.0), (3.3, 3.5), (3.7, 3.5), (3.8, 4.0), (5.0, 5.0)],
)
def test_medians_round_to_the_nearest_half_level(value: float, expected: float) -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, value)
    q = rows.query(0, RES, 100.0)

    assert rows.solve()[q] == expected


def test_even_sized_group_uses_mean_of_middle_values() -> None:
    rows = Rows().sampled(N // 2, 0, RES, 100.0, 3.0).sampled(N // 2, 0, RES, 100.0, 4.0)
    q = rows.query(0, RES, 100.0)

    assert rows.solve()[q] == 3.5


def test_final_heights_pass_tagged_values_and_default_the_rest() -> None:
    rows = Rows().sampled(N, 0, RES, 100.0, 4.0)
    q = rows.query(0, RES, 100.0)
    size = len(rows.levels)
    tagged = np.full(size, np.nan, np.float64)
    source = np.full(size, int(HeightSource.LEVELS), np.uint8)
    tagged[0] = 17.3
    source[0] = int(HeightSource.HEIGHT)
    tagged[1] = 12.8
    source[q] = int(HeightSource.DEFAULT)

    out = final_heights(
        tagged,
        source,
        np.array(rows.district, np.uint16),
        np.array(rows.kind, np.uint8),
        np.array(rows.area, np.float64),
        np.array(rows.levels, np.float64),
    )

    assert out.dtype == np.float32
    assert out[0] == np.float32(17.3)
    assert out[1] == np.float32(12.8)
    assert out[q] == pytest.approx(4.0 * LEVEL_HEIGHT_M)


def test_final_heights_use_two_levels_when_nothing_is_tagged() -> None:
    out = final_heights(
        np.array([np.nan, np.nan]),
        np.array([HeightSource.DEFAULT, HeightSource.DEFAULT], np.uint8),
        np.array([0, 1], np.uint16),
        np.array([RES, COM], np.uint8),
        np.array([100.0, 900.0]),
        np.array([np.nan, np.nan]),
    )

    assert np.allclose(out, 2 * LEVEL_HEIGHT_M)
