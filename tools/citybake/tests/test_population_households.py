import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from population_fixture import CENSUS_COUNTS, CENSUS_SIZES, synthetic_households_input

from twl_citybake.population.households import Households, synthesise_households

SIZES = np.asarray(CENSUS_SIZES, np.int64)
COUNTS = np.asarray(CENSUS_COUNTS, np.int64)
N_DISTRICTS = 4


def synthesise(case: dict[str, np.ndarray]) -> Households:
    return synthesise_households(
        case["placed"],
        case["marginal"],
        case["capacity"],
        case["district"],
        N_DISTRICTS,
        SIZES,
        COUNTS,
    )


def building_persons(h: Households, n: int) -> np.ndarray:
    return np.bincount(h.building, weights=h.size, minlength=n)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_district_totals_are_within_one_percent_of_marginals(seed: int) -> None:
    case = synthetic_households_input(seed)
    h = synthesise(case)
    persons = np.bincount(h.district, weights=h.size, minlength=N_DISTRICTS)
    big = case["marginal"] >= 1000
    assert big.any()
    rel = np.abs(persons - case["marginal"])[big] / case["marginal"][big]
    assert (rel <= 0.01).all(), rel


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_no_building_is_over_capacity(seed: int) -> None:
    case = synthetic_households_input(seed)
    h = synthesise(case)
    persons = building_persons(h, len(case["capacity"]))
    assert (persons <= case["capacity"]).all()


def test_full_buildings_stay_within_capacity_and_close_to_district_marginals() -> None:
    case = synthetic_households_input(21)
    case["placed"] = case["capacity"].astype(np.float64)
    case["marginal"] = np.bincount(case["district"], weights=case["placed"], minlength=N_DISTRICTS)
    h = synthesise(case)
    assert (building_persons(h, len(case["capacity"])) <= case["capacity"]).all()
    persons = np.bincount(h.district, weights=h.size, minlength=N_DISTRICTS)
    assert (np.abs(persons - case["marginal"]) <= 0.01 * case["marginal"]).all()


def test_household_size_shares_follow_the_census() -> None:
    h = synthesise(synthetic_households_input(11))
    shares = np.asarray([(h.size == s).mean() for s in SIZES])
    census = COUNTS / COUNTS.sum()
    assert np.abs(shares - census).max() < 0.02


def test_households_are_sorted_by_building_then_size() -> None:
    h = synthesise(synthetic_households_input(11))
    order = np.lexsort((h.size, h.building))
    assert np.array_equal(order, np.arange(len(h.building)))


def test_household_columns_have_the_documented_dtypes_and_lengths() -> None:
    h = synthesise(synthetic_households_input(11))
    assert h.building.dtype == np.uint32
    assert h.size.dtype == np.uint8
    assert h.district.dtype == np.uint16
    assert len(h.building) == len(h.size) == len(h.district) > 0
    assert isinstance(h.report, dict)


def test_household_district_is_the_district_of_its_building() -> None:
    case = synthetic_households_input(11)
    h = synthesise(case)
    assert np.array_equal(h.district, case["district"][h.building])


def test_household_sizes_come_from_the_census_bins() -> None:
    h = synthesise(synthetic_households_input(11))
    assert set(np.unique(h.size).tolist()) <= set(CENSUS_SIZES)


def test_buildings_smaller_than_a_household_size_never_get_that_size() -> None:
    case = synthetic_households_input(11)
    case["capacity"] = np.minimum(case["capacity"], 10)
    case["placed"] = np.minimum(case["placed"], case["capacity"] * 0.8)
    case["marginal"] = np.bincount(case["district"], weights=case["placed"], minlength=N_DISTRICTS)
    h = synthesise(case)
    assert (h.size <= case["capacity"][h.building]).all()
    assert h.size.max() <= 10


def test_districts_without_placed_people_get_no_households() -> None:
    case = synthetic_households_input(11)
    empty = case["district"] == 3
    case["placed"] = np.where(empty, 0.0, case["placed"])
    case["marginal"] = np.bincount(case["district"], weights=case["placed"], minlength=N_DISTRICTS)
    h = synthesise(case)
    assert not (h.district == 3).any()
    assert (h.district == 0).any()


def test_synthesis_is_deterministic() -> None:
    case = synthetic_households_input(11)
    a, b = synthesise(case), synthesise(case)
    assert np.array_equal(a.building, b.building)
    assert np.array_equal(a.size, b.size)


def single_district(placed: np.ndarray, capacity: np.ndarray) -> Households:
    return synthesise_households(
        placed,
        [placed.sum()],
        capacity,
        np.zeros(len(placed), np.uint16),
        1,
        SIZES,
        COUNTS,
    )


def test_singles_absorb_per_column_rounding_so_district_persons_match_placed() -> None:
    placed = np.full(100, 20.01)
    expected = COUNTS * SIZES / (COUNTS * SIZES).sum() * placed.sum() / SIZES
    assert abs(placed.sum() - np.round(expected).astype(int) @ SIZES) >= 10
    h = single_district(placed, np.full(100, 200, np.int64))
    assert abs(int(h.size.sum()) - placed.sum()) <= 0.5


@st.composite
def ample_districts(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(60, 200))
    return np.asarray(draw(st.lists(st.floats(5.0, 60.0), min_size=n, max_size=n)))


@settings(deadline=None, max_examples=40)
@given(ample_districts())
def test_district_persons_are_within_half_a_person_of_placed_when_capacity_is_ample(
    placed: np.ndarray,
) -> None:
    h = single_district(placed, np.full(len(placed), 1000, np.int64))
    assert abs(int(h.size.sum()) - placed.sum()) <= 0.5


def test_a_rounding_overshoot_larger_than_the_singles_sheds_small_households() -> None:
    placed = np.full(100, 20.02)
    h = single_district(placed, np.full(100, 1000, np.int64))
    assert int(h.size.sum()) == 2002
