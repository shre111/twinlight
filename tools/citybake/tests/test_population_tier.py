from random import Random

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from twl_citybake.population.tier import HIGH, LOW, MIDDLE, Tiers, building_tiers


def tiers_of(
    area: npt.ArrayLike,
    floors: npt.ArrayLike,
    persons: npt.ArrayLike,
    households: npt.ArrayLike,
    low: float = 0.25,
    high: float = 0.25,
) -> Tiers:
    return building_tiers(
        np.asarray(area, np.float64),
        np.asarray(floors, np.int64),
        np.asarray(persons, np.int64),
        np.asarray(households, np.int64),
        low,
        high,
    )


def test_tier_constants_are_zero_one_two() -> None:
    assert (LOW, MIDDLE, HIGH) == (0, 1, 2)


def test_eight_equal_buildings_are_cut_at_the_second_and_sixth() -> None:
    ones = np.ones(8)
    result = tiers_of(np.arange(1, 9) * 10.0, ones, ones, ones)
    assert result.tier.tolist() == [LOW, MIDDLE, MIDDLE, MIDDLE, MIDDLE, HIGH, HIGH, HIGH]
    assert result.low_cut == 20.0
    assert result.high_cut == 60.0
    assert result.tier.dtype == np.uint8


def test_cuts_are_weighted_by_households_not_buildings() -> None:
    result = tiers_of([50, 20, 60, 160], [1, 1, 1, 1], [5, 1, 2, 4], [5, 1, 2, 4])
    assert result.low_cut == 10.0
    assert result.high_cut == 40.0
    assert result.tier.tolist() == [MIDDLE, MIDDLE, MIDDLE, HIGH]


def test_space_is_area_times_floors_over_persons_and_unhoused_buildings_are_middle() -> None:
    result = tiers_of(
        [100, 40, 90, 500], [3, 1, 2, 5], [6, 4, 3, 0], [2, 1, 1, 0], low=0.25, high=0.25
    )
    assert (result.low_cut, result.high_cut) == (10.0, 50.0)
    assert result.tier.tolist() == [HIGH, MIDDLE, HIGH, MIDDLE]


def test_input_order_does_not_decide_the_cut() -> None:
    area = np.array([80, 10, 60, 30, 70, 20, 50, 40], np.float64)
    ones = np.ones(8)
    result = tiers_of(area, ones, ones, ones)
    assert result.tier.tolist() == [HIGH, LOW, HIGH, MIDDLE, HIGH, MIDDLE, MIDDLE, MIDDLE]


def test_equal_space_buildings_all_land_in_the_same_tier() -> None:
    ones = np.ones(4)
    result = tiers_of(np.full(4, 10.0), ones, ones, ones)
    assert result.tier.tolist() == [HIGH] * 4
    assert result.low_cut == result.high_cut == 10.0


def test_the_same_inputs_give_the_same_result() -> None:
    args = ([10, 25, 17, 90], [1, 2, 1, 3], [3, 4, 2, 5], [1, 1, 1, 2])
    a, b = tiers_of(*args), tiers_of(*args)
    assert a.tier.tolist() == b.tier.tolist()
    assert (a.low_cut, a.high_cut) == (b.low_cut, b.high_cut)


@pytest.mark.parametrize(
    "args",
    [
        ([10, 20], [1], [1, 1], [1, 1]),
        ([10, 20], [1, 1], [1, 1, 1], [1, 1]),
        ([10, 20], [1, 1], [1, 1], [1]),
        ([10, 20], [1, 1], [1, 1], [1, 1, 1]),
    ],
)
def test_arrays_of_different_lengths_are_refused(args: tuple) -> None:
    with pytest.raises(ValueError):
        tiers_of(*args)


def test_persons_without_households_are_refused() -> None:
    with pytest.raises(ValueError):
        tiers_of([10, 20], [1, 1], [1, 3], [1, 0])


def test_households_without_persons_are_refused() -> None:
    with pytest.raises(ValueError):
        tiers_of([10, 20], [1, 1], [1, 0], [1, 1])


def test_a_city_with_no_housed_building_is_refused() -> None:
    with pytest.raises(ValueError):
        tiers_of([10, 20], [1, 1], [0, 0], [0, 0])


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.0, 0.2), (0.2, 0.0), (1.0, 0.2), (0.2, 1.0), (-0.1, 0.2), (0.5, 0.5), (0.7, 0.4)],
)
def test_shares_outside_the_open_interval_or_summing_to_one_are_refused(
    low: float, high: float
) -> None:
    with pytest.raises(ValueError):
        tiers_of([10, 20, 30], [1, 1, 1], [1, 1, 1], [1, 1, 1], low, high)


@st.composite
def cities(draw: st.DrawFn) -> dict:
    n = draw(st.integers(1, 30))
    households = np.array(draw(st.lists(st.integers(0, 5), min_size=n, max_size=n)), np.int64)
    households[draw(st.integers(0, n - 1))] += 1
    size = np.array(draw(st.lists(st.integers(1, 6), min_size=n, max_size=n)), np.int64)
    return {
        "area": np.array(draw(st.lists(st.integers(10, 2000), min_size=n, max_size=n)), np.float64),
        "floors": np.array(draw(st.lists(st.integers(1, 8), min_size=n, max_size=n)), np.int64),
        "persons": np.where(households > 0, size, 0),
        "households": households,
        "low": draw(st.floats(0.05, 0.9)),
        "high": draw(st.floats(0.05, 0.9)),
    }


def run(city: dict, **over: np.ndarray) -> Tiers:
    c = {**city, **over}
    assume(c["low"] + c["high"] < 0.99)
    return building_tiers(
        c["area"], c["floors"], c["persons"], c["households"], c["low"], c["high"]
    )


def space_of(city: dict) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return city["area"] * city["floors"] / city["persons"]


@settings(max_examples=150, deadline=None)
@given(cities())
def test_realised_low_share_is_at_most_and_high_share_at_least_the_asked_share(city: dict) -> None:
    result = run(city)
    hh = city["households"]
    total = hh.sum()
    assert hh[result.tier == LOW].sum() <= city["low"] * total + 1e-9
    assert hh[result.tier == HIGH].sum() >= city["high"] * total - 1e-9


@settings(max_examples=150, deadline=None)
@given(cities())
def test_cuts_are_ordered_and_tier_never_falls_as_space_rises(city: dict) -> None:
    result = run(city)
    assert result.low_cut <= result.high_cut
    housed = city["households"] > 0
    order = np.argsort(space_of(city)[housed], kind="stable")
    assert (np.diff(result.tier[housed][order].astype(int)) >= 0).all()


@settings(max_examples=150, deadline=None)
@given(cities())
def test_buildings_with_equal_space_share_a_tier(city: dict) -> None:
    result = run(city)
    housed = city["households"] > 0
    space = space_of(city)[housed]
    tier = result.tier[housed]
    for value in np.unique(space):
        assert len(set(tier[space == value].tolist())) == 1


@settings(max_examples=100, deadline=None)
@given(cities(), st.integers(-4, 6))
def test_scaling_every_area_by_a_constant_scales_the_cuts_and_keeps_the_tiers(
    city: dict, exponent: int
) -> None:
    k = 2.0**exponent
    base, scaled = run(city), run(city, area=city["area"] * k)
    assert scaled.tier.tolist() == base.tier.tolist()
    assert scaled.low_cut == base.low_cut * k
    assert scaled.high_cut == base.high_cut * k


@settings(max_examples=100, deadline=None)
@given(cities(), st.randoms(use_true_random=False))
def test_permuting_buildings_permutes_the_tiers_with_them(city: dict, rnd: Random) -> None:
    order = list(range(len(city["area"])))
    rnd.shuffle(order)
    perm = np.array(order)
    base = run(city)
    shuffled = run(city, **{k: city[k][perm] for k in ("area", "floors", "persons", "households")})
    assert shuffled.tier.tolist() == base.tier[perm].tolist()
    assert (shuffled.low_cut, shuffled.high_cut) == (base.low_cut, base.high_cut)


@settings(max_examples=100, deadline=None)
@given(cities())
def test_unhoused_buildings_are_middle_and_do_not_move_the_cuts(city: dict) -> None:
    result = run(city)
    assert (result.tier[city["households"] == 0] == MIDDLE).all()
    housed = city["households"] > 0
    only = building_tiers(
        city["area"][housed],
        city["floors"][housed],
        city["persons"][housed],
        city["households"][housed],
        city["low"],
        city["high"],
    )
    assert (only.low_cut, only.high_cut) == (result.low_cut, result.high_cut)
    assert only.tier.tolist() == result.tier[housed].tolist()
