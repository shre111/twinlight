from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from twl_citybake.personas.pool import FEMALE, MALE, PersonaPool
from twl_citybake.personas.tilt import odds_ratios, tier_education_shares, tilted_weights

MARRIED_CODE = 1
AHMEDABAD_QUINTILES = (3.2, 7.2, 15.5, 28.6, 45.5)
AHMEDABAD_FEMALE = (7.5, 15.5, 23.9, 34.6, 53.4)
AHMEDABAD_MALE = (12.0, 23.1, 33.7, 45.2, 63.6)


def pool_of(sex: NDArray[np.uint8], marital: NDArray[np.uint8]) -> PersonaPool:
    n = len(sex)
    return PersonaPool(
        uuid=np.zeros(n, "S36"),
        sex=sex,
        age=np.full(n, 30, np.uint8),
        marital=marital,
        education=np.zeros(n, np.uint8),
        occupation=np.zeros(n, np.uint16),
        language=np.zeros(n, np.uint8),
        vocab={},
    )


@dataclass(frozen=True)
class Case:
    weights: NDArray[np.float64]
    pool: PersonaPool
    ten_plus: NDArray[np.bool_]
    tier_share: tuple[float, float, float]
    ratios_female: tuple[float, float, float]
    ratios_male: tuple[float, float, float]

    def run(self) -> NDArray[np.float64]:
        return tilted_weights(
            self.weights,
            self.pool,
            MARRIED_CODE,
            self.ten_plus,
            self.tier_share,
            self.ratios_female,
            self.ratios_male,
        )

    def groups(self) -> list[NDArray[np.bool_]]:
        married = self.pool.marital == MARRIED_CODE
        female = self.pool.sex == FEMALE
        return [married & female, married & ~female, ~married & female, ~married & ~female]


ratio = st.floats(0.2, 5.0)
ratio_triple = st.tuples(ratio, ratio, ratio)


@st.composite
def cases(draw: st.DrawFn, ratios: bool = True) -> Case:
    n = draw(st.integers(8, 60))
    sex = np.array(
        draw(st.lists(st.sampled_from([FEMALE, MALE]), min_size=n, max_size=n)), np.uint8
    )
    marital = np.array(draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n)), np.uint8)
    ten_plus = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)), np.bool_)
    raw = np.array(draw(st.lists(st.integers(1, 20), min_size=n, max_size=n)), np.float64)
    shares = np.array(draw(st.lists(st.integers(0, 10), min_size=3, max_size=3)), np.float64)
    if shares.sum() == 0:
        shares[1] = 1.0
    shares /= shares.sum()
    ones = (1.0, 1.0, 1.0)
    return Case(
        weights=raw / raw.sum(),
        pool=pool_of(sex, marital),
        ten_plus=ten_plus,
        tier_share=(float(shares[0]), float(shares[1]), float(shares[2])),
        ratios_female=draw(ratio_triple) if ratios else ones,
        ratios_male=draw(ratio_triple) if ratios else ones,
    )


def mass_and_p(case: Case, group: NDArray[np.bool_]) -> tuple[float, float]:
    mass = float(case.weights[group].sum())
    ten = float(case.weights[group & case.ten_plus].sum())
    return mass, (ten / mass if mass > 0 else 0.0)


def test_shares_for_equal_quintiles_average_the_quintiles_each_tier_covers() -> None:
    low, middle, high = tier_education_shares((20, 20, 20, 20, 20), (10, 20, 30, 40, 50), 0.2, 0.2)
    assert (low, middle, high) == pytest.approx((0.10, 0.30, 0.50))


def test_a_tier_cut_inside_a_quintile_takes_the_overlap_weighted_mean() -> None:
    low, middle, high = tier_education_shares((20, 20, 20, 20, 20), (10, 20, 30, 40, 50), 0.3, 0.1)
    assert low == pytest.approx((0.2 * 10 + 0.1 * 20) / 0.3 / 100)
    assert middle == pytest.approx((0.1 * 20 + 0.2 * 30 + 0.2 * 40 + 0.1 * 50) / 0.6 / 100)
    assert high == pytest.approx(0.50)


def test_quintile_population_is_normalised_before_the_tiers_are_laid_out() -> None:
    a = tier_education_shares((1, 1, 1, 1, 1), (10, 20, 30, 40, 50), 0.3, 0.1)
    b = tier_education_shares((7, 7, 7, 7, 7), (10, 20, 30, 40, 50), 0.3, 0.1)
    assert a == pytest.approx(b)


def test_ahmedabad_female_shares_match_the_spec_to_four_places() -> None:
    shares = tier_education_shares(AHMEDABAD_QUINTILES, AHMEDABAD_FEMALE, 0.38, 0.20)
    assert [round(s, 4) for s in shares] == [0.2433, 0.4601, 0.534]


def test_ahmedabad_male_shares_match_the_spec_to_four_places() -> None:
    shares = tier_education_shares(AHMEDABAD_QUINTILES, AHMEDABAD_MALE, 0.38, 0.20)
    assert [round(s, 4) for s in shares] == [0.3353, 0.5637, 0.636]


@given(
    st.lists(st.floats(0.0, 100.0), min_size=5, max_size=5),
    st.floats(0.05, 0.6),
    st.floats(0.05, 0.3),
)
def test_a_constant_education_curve_gives_that_constant_in_every_tier(
    quintile_population: list[float], low: float, high: float
) -> None:
    if sum(quintile_population) < 1.0:
        quintile_population = [1.0] * 5
    shares = tier_education_shares(tuple(quintile_population), (42.0,) * 5, low, high)
    assert shares == pytest.approx((0.42, 0.42, 0.42))


def test_odds_ratios_are_relative_to_the_middle_tier() -> None:
    assert odds_ratios((0.2, 0.5, 0.8)) == pytest.approx((0.25, 1.0, 4.0))


def test_ahmedabad_odds_ratios_rise_from_low_to_high() -> None:
    for by_quintile in (AHMEDABAD_FEMALE, AHMEDABAD_MALE):
        low, middle, high = odds_ratios(
            tier_education_shares(AHMEDABAD_QUINTILES, by_quintile, 0.38, 0.20)
        )
        assert middle == 1.0
        assert low < 1.0 < high


@given(st.tuples(*[st.floats(0.01, 0.99)] * 3))
def test_odds_ratio_times_the_middle_odds_is_the_tier_odds(
    shares: tuple[float, float, float],
) -> None:
    ratios = odds_ratios(shares)
    odds = [s / (1 - s) for s in shares]
    assert ratios[1] == pytest.approx(1.0)
    assert ratios == pytest.approx(tuple(o / odds[1] for o in odds), rel=1e-9)


@settings(max_examples=150, deadline=None)
@given(cases())
def test_each_tier_row_sums_to_one_and_has_a_row_per_pool_row(case: Case) -> None:
    out = case.run()
    assert out.shape == (3, len(case.weights))
    assert out.sum(axis=1) == pytest.approx(1.0, abs=1e-9)
    assert (out >= 0).all()


@settings(max_examples=150, deadline=None)
@given(cases())
def test_every_tier_keeps_each_groups_mass(case: Case) -> None:
    out = case.run()
    for group in case.groups():
        mass, _ = mass_and_p(case, group)
        assert out[:, group].sum(axis=1) == pytest.approx(mass, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(cases())
def test_the_slot_weighted_mean_of_the_ten_plus_share_is_the_pool_share(case: Case) -> None:
    out = case.run()
    for group in case.groups():
        mass, p = mass_and_p(case, group)
        if mass == 0:
            continue
        by_tier = out[:, group & case.ten_plus].sum(axis=1) / out[:, group].sum(axis=1)
        assert float(np.dot(case.tier_share, by_tier)) == pytest.approx(p, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(cases())
def test_odds_of_ten_plus_within_a_group_follow_the_ratios(case: Case) -> None:
    out = case.run()
    female = case.pool.sex == FEMALE
    for group in case.groups():
        _, p = mass_and_p(case, group)
        if not 0 < p < 1:
            continue
        ratios = case.ratios_female if female[group].all() else case.ratios_male
        ten = out[:, group & case.ten_plus].sum(axis=1)
        rest = out[:, group & ~case.ten_plus].sum(axis=1)
        odds = ten / rest
        assert odds / odds[1] == pytest.approx(np.array(ratios) / ratios[1], rel=1e-6)


@settings(max_examples=100, deadline=None)
@given(cases(ratios=False))
def test_all_ones_ratios_give_the_input_weights_in_every_tier(case: Case) -> None:
    out = case.run()
    for tier in range(3):
        np.testing.assert_array_equal(out[tier], case.weights)


@settings(max_examples=100, deadline=None)
@given(cases())
def test_a_group_with_no_or_all_ten_plus_is_unchanged_in_every_tier(case: Case) -> None:
    out = case.run()
    for group in case.groups():
        mass, p = mass_and_p(case, group)
        if mass > 0 and p in (0.0, 1.0):
            for tier in range(3):
                np.testing.assert_array_equal(out[tier, group], case.weights[group])


def test_a_group_without_ten_plus_stays_flat_while_the_others_tilt() -> None:
    n = 40
    sex = np.array([FEMALE, MALE] * (n // 2), np.uint8)
    marital = np.zeros(n, np.uint8)
    ten_plus = np.arange(n) % 4 == 1
    weights = np.full(n, 1 / n)
    out = tilted_weights(
        weights,
        pool_of(sex, marital),
        MARRIED_CODE,
        ten_plus,
        (0.4, 0.4, 0.2),
        (0.3, 1.0, 3.0),
        (0.3, 1.0, 3.0),
    )
    for tier in range(3):
        np.testing.assert_array_equal(out[tier, sex == FEMALE], weights[sex == FEMALE])
    assert not np.allclose(out[0], out[2])


def test_a_tier_with_no_slots_does_not_break_the_other_tiers() -> None:
    rng = np.random.default_rng(3)
    n = 30
    pool = pool_of(rng.integers(0, 2, n).astype(np.uint8), rng.integers(0, 2, n).astype(np.uint8))
    ten_plus = rng.random(n) < 0.5
    weights = np.full(n, 1 / n)
    out = tilted_weights(
        weights, pool, MARRIED_CODE, ten_plus, (0.0, 0.7, 0.3), (0.5, 1, 2), (0.5, 1, 2)
    )
    assert out.sum(axis=1) == pytest.approx(1.0)
    assert np.isfinite(out).all()
