from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from personas_fixture import (
    NONE,
    Row,
    build_pool,
    household_runs,
    married_code,
    married_pool_rows,
    uuid_for,
)

from twl_citybake.personas.assign import Assignment, assign_personas, free_draw_weights
from twl_citybake.personas.pool import PersonaPool
from twl_citybake.rng import Stream, stream


def assign(
    adults: npt.ArrayLike, pool: PersonaPool, seed: int = 1
) -> tuple[Assignment, npt.NDArray[np.uint32]]:
    household = household_runs(adults)
    segments = np.zeros(len(pool.uuid), np.uint16)
    rng = stream(seed, Stream.PERSONAS)
    return assign_personas(household, pool, segments, married_code(pool), rng), household


def couples(a: Assignment) -> npt.NDArray[np.intp]:
    return np.flatnonzero(a.spouse != NONE)


@pytest.fixture(scope="module")
def all_married(tmp_path_factory: pytest.TempPathFactory) -> PersonaPool:
    return build_pool(tmp_path_factory.mktemp("allm"), married_pool_rows(range(25, 56)))


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory) -> PersonaPool:
    return build_pool(
        tmp_path_factory.mktemp("mixed"), married_pool_rows(range(20, 70), married_share=0.55)
    )


def test_spouses_are_mutual_married_other_sex_and_in_the_same_household(
    mixed: PersonaPool,
) -> None:
    a, household = assign(np.full(3000, 2), mixed)
    idx = couples(a)
    assert len(idx) > 500
    partner = a.spouse[idx].astype(np.intp)
    assert (a.spouse[partner] == idx).all()
    assert (household[partner] == household[idx]).all()
    code = married_code(mixed)
    ref = a.persona_ref
    assert (mixed.marital[ref[idx]] == code).all()
    assert (mixed.marital[ref[partner]] == code).all()
    assert (mixed.sex[ref[idx]] != mixed.sex[ref[partner]]).all()


def test_every_resident_gets_a_valid_pool_row(mixed: PersonaPool) -> None:
    a, household = assign(household_sizes := np.array([1, 2, 3, 4, 5, 6] * 200), mixed)
    assert len(a.persona_ref) == len(a.spouse) == int(household_sizes.sum()) == len(household)
    assert a.persona_ref.dtype == np.uint32 and a.spouse.dtype == np.uint32
    assert int(a.persona_ref.max()) < len(mixed.uuid)


def test_households_of_one_have_no_spouse(mixed: PersonaPool) -> None:
    a, _ = assign(np.ones(2000, np.int64), mixed)
    assert (a.spouse == NONE).all()


def test_a_household_of_three_holds_at_most_one_couple_and_a_free_third(
    all_married: PersonaPool,
) -> None:
    a, household = assign(np.full(600, 3), all_married)
    in_couple = (a.spouse != NONE).reshape(-1, 3)
    assert (in_couple.sum(axis=1) == 2).all()
    assert not in_couple[:, 2].any()
    assert len(household) == 1800


def test_a_household_of_four_can_hold_two_couples(all_married: PersonaPool) -> None:
    a, _ = assign(np.full(400, 4), all_married)
    spouse = a.spouse.reshape(-1, 4)
    base = np.arange(0, len(a.spouse), 4)[:, None] + np.arange(4)[None, :]
    assert (spouse != NONE).all()
    assert (spouse[:, 0] == base[:, 1]).all()
    assert (spouse[:, 2] == base[:, 3]).all()


def test_mixed_pool_yields_both_one_and_two_couple_households_of_four(
    mixed: PersonaPool,
) -> None:
    a, _ = assign(np.full(1500, 4), mixed)
    per_household = ((a.spouse != NONE).reshape(-1, 4).sum(axis=1)) // 2
    assert {0, 1, 2} <= set(per_household.tolist())


def test_age_gap_is_mostly_zero_to_eight_with_few_reversals(all_married: PersonaPool) -> None:
    a, _ = assign(np.full(20000, 2), all_married)
    age = all_married.age[a.persona_ref].astype(np.int64)
    male = all_married.sex[a.persona_ref] == 1
    first = np.arange(0, len(age), 2)
    first = first[(age[first] >= 33) & (age[first] <= 47)]
    gap = np.where(male[first], age[first] - age[first + 1], age[first + 1] - age[first])
    assert len(gap) > 4000
    forward = float(((gap >= 0) & (gap <= 8)).mean())
    reversed_share = float((gap < 0).mean())
    print(f"realised gap share 0..8: {forward:.3f}, reversed: {reversed_share:.3f}")
    assert 0.91 <= forward <= 0.97
    assert 0.03 <= reversed_share <= 0.09
    assert gap.min() >= -3 and gap.max() <= 8
    assert 2.0 <= float(np.median(gap)) <= 4.0


def test_nearest_age_is_used_when_no_persona_exists_at_the_target(tmp_path: Path) -> None:
    rows = [
        *[Row(uuid_for(i), "Female", 30, "Currently Married") for i in range(60)],
        *[Row(uuid_for(100 + i), "Male", 38, "Currently Married") for i in range(60)],
        *[Row(uuid_for(200 + i), "Male", 50, "Currently Married") for i in range(60)],
    ]
    pool = build_pool(tmp_path, rows)
    a, _ = assign(np.full(3000, 2), pool)
    idx = couples(a)
    assert len(idx) > 1000
    ages = pool.age[a.persona_ref]
    females = pool.sex[a.persona_ref] == 0
    assert (ages[idx][females[idx]] == 30).all()
    assert set(ages[idx][~females[idx]].tolist()) == {38, 50}
    wife_first = idx[females[idx] & (idx % 2 == 0)]
    assert (ages[a.spouse[wife_first].astype(np.intp)] == 38).all()


def test_same_seed_gives_identical_arrays(mixed: PersonaPool) -> None:
    sizes = np.tile([1, 2, 3, 4, 2], 400)
    a, _ = assign(sizes, mixed, seed=7)
    b, _ = assign(sizes, mixed, seed=7)
    assert np.array_equal(a.persona_ref, b.persona_ref)
    assert np.array_equal(a.spouse, b.spouse)


def test_a_different_seed_changes_the_assignment(mixed: PersonaPool) -> None:
    sizes = np.tile([1, 2, 3, 4, 2], 400)
    a, _ = assign(sizes, mixed, seed=7)
    b, _ = assign(sizes, mixed, seed=8)
    assert not np.array_equal(a.persona_ref, b.persona_ref)


def test_free_draw_weights_are_a_distribution_over_pool_rows(mixed: PersonaPool) -> None:
    w = free_draw_weights(np.array([1, 2, 3, 4] * 100), mixed, married_code(mixed))
    assert w.shape == (len(mixed.uuid),)
    assert w.dtype == np.float64
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_free_draw_weights_are_flat_within_each_married_and_sex_class(
    mixed: PersonaPool,
) -> None:
    w = free_draw_weights(np.array([2] * 500), mixed, married_code(mixed))
    married = mixed.marital == married_code(mixed)
    for is_married in (True, False):
        for sex in (0, 1):
            cls = w[(married == is_married) & (mixed.sex == sex)]
            assert cls.max() == pytest.approx(cls.min())


def test_free_draw_weights_favour_fewer_married_when_couples_add_married_residents(
    mixed: PersonaPool,
) -> None:
    code = married_code(mixed)
    married = mixed.marital == code
    singles = free_draw_weights(np.ones(1000, np.int64), mixed, code)[married].sum()
    pairs = free_draw_weights(np.full(1000, 2), mixed, code)[married].sum()
    assert pairs < singles
    assert singles == pytest.approx(float(married.mean()), abs=1e-6)


@pytest.fixture(scope="module")
def share_pools(tmp_path_factory: pytest.TempPathFactory) -> list[PersonaPool]:
    pools = []
    for i, share in enumerate((0.3, 0.5, 0.7)):
        root = tmp_path_factory.mktemp(f"share{i}")
        rows = married_pool_rows(range(20, 70), per_age=8, married_share=share, seed=i)
        pools.append(build_pool(root, rows))
    return pools


@settings(deadline=None, max_examples=25)
@given(
    pool_index=st.integers(0, 2),
    counts=st.lists(st.integers(0, 500), min_size=6, max_size=6),
    seed=st.integers(0, 10_000),
)
def test_free_draw_solver_hits_the_pool_married_and_sex_shares(
    share_pools: list[PersonaPool], pool_index: int, counts: list[int], seed: int
) -> None:
    sizes = np.repeat(np.arange(1, 7), counts)
    assume(len(sizes) >= 600 and int(sizes.sum()) >= 1500)
    pool = share_pools[pool_index]
    code = married_code(pool)
    a, _ = assign(sizes, pool, seed=seed)
    married = pool.marital == code
    resident_married = married[a.persona_ref]
    assert abs(resident_married.mean() - married.mean()) < 0.05
    pool_female = float((pool.sex[married] == 0).mean())
    resident_female = float((pool.sex[a.persona_ref][resident_married] == 0).mean())
    assert abs(resident_female - pool_female) < 0.08
    w = free_draw_weights(sizes, pool, code)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
