import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from personas_fixture import (
    MARRIED,
    NONE,
    Row,
    build_pool,
    decoy_rows,
    household_adults,
    household_runs,
    married_code,
    personas_config,
    realistic_rows,
    uuid_for,
    write_source,
)

from twl_citybake.config import CityConfig, load_city
from twl_citybake.personas.assign import Assignment, assign_personas
from twl_citybake.personas.pool import FEMALE, MALE, PersonaPool, fetch_pool
from twl_citybake.personas.stage import PersonaLayers, run_personas
from twl_citybake.rng import Stream, stream
from twl_citybake.store import write_table

SEED = 7
WEIGHT = 9.0
TIERS = 3
TEN_PLUS = ("Secondary", "Higher Secondary", "Graduate & above")
MERGED_GOLDEN = "ba3a66b9d573894f5f94a5039bdb0997bae1a36ad4d94985093bd10ce4b6f866"

Weights = npt.NDArray[np.float64]


def spouse_rows(ages: range, married_share: float = 1.0, per_age: int = 40) -> list[Row]:
    rng = np.random.default_rng(3)
    rows: list[Row] = []
    for age in ages:
        for sex in ("Female", "Male"):
            for k in range(per_age):
                married = rng.random() < married_share
                rows.append(
                    Row(
                        uuid=uuid_for(len(rows)),
                        sex=sex,
                        age=age,
                        marital_status=MARRIED if married else "Never Married",
                        education_level="Graduate & above" if k % 2 else "Primary",
                    )
                )
    return rows


def is_ten_plus(pool: PersonaPool) -> npt.NDArray[np.bool_]:
    return pool.education == pool.vocab["education"].index("Graduate & above")


def flat(pool: PersonaPool) -> Weights:
    return np.full((TIERS, len(pool.uuid)), 1.0 / len(pool.uuid))


def favour(pool: PersonaPool, tier: int, sex: int, weight: float = WEIGHT) -> Weights:
    table = np.ones((TIERS, len(pool.uuid)))
    table[tier, is_ten_plus(pool) & (pool.sex == sex)] = weight
    return table / table.sum(axis=1, keepdims=True)


def run(
    pool: PersonaPool,
    adults: npt.ArrayLike,
    tier_of_household: npt.NDArray[np.uint8],
    spouse_weights: Weights | None,
    free_weights: Weights | None = None,
    seed: int = SEED,
) -> tuple[Assignment, npt.NDArray[np.uint32], npt.NDArray[np.uint8], np.random.Generator]:
    household = household_runs(adults)
    slot_tier = tier_of_household[household]
    rng = stream(seed, Stream.PERSONAS)
    out = assign_personas(
        household,
        pool,
        np.zeros(len(pool.uuid), np.uint16),
        married_code(pool),
        rng,
        slot_tier=slot_tier,
        free_weights=flat(pool) if free_weights is None else free_weights,
        spouse_weights=spouse_weights,
    )
    return out, household, slot_tier, rng


def spouse_slots(a: Assignment) -> npt.NDArray[np.intp]:
    spouse = a.spouse.astype(np.int64)
    return np.flatnonzero((spouse != NONE) & (spouse < np.arange(len(spouse))))


@dataclass(frozen=True)
class Case:
    pool: PersonaPool
    adults: npt.NDArray[np.int64]
    tier: npt.NDArray[np.uint8]


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    pool = build_pool(tmp_path_factory.mktemp("spouse"), spouse_rows(range(25, 45)))
    adults = np.full(6000, 2)
    tier = np.random.default_rng(1).integers(0, TIERS, len(adults)).astype(np.uint8)
    return Case(pool, adults, tier)


def ten_share(case: Case, a: Assignment, slot_tier: npt.NDArray, tier: int, sex: int) -> float:
    slots = spouse_slots(a)
    slots = slots[slot_tier[slots] == tier]
    rows = a.persona_ref[slots]
    rows = rows[case.pool.sex[rows] == sex]
    assert len(rows) > 300
    return float(is_ten_plus(case.pool)[rows].mean())


def merged_case(root: Path) -> tuple[npt.NDArray[np.uint32], PersonaPool, Weights, npt.NDArray]:
    pool = build_pool(root, spouse_rows(range(20, 70), married_share=0.55))
    household = household_runs(household_adults(1500))
    weights = np.random.default_rng(1).random((TIERS, len(pool.uuid))) + 0.1
    weights /= weights.sum(axis=1, keepdims=True)
    return household, pool, weights, (np.arange(len(household)) % TIERS).astype(np.uint8)


def digest(a: Assignment) -> str:
    return hashlib.sha256(a.persona_ref.tobytes() + a.spouse.tobytes()).hexdigest()


def test_without_spouse_weights_the_assignment_is_the_merged_one(tmp_path: Path) -> None:
    household, pool, weights, slot_tier = merged_case(tmp_path)
    a = assign_personas(
        household,
        pool,
        np.zeros(len(pool.uuid), np.uint16),
        married_code(pool),
        stream(SEED, Stream.PERSONAS),
        slot_tier=slot_tier,
        free_weights=weights,
    )
    assert digest(a) == MERGED_GOLDEN


def test_spouse_weights_none_equals_omitting_them(case: Case) -> None:
    omitted, *_ = run(case.pool, case.adults, case.tier, None)
    household = household_runs(case.adults)
    explicit = assign_personas(
        household,
        case.pool,
        np.zeros(len(case.pool.uuid), np.uint16),
        married_code(case.pool),
        stream(SEED, Stream.PERSONAS),
        slot_tier=case.tier[household],
        free_weights=flat(case.pool),
    )
    np.testing.assert_array_equal(omitted.persona_ref, explicit.persona_ref)
    np.testing.assert_array_equal(omitted.spouse, explicit.spouse)


@pytest.mark.parametrize("sex", [FEMALE, MALE], ids=["female", "male"])
def test_favouring_ten_plus_of_one_sex_in_a_tier_raises_that_sexs_spouses_there(
    case: Case, sex: int
) -> None:
    base, _, slot_tier, _ = run(case.pool, case.adults, case.tier, None)
    tilted, *_ = run(case.pool, case.adults, case.tier, favour(case.pool, 2, sex))
    before = ten_share(case, base, slot_tier, 2, sex)
    after = ten_share(case, tilted, slot_tier, 2, sex)
    assert before == pytest.approx(0.5, abs=0.06)
    assert after == pytest.approx(WEIGHT * before / (WEIGHT * before + 1 - before), abs=0.05)


@pytest.mark.parametrize("sex", [FEMALE, MALE], ids=["female", "male"])
def test_other_sex_and_other_tiers_are_unaffected(case: Case, sex: int) -> None:
    base, _, slot_tier, _ = run(case.pool, case.adults, case.tier, None)
    tilted, *_ = run(case.pool, case.adults, case.tier, favour(case.pool, 2, sex))
    other = FEMALE if sex == MALE else MALE
    for tier, spouse_sex in ((2, other), (0, sex), (1, sex), (0, other), (1, other)):
        assert ten_share(case, tilted, slot_tier, tier, spouse_sex) == pytest.approx(
            ten_share(case, base, slot_tier, tier, spouse_sex), abs=0.02
        )


def test_each_tier_uses_its_own_row_of_weights(case: Case) -> None:
    table = flat(case.pool)
    ten = is_ten_plus(case.pool)
    table[0, ~ten] = 0.0
    table[2, ten] = 0.0
    a, _, slot_tier, _ = run(case.pool, case.adults, case.tier, table)
    for sex in (FEMALE, MALE):
        assert ten_share(case, a, slot_tier, 0, sex) == 1.0
        assert ten_share(case, a, slot_tier, 2, sex) == 0.0
        assert 0.3 < ten_share(case, a, slot_tier, 1, sex) < 0.7


def test_the_picked_spouse_keeps_todays_age_bucket_and_everything_else_unchanged(
    case: Case,
) -> None:
    base, _, _, base_rng = run(case.pool, case.adults, case.tier, None)
    tilted, _, _, rng = run(case.pool, case.adults, case.tier, favour(case.pool, 2, MALE))
    np.testing.assert_array_equal(tilted.spouse, base.spouse)
    slots = spouse_slots(tilted)
    openers = tilted.spouse[slots].astype(np.intp)
    np.testing.assert_array_equal(tilted.persona_ref[openers], base.persona_ref[openers])
    assert (tilted.persona_ref != base.persona_ref).any()
    np.testing.assert_array_equal(
        case.pool.age[tilted.persona_ref[slots]], case.pool.age[base.persona_ref[slots]]
    )
    assert rng.bit_generator.state == base_rng.bit_generator.state


def test_tilted_spouses_are_mutual_married_other_sex_and_in_the_household(case: Case) -> None:
    a, household, _, _ = run(case.pool, case.adults, case.tier, favour(case.pool, 2, MALE))
    paired = np.flatnonzero(a.spouse != NONE)
    partner = a.spouse[paired].astype(np.intp)
    assert len(paired) == len(a.spouse)
    assert (a.spouse[partner] == paired).all()
    assert (household[partner] == household[paired]).all()
    pool = case.pool
    assert (pool.marital[a.persona_ref[paired]] == married_code(pool)).all()
    assert (pool.sex[a.persona_ref[paired]] != pool.sex[a.persona_ref[partner]]).all()


def test_rows_with_zero_weight_are_never_picked_as_spouses(case: Case) -> None:
    table = flat(case.pool)
    table[:, ~is_ten_plus(case.pool)] = 0.0
    a, _, _, _ = run(case.pool, case.adults, case.tier, table)
    assert is_ten_plus(case.pool)[a.persona_ref[spouse_slots(a)]].all()


@pytest.mark.parametrize("positions", [(0, -1), (5, 17), (0, 1)])
def test_weight_on_two_rows_per_bucket_picks_only_those_rows(
    case: Case, positions: tuple[int, int]
) -> None:
    pool = case.pool
    chosen = np.zeros(len(pool.uuid), bool)
    for age in np.unique(pool.age):
        for sex in (FEMALE, MALE):
            bucket = np.flatnonzero((pool.age == age) & (pool.sex == sex))
            chosen[bucket[list(positions)]] = True
    table = np.where(chosen, 1.0, 0.0)[None, :].repeat(TIERS, axis=0)
    a, _, _, _ = run(pool, case.adults, case.tier, table)
    flat_run, _, _, _ = run(pool, case.adults, case.tier, None)
    slots = spouse_slots(a)
    picked = a.persona_ref[slots]
    assert chosen[picked].all()
    assert 0.4 < np.isin(picked, np.flatnonzero(chosen)[::2]).mean() < 0.6
    np.testing.assert_array_equal(pool.age[picked], pool.age[flat_run.persona_ref[slots]])


def test_a_tier_with_all_zero_weights_falls_back_to_the_uniform_pick(case: Case) -> None:
    table = favour(case.pool, 2, MALE)
    table[0] = 0.0
    base, _, slot_tier, _ = run(case.pool, case.adults, case.tier, None)
    a, *_ = run(case.pool, case.adults, case.tier, table)
    slots = spouse_slots(a)
    np.testing.assert_array_equal(
        case.pool.age[a.persona_ref[slots]], case.pool.age[base.persona_ref[slots]]
    )
    for sex in (FEMALE, MALE):
        assert ten_share(case, a, slot_tier, 0, sex) == pytest.approx(
            ten_share(case, base, slot_tier, 0, sex), abs=0.02
        )


def test_a_bucket_with_all_zero_weights_still_gets_a_spouse_of_the_right_age(case: Case) -> None:
    table = flat(case.pool)
    table[1, case.pool.age == 33] = 0.0
    base, *_ = run(case.pool, case.adults, case.tier, None)
    a, *_ = run(case.pool, case.adults, case.tier, table)
    slots = spouse_slots(a)
    np.testing.assert_array_equal(
        case.pool.age[a.persona_ref[slots]], case.pool.age[base.persona_ref[slots]]
    )


def test_the_same_seed_gives_the_same_tilted_assignment(case: Case) -> None:
    table = favour(case.pool, 1, FEMALE)
    a, *_ = run(case.pool, case.adults, case.tier, table)
    b, *_ = run(case.pool, case.adults, case.tier, table)
    np.testing.assert_array_equal(a.persona_ref, b.persona_ref)
    np.testing.assert_array_equal(a.spouse, b.spouse)


def test_a_different_seed_changes_the_tilted_assignment(case: Case) -> None:
    table = favour(case.pool, 1, FEMALE)
    a, *_ = run(case.pool, case.adults, case.tier, table)
    b, *_ = run(case.pool, case.adults, case.tier, table, seed=SEED + 1)
    assert not np.array_equal(a.persona_ref, b.persona_ref)


@pytest.mark.parametrize(
    ("tiers", "short"), [(2, 0), (4, 0), (3, 1)], ids=["two-tiers", "four-tiers", "short-rows"]
)
def test_spouse_weights_of_the_wrong_shape_are_refused(case: Case, tiers: int, short: int) -> None:
    wrong = np.ones((tiers, len(case.pool.uuid) - short))
    with pytest.raises(ValueError):
        run(case.pool, case.adults, case.tier, wrong)


def test_one_dimensional_spouse_weights_are_refused(case: Case) -> None:
    with pytest.raises(ValueError):
        run(case.pool, case.adults, case.tier, np.ones(len(case.pool.uuid)))


def test_spouse_weights_without_slot_tier_are_refused(case: Case) -> None:
    household = household_runs(case.adults)
    with pytest.raises(ValueError):
        assign_personas(
            household,
            case.pool,
            np.zeros(len(case.pool.uuid), np.uint16),
            married_code(case.pool),
            stream(SEED, Stream.PERSONAS),
            free_weights=np.full(len(case.pool.uuid), 1.0 / len(case.pool.uuid)),
            spouse_weights=flat(case.pool),
        )


STAGE_HOUSEHOLDS = 15_000
STAGE_TIER_SHARES = (0.38, 0.42, 0.20)


@dataclass(frozen=True)
class Staged:
    city: CityConfig
    population: Path
    pool_path: Path
    tier: npt.NDArray[np.uint8]
    tilted: PersonaLayers
    plain: PersonaLayers


def write_population(path: Path, tier: npt.NDArray[np.uint8], adults: npt.ArrayLike) -> None:
    household = household_runs(adults)
    write_table(
        path / "residents",
        {
            "household": household,
            "weight": np.ones(len(household), np.float32),
            "tier": tier[household],
        },
    )


def tilted_city() -> CityConfig:
    city = load_city("ahmedabad")
    assert city.personas is not None and city.personas.tier_tilt is not None
    tilt = replace(city.personas.tier_tilt, education_10plus=TEN_PLUS)
    return replace(city, seed=SEED, personas=replace(city.personas, tier_tilt=tilt))


@pytest.fixture(scope="module")
def staged(tmp_path_factory: pytest.TempPathFactory) -> Staged:
    root = tmp_path_factory.mktemp("spouse-stage")
    source = write_source(root / "source.parquet", [*realistic_rows(), *decoy_rows(400)])
    pool_path = fetch_pool(personas_config(), root / "pool.parquet", urls=[str(source)])
    adults = household_adults(STAGE_HOUSEHOLDS)
    tier = np.random.default_rng(17).choice(3, len(adults), p=STAGE_TIER_SHARES).astype(np.uint8)
    write_population(root / "population", tier, adults)
    city = tilted_city()
    assert city.personas is not None
    plain = replace(city, personas=replace(city.personas, tier_tilt=None))
    return Staged(
        city=city,
        population=root / "population",
        pool_path=pool_path,
        tier=tier[household_runs(adults)],
        tilted=run_personas(root / "population", pool_path, city),
        plain=run_personas(root / "population", pool_path, plain),
    )


def stage_shares(staged: Staged, layers: PersonaLayers, sex: int) -> list[float]:
    spouse = layers.assignment["spouse"].astype(np.int64)
    is_spouse = (spouse != NONE) & (spouse < np.arange(len(spouse)))
    rows = layers.assignment["persona_ref"]
    vocab = layers.vocab["education"]
    ten = np.isin(layers.pool["education"][rows], [vocab.index(n) for n in TEN_PLUS])
    chosen = is_spouse & (layers.pool["sex"][rows] == sex)
    out = []
    for tier in range(3):
        cell = chosen & (staged.tier == tier)
        out.append(float(ten[cell].mean()) if cell.any() else 0.0)
    return out


def test_the_stage_report_carries_spouses_ten_plus_by_sex_and_tier(staged: Staged) -> None:
    report = staged.tilted.report["tier_tilt"]
    assert report is not None
    assert set(report["spouses_10plus"]) == {"female", "male"}
    assert all(len(v) == 3 for v in report["spouses_10plus"].values())
    assert {"free_draws_10plus", "residents_10plus", "odds_ratios"} <= set(report)


def test_the_stage_spouses_ten_plus_matches_the_assignment(staged: Staged) -> None:
    report = staged.tilted.report["tier_tilt"]["spouses_10plus"]
    assert report["female"] == pytest.approx(stage_shares(staged, staged.tilted, FEMALE), abs=1e-4)
    assert report["male"] == pytest.approx(stage_shares(staged, staged.tilted, MALE), abs=1e-4)


@pytest.mark.parametrize("sex", ["female", "male"])
def test_stage_spouses_ten_plus_rises_from_low_to_high_tier(staged: Staged, sex: str) -> None:
    low, middle, high = staged.tilted.report["tier_tilt"]["spouses_10plus"][sex]
    assert low < middle < high
    assert high - low > 0.05


@pytest.mark.parametrize("sex", [FEMALE, MALE], ids=["female", "male"])
def test_stage_without_a_tilt_spouses_do_not_differ_by_tier(staged: Staged, sex: int) -> None:
    shares = stage_shares(staged, staged.plain, sex)
    assert max(shares) - min(shares) < 0.06


def test_stage_without_a_tilt_the_report_has_no_tilt_section(staged: Staged) -> None:
    assert staged.plain.report["tier_tilt"] is None


def test_stage_rerun_gives_the_same_spouses_and_report(staged: Staged) -> None:
    again = run_personas(staged.population, staged.pool_path, staged.city)
    np.testing.assert_array_equal(
        again.assignment["persona_ref"], staged.tilted.assignment["persona_ref"]
    )
    assert again.report == staged.tilted.report


def test_stage_spouses_ten_plus_is_zero_for_a_tier_with_no_spouses(
    staged: Staged, tmp_path: Path
) -> None:
    adults = household_adults(3000)
    write_population(tmp_path / "population", np.zeros(len(adults), np.uint8), adults)
    layers = run_personas(tmp_path / "population", staged.pool_path, staged.city)
    cells = layers.report["tier_tilt"]["spouses_10plus"]
    for sex in ("female", "male"):
        assert cells[sex][0] > 0.0
        assert cells[sex][1:] == [0.0, 0.0]
