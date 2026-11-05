import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city, build_uniform_city
from twl_engine.run import RunInputs, run_seed
from twl_engine.scenario import Campaign, Params, Scenario
from twl_tape import NEVER, SOURCE_NONE, SOURCE_PAID, Tape

RESIDENTS = 20_000
CPM = 100.0
PAID_ONLY = Params(q=0.0, occasions_per_day=0.0, awareness_half_life_days=None)


def budget(impressions_per_resident: float, residents: int = RESIDENTS) -> float:
    return impressions_per_resident * residents * CPM / 1000.0


def flat(per_resident_per_day: float, days: int, **overrides: object) -> Scenario:
    fields: dict[str, object] = {
        "name": "flat",
        "start_minute": 0,
        "end_minute": days * 1440,
        "budget_per_day": budget(per_resident_per_day),
        "cpm": CPM,
    }
    return Scenario(days=days, campaigns=(Campaign(**(fields | overrides)),))  # type: ignore[arg-type]


def run(city: ToyCity, scenario: Scenario, params: Params, seed: int = 1) -> Tape:
    return run_seed(RunInputs(city, bytes(32), scenario, params, seed))


@pytest.fixture(scope="module")
def uniform() -> ToyCity:
    return build_uniform_city(RESIDENTS)


def test_uniform_city_is_identical_residents_without_ties() -> None:
    city = build_uniform_city(1_000)
    assert city.residents.x.shape == (1_000,)
    assert not city.residents.segment.any() and (city.residents.weight == 1.0).all()
    assert not city.residents.building.any() and len(city.districts) == 1
    assert city.ties.targets.shape == (0,) and not city.ties.offsets.any()
    assert city.stores.x.shape == (1,)


def test_one_impression_is_enough_at_gain_one_and_matches_the_exponential(
    uniform: ToyCity,
) -> None:
    per_day = 0.08
    params = Params(
        q=0.0, occasions_per_day=0.0, awareness_half_life_days=None, impression_gain=1.0
    )
    tape = run(uniform, flat(per_day, days=10, frequency_cap=None), params)
    share = tape.aggregates.aware[95::96] / RESIDENTS
    expected = 1.0 - np.exp(-per_day * np.arange(1, 11))
    assert np.abs(share - expected).max() < 0.015


def test_paid_awareness_is_marked_paid_with_no_talker_inside_the_run(uniform: ToyCity) -> None:
    tape = run(uniform, flat(1.0, days=2), PAID_ONLY)
    aware = tape.residents.aware_at != NEVER
    assert 0.2 < aware.mean() < 0.8
    assert (tape.residents.first_source[aware] == SOURCE_PAID).all()
    assert (tape.residents.first_source[~aware] == SOURCE_NONE).all()
    assert (tape.residents.heard_from == NEVER).all()
    assert tape.residents.aware_at[aware].max() < 2 * 1440
    assert tape.header.chatter_rows == 0
    assert np.array_equal(
        tape.aggregates.aware[95::96],
        [(tape.residents.aware_at < day * 1440).sum() for day in (1, 2)],
    )


def test_two_default_impressions_are_needed_so_a_cap_of_one_keeps_everyone_unaware(
    uniform: ToyCity,
) -> None:
    capped = run(uniform, flat(3.0, days=1, frequency_cap=1), PAID_ONLY)
    assert (capped.residents.aware_at == NEVER).all()
    open_cap = run(uniform, flat(3.0, days=1, frequency_cap=3), PAID_ONLY)
    share = (open_cap.residents.aware_at != NEVER).mean()
    assert share == pytest.approx(1.0 - 4.0 * np.exp(-3.0), abs=0.02)


def test_the_cap_resets_each_day(uniform: ToyCity) -> None:
    tape = run(uniform, flat(50.0, days=2, frequency_cap=1), PAID_ONLY)
    aware_at = tape.residents.aware_at
    assert (aware_at >= 1440).all() and (aware_at < 2880).all()


def test_campaign_window_bounds_when_people_become_aware(uniform: ToyCity) -> None:
    scenario = flat(2.0, days=3, start_minute=1440 + 540, end_minute=1440 + 720)
    tape = run(uniform, scenario, PAID_ONLY)
    aware_at = tape.residents.aware_at[tape.residents.aware_at != NEVER]
    assert aware_at.size > 0
    assert aware_at.min() >= 1440 + 540 and aware_at.max() < 1440 + 720


def test_district_and_segment_weights_decide_who_is_reached() -> None:
    city = build_toy_city(seed=49, residents=5_000)
    district = city.buildings.district[city.residents.building]
    only_district_9 = tuple(1.0 if d == 9 else 0.0 for d in range(len(city.districts)))
    campaign = Campaign(
        name="creator",
        start_minute=0,
        end_minute=1440,
        budget_per_day=budget(2.0, 5_000),
        cpm=CPM,
        district_weights=only_district_9,
        segment_weights=(1.0, 1.0, 0.0, 0.0, 0.0),
    )
    tape = run(city, Scenario(days=1, campaigns=(campaign,)), PAID_ONLY)
    aware = tape.residents.aware_at != NEVER
    assert aware.any()
    assert (district[aware] == 9).all()
    assert (city.residents.segment[aware] <= 1).all()


def test_weights_shorter_than_the_city_are_rejected() -> None:
    city = build_toy_city(seed=49, residents=500)
    campaign = Campaign("short", 0, 1440, 10.0, CPM, district_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="district_weights"):
        run(city, Scenario(days=1, campaigns=(campaign,)), PAID_ONLY)


def test_segment_tables_must_cover_the_city() -> None:
    city = build_toy_city(seed=49, residents=500)
    with pytest.raises(ValueError, match="kappa"):
        run(city, Scenario(days=1), Params(kappa=(1.0, 1.0)))


def test_paid_scale_multiplies_reach(uniform: ToyCity) -> None:
    base = run(uniform, flat(0.5, days=1), PAID_ONLY)
    doubled = run(uniform, flat(0.5, days=1), Params(**{**vars(PAID_ONLY), "paid_scale": 2.0}))
    same_as_doubled = run(uniform, flat(1.0, days=1), PAID_ONLY)
    assert doubled.aggregates.aware[-1] > 2 * base.aggregates.aware[-1]
    assert np.array_equal(doubled.residents.aware_at, same_as_doubled.residents.aware_at)


def test_awareness_decay_slows_two_impression_crossings(uniform: ToyCity) -> None:
    kept = run(uniform, flat(0.2, days=10, frequency_cap=1), PAID_ONLY)
    fading = run(
        uniform,
        flat(0.2, days=10, frequency_cap=1),
        Params(q=0.0, occasions_per_day=0.0, awareness_half_life_days=1.0),
    )
    assert fading.aggregates.aware[-1] < kept.aggregates.aware[-1]
