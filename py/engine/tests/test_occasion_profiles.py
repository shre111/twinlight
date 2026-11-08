"""P-031 (D-036): occasion rates and hourly weights per segment, weekends and the calendar.

Observable: every resident is made aware in tick 0 and every occasion turns into an order
(very high `alpha_0` and `alpha_rep`), so a resident's order count is its occasion count and
`aggregates.orders[tick]` is Binomial(n, p_tick) over residents with an occasion. Expected counts
come from a reference model written here from D-036 and MODEL.md section 4, using zoneinfo for the
calendar.

Statistical bounds: counts are sums of independent Bernoulli draws, so the standard deviation is
sqrt(sum n p (1 - p)). Every comparison allows 6 sd (two-sided normal tail 2e-9); the file makes
under 60 such comparisons, so the false failure rate is under 1e-6 at any seed, and the seeds are
fixed anyway.
"""

import dataclasses
import tomllib
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from twl_engine.clock import Clock
from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.goldens import PROTOTYPE_SCENARIO_PATH
from twl_engine.run import RunInputs, run_seed
from twl_engine.scenario import PROTOTYPE_TOD, Campaign, Params, Scenario, load_scenario
from twl_engine.world import build_world
from twl_tape import Tape

SEGMENTS = 5
DAY = 1440
TICK = 15
TICKS_PER_DAY = DAY // TICK
THURSDAY = "2026-11-12"
SIX_SIGMA = 6.0
HOUR_7 = tuple(1.0 if h == 7 else 0.0 for h in range(24))


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=6_000)


def group_sizes(city: ToyCity) -> np.ndarray:
    return np.bincount(city.residents.segment, minlength=SEGMENTS)


def everyone_aware(days: int, start_date: str | None) -> Scenario:
    seeding = Campaign(
        name="seed",
        start_minute=0,
        end_minute=TICK,
        budget_per_day=20_000.0,
        cpm=1.0,
        frequency_cap=None,
        channel="social",
    )
    return Scenario(days=days, campaigns=(seeding,), start_date=start_date)


def ordering(**tables: object) -> Params:
    return Params(
        q=0.0,
        alpha_0=60.0,
        alpha_rep=60.0,
        awareness_half_life_days=None,
        kappa=(1.0,) * SEGMENTS,
        **tables,  # type: ignore[arg-type]
    )


def run(city: ToyCity, scenario: Scenario, params: Params, seed: int = 1) -> Tape:
    tape = run_seed(RunInputs(city, bytes(32), scenario, params, seed))
    assert (tape.residents.aware_at < TICK).all(), "seeding campaign must reach every resident"
    return tape


def local_clock(start_date: str, zone: str, minute: int) -> datetime:
    midnight = datetime.combine(date.fromisoformat(start_date), time(0), ZoneInfo(zone))
    return (midnight.astimezone(UTC) + timedelta(minutes=minute)).astimezone(ZoneInfo(zone))


def tick_probabilities(
    params: Params,
    ticks: int,
    start_date: str | None,
    zone: str = "Asia/Kolkata",
    weekend_days: tuple[int, ...] = (6, 7),
) -> np.ndarray:
    """(ticks, SEGMENTS): 1 - exp(-rate dt), rate = daily / 24 * profile[h] / mean(profile)."""
    rates = params.segment_occasions_per_day or (params.occasions_per_day,) * SEGMENTS
    factors = params.weekend_occasion_factor or (1.0,) * SEGMENTS
    out = np.zeros((ticks, SEGMENTS))
    for s in range(SEGMENTS):
        working = np.asarray(
            (params.segment_occasion_profile or ((),) * SEGMENTS)[s] or params.occasion_profile
        )
        weekend = np.asarray(
            (params.weekend_occasion_profile or ((),) * SEGMENTS)[s] or tuple(working)
        )
        for t in range(ticks):
            minute = t * TICK
            if start_date is None:
                hour, is_weekend = (minute % DAY) // 60, False
            else:
                local = local_clock(start_date, zone, minute)
                hour, is_weekend = local.hour, local.isoweekday() in weekend_days
            daily = rates[s] * (factors[s] if is_weekend else 1.0)
            profile = weekend if is_weekend else working
            if daily > 0.0:
                per_hour = daily / 24.0 * profile[hour] / profile.mean()
                out[t, s] = -np.expm1(-per_hour * TICK / 60.0)
    return out


def assert_within_six_sd(observed: float, n: int, p: np.ndarray) -> None:
    mean = n * float(p.sum())
    sd = float(np.sqrt(n * (p * (1.0 - p)).sum()))
    assert abs(observed - mean) <= SIX_SIGMA * sd + 1.0, (observed, mean, sd)


def counts_by_segment(city: ToyCity, tape: Tape) -> np.ndarray:
    return np.bincount(
        city.residents.segment, weights=tape.residents.order_count, minlength=SEGMENTS
    )


def only_segment(s: int, rate: float) -> tuple[float, ...]:
    return tuple(rate if i == s else 0.0 for i in range(SEGMENTS))


def rebuild_world(city: ToyCity, scenario: Scenario, params: Params) -> None:
    clock = Clock(sim_minutes=scenario.sim_minutes, tick_minutes=scenario.tick_minutes)
    build_world(city, scenario, params, clock, np.random.default_rng(0))


@pytest.mark.parametrize(
    "start_date", ["2026-13-01", "2026-02-30", "11/12/2026", "12-11-2026", "tomorrow", ""]
)
def test_an_invalid_start_date_is_refused(start_date: str) -> None:
    with pytest.raises(ValueError):
        Scenario(start_date=start_date)


def test_a_valid_start_date_is_kept_and_none_is_the_default() -> None:
    assert Scenario().start_date is None
    assert Scenario(start_date=THURSDAY).start_date == THURSDAY


@pytest.mark.parametrize(
    "tables",
    [
        {"segment_occasions_per_day": (1.0,) * 4},
        {"segment_occasions_per_day": (1.0,) * 6},
        {"segment_occasion_profile": ((),) * 4},
        {"segment_occasion_profile": ((1.0,) * 24,) * 3 + ((),)},
        {"segment_occasion_profile": ((1.0,) * 23,) * SEGMENTS},
        {"segment_occasion_profile": ((),) * 4 + ((1.0,) * 23,)},
        {"weekend_occasion_factor": (1.0,) * 4},
        {"weekend_occasion_profile": ((),) * 4},
        {"weekend_occasion_profile": ((),) * 4 + ((1.0,) * 23,)},
        {"segment_occasions_per_day": (1.0, 1.0, -0.1, 1.0, 1.0)},
        {"segment_occasion_profile": ((),) * 4 + ((1.0,) * 23 + (-1.0,),)},
        {"weekend_occasion_factor": (1.0, 1.0, 1.0, 1.0, -2.0)},
        {"weekend_occasion_profile": ((),) * 4 + ((-1.0,) + (1.0,) * 23,)},
    ],
    ids=lambda tables: next(iter(tables)) + "-" + str(len(next(iter(tables.values())))),
)
def test_malformed_or_negative_tables_are_refused(city: ToyCity, tables: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        rebuild_world(city, Scenario(days=1), Params(**tables))  # type: ignore[arg-type]


def test_an_unknown_time_zone_is_refused() -> None:
    uniform = build_toy_city(seed=49, residents=500)
    bad = dataclasses.replace(uniform, timezone="Mars/Olympus_Mons")
    with pytest.raises(ValueError):
        run_seed(RunInputs(bad, bytes(32), Scenario(days=1, start_date=THURSDAY), Params(), 1))


def test_toy_city_defaults_to_kolkata_and_a_saturday_sunday_weekend(city: ToyCity) -> None:
    assert city.timezone == "Asia/Kolkata"
    assert city.weekend_days == (6, 7)


def test_equal_values_of_different_number_types_hash_equally() -> None:
    ints = Params(
        segment_occasions_per_day=(1, 2, 3, 4, 5),
        segment_occasion_profile=((),) * 4 + ((1,) * 24,),
        weekend_occasion_factor=(1, 2, 3, 1, 1),
        weekend_occasion_profile=((1,) * 24,) + ((),) * 4,
    )
    floats = Params(
        segment_occasions_per_day=(1.0, 2.0, 3.0, 4.0, 5.0),
        segment_occasion_profile=((),) * 4 + ((1.0,) * 24,),
        weekend_occasion_factor=(1.0, 2.0, 3.0, 1.0, 1.0),
        weekend_occasion_profile=((1.0,) * 24,) + ((),) * 4,
    )
    assert ints.hash() == floats.hash()
    assert Scenario(start_date=THURSDAY).hash() == Scenario(start_date=THURSDAY).hash()


def test_the_new_fields_are_part_of_the_hashes() -> None:
    assert Scenario(start_date=THURSDAY).hash() != Scenario().hash()
    assert Scenario(start_date=THURSDAY).hash() != Scenario(start_date="2026-11-13").hash()
    base = Params().hash()
    assert Params(segment_occasions_per_day=(3.0,) * 5).hash() != base
    assert Params(weekend_occasion_factor=(1.0,) * 5).hash() != base


def test_neutral_tables_and_a_calendar_leave_every_tape_section_unchanged(city: ToyCity) -> None:
    lively = Params(q=0.06, alpha_0=-2.0)
    plain = Scenario(days=5, campaigns=everyone_aware(5, None).campaigns)
    dated = dataclasses.replace(plain, start_date=THURSDAY)
    neutral = dataclasses.replace(lively, weekend_occasion_factor=(1.0,) * SEGMENTS)
    before = run_seed(RunInputs(city, bytes(32), plain, lively, 7))
    after = run_seed(RunInputs(city, bytes(32), dated, neutral, 7))
    assert before.aggregates.orders.sum() > 1_000
    for section in ("residents", "chatter", "deliveries", "stores", "aggregates"):
        a, b = getattr(before, section), getattr(after, section)
        for field in dataclasses.fields(a):
            assert np.array_equal(getattr(a, field.name), getattr(b, field.name)), field
    differing = {
        f.name
        for f in dataclasses.fields(before.header)
        if getattr(before.header, f.name) != getattr(after.header, f.name)
    }
    assert differing == {"scenario_hash", "run_id"}


def test_the_prototype_fixture_pins_its_occasion_profile() -> None:
    raw = tomllib.loads(PROTOTYPE_SCENARIO_PATH.read_text())["params"]
    assert raw["occasions_per_day"] == 3.0
    assert tuple(raw["occasion_profile"]) == PROTOTYPE_TOD
    _, params = load_scenario(PROTOTYPE_SCENARIO_PATH)
    assert params.occasions_per_day == 3.0
    assert params.occasion_profile == PROTOTYPE_TOD
    assert not params.segment_occasions_per_day and not params.weekend_occasion_factor


def test_segments_with_daily_rates_one_and_six_order_in_that_proportion(city: ToyCity) -> None:
    rates = (1.0, 6.0, 0.0, 0.0, 0.0)
    params = ordering(segment_occasions_per_day=rates)
    days = 12
    tape = run(city, everyone_aware(days, None), params)
    p = tick_probabilities(params, days * TICKS_PER_DAY, None)
    sizes, observed = group_sizes(city), counts_by_segment(city, tape)
    for s in range(SEGMENTS):
        assert_within_six_sd(observed[s], int(sizes[s]), p[:, s])
    assert observed[2:].sum() == 0
    assert observed[1] / sizes[1] > 4.0 * observed[0] / sizes[0]


def test_a_profile_that_is_zero_but_one_hour_orders_only_in_that_hour(city: ToyCity) -> None:
    rows = (HOUR_7,) + ((),) * (SEGMENTS - 1)
    params = ordering(segment_occasions_per_day=only_segment(0, 3.0), segment_occasion_profile=rows)
    days = 6
    tape = run(city, everyone_aware(days, None), params)
    orders = tape.aggregates.orders.astype(np.int64)
    hour = (np.arange(orders.shape[0]) % TICKS_PER_DAY) * TICK // 60
    assert orders[hour != 7].sum() == 0
    p = tick_probabilities(params, orders.shape[0], None)[:, 0]
    assert_within_six_sd(float(orders.sum()), int(group_sizes(city)[0]), p)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_occasion_counts_match_their_rates_when_segments_differ(city: ToyCity, seed: int) -> None:
    shapes = tuple(tuple(1.0 + (h + 5 * s) % 7 for h in range(24)) for s in range(SEGMENTS))
    params = ordering(
        segment_occasions_per_day=(0.5, 1.0, 2.0, 4.0, 1.5),
        segment_occasion_profile=shapes,
        weekend_occasion_factor=(1.0, 2.0, 0.5, 1.5, 3.0),
        weekend_occasion_profile=((),) * 3 + (HOUR_7, shapes[0]),
    )
    days = 7
    tape = run(city, everyone_aware(days, THURSDAY), params, seed)
    p = tick_probabilities(params, days * TICKS_PER_DAY, THURSDAY)
    sizes, observed = group_sizes(city), counts_by_segment(city, tape)
    for s in range(SEGMENTS):
        assert_within_six_sd(observed[s], int(sizes[s]), p[:, s])
    assert int(tape.aggregates.orders.sum()) == int(observed.sum())


def daily_orders(tape: Tape) -> np.ndarray:
    return tape.aggregates.orders.astype(np.int64).reshape(-1, TICKS_PER_DAY).sum(axis=1)


@pytest.mark.parametrize(
    ("weekend_days", "boosted"), [((6, 7), (2, 3)), ((5, 6), (1, 2))], ids=["sat-sun", "fri-sat"]
)
def test_weekend_factor_lands_on_the_citys_weekend_days(
    city: ToyCity, weekend_days: tuple[int, ...], boosted: tuple[int, ...]
) -> None:
    params = ordering(
        segment_occasions_per_day=only_segment(0, 2.0),
        weekend_occasion_factor=(3.0, 1.0, 1.0, 1.0, 1.0),
    )
    days = 7
    variant = dataclasses.replace(city, weekend_days=weekend_days)
    tape = run(variant, everyone_aware(days, THURSDAY), params)
    p = tick_probabilities(params, days * TICKS_PER_DAY, THURSDAY, weekend_days=weekend_days)
    size, per_day = int(group_sizes(city)[0]), daily_orders(tape)
    assert size > 1_000
    for day in range(days):
        window = slice(day * TICKS_PER_DAY, (day + 1) * TICKS_PER_DAY)
        assert_within_six_sd(float(per_day[day]), size, p[window, 0])
    others = [d for d in range(days) if d not in boosted]
    assert min(per_day[list(boosted)]) > 2 * max(per_day[others])


def test_without_a_start_date_every_day_is_a_working_day(city: ToyCity) -> None:
    params = ordering(
        segment_occasions_per_day=only_segment(0, 2.0),
        weekend_occasion_factor=(5.0, 1.0, 1.0, 1.0, 1.0),
    )
    per_day = daily_orders(run(city, everyone_aware(7, None), params))
    assert per_day.max() < 1.5 * per_day.min()


def hit_ticks(tape: Tape) -> list[int]:
    orders = tape.aggregates.orders
    return np.flatnonzero(orders > 0.5 * tape.residents.aware_at.shape[0]).tolist()


def hour_7_ticks(start_date: str, zone: str, ticks: int) -> list[int]:
    return [t for t in range(ticks) if local_clock(start_date, zone, t * TICK).hour == 7]


@pytest.mark.parametrize(
    ("zone", "start_date"),
    [("Europe/London", "2026-10-24"), ("Asia/Kolkata", "2026-10-24")],
)
def test_an_hour_profile_follows_local_time_across_a_clock_change(
    city: ToyCity, zone: str, start_date: str
) -> None:
    days = 3
    params = ordering(
        segment_occasions_per_day=(24.0,) * SEGMENTS,
        segment_occasion_profile=(HOUR_7,) * SEGMENTS,
    )
    variant = dataclasses.replace(city, timezone=zone)
    tape = run(variant, everyone_aware(days, start_date), params)
    expected = hour_7_ticks(start_date, zone, days * TICKS_PER_DAY)
    assert hit_ticks(tape) == expected
    assert int(tape.aggregates.orders.sum()) == pytest.approx(
        len(expected) * city.residents.x.shape[0], rel=0.01
    )
    naive = [d * TICKS_PER_DAY + t for d in range(days) for t in range(28, 32)]
    if zone == "Asia/Kolkata":
        assert expected == naive
    else:
        assert expected[:4] == naive[:4]
        assert expected[4:] == [t + 4 for t in naive[4:]]
