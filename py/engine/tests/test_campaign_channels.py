import dataclasses
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from twl_engine.awareness import paid_exposure
from twl_engine.clock import Clock
from twl_engine.events import EventQueue
from twl_engine.fixtures import ToyCity, build_toy_city, build_uniform_city
from twl_engine.rng import spawn_streams
from twl_engine.run import STREAMS, RunInputs, run_seed
from twl_engine.scenario import CHANNEL_CPM_INR, CHANNELS, Campaign, Params, Scenario
from twl_engine.state import Sim, new_sim
from twl_engine.world import (
    build_paid_plan,
    build_world,
    campaign_presence,
    channel_kappa,
)
from twl_tape import NEVER, Tape

PAID_ONLY = Params(q=0.0, occasions_per_day=0.0, awareness_half_life_days=None)
CPM = 100.0
SEGMENTS = 5
DAY = 1440


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=20_000)


def districts(city: ToyCity) -> int:
    return len(city.districts)


def uneven_districts(count: int) -> tuple[float, ...]:
    return tuple(1.0 + (d % 2) for d in range(count))


def campaign(**overrides: Any) -> Campaign:
    days = overrides.pop("days", 1)
    fields: dict[str, object] = {
        "name": "c",
        "start_minute": 0,
        "end_minute": days * DAY,
        "budget_per_day": 1000.0,
        "channel": "social",
    }
    return Campaign(**(fields | overrides))  # type: ignore[arg-type]


def plan_for(city: ToyCity, campaigns: tuple[Campaign, ...], params: Params, days: int = 1):
    clock = Clock(sim_minutes=days * DAY, tick_minutes=15)
    scenario = Scenario(days=days, campaigns=campaigns)
    return build_paid_plan(city, scenario, params, clock)


def agents_per_impression(city: ToyCity, params: Params) -> float:
    return params.paid_scale / float(city.residents.weight.mean())


def drive(
    city: ToyCity, scenario: Scenario, params: Params, seed: int = 3
) -> tuple[Sim, list[np.ndarray]]:
    """Run only the paid hook; awareness deltas per tick are impressions times gain."""
    clock = Clock(sim_minutes=scenario.sim_minutes, tick_minutes=scenario.tick_minutes)
    streams = spawn_streams(seed, STREAMS)
    sim = new_sim(build_world(city, scenario, params, clock, streams["params"]))
    queue = EventQueue()
    deltas = []
    for tick in range(clock.ticks):
        start = clock.tick_start(tick)
        end = start + clock.tick_minutes
        before = sim.residents.awareness.copy()
        paid_exposure(sim, streams, queue, start, end, queue.pop_until(end - 1))
        deltas.append(sim.residents.awareness - before)
    return sim, deltas


def impressions_by_segment(city: ToyCity, delta: np.ndarray) -> np.ndarray:
    gain = Params().impression_gain * np.asarray(Params().kappa)[city.residents.segment]
    return np.bincount(city.residents.segment, weights=delta / gain, minlength=SEGMENTS)


def six_sigma(expected: float) -> float:
    return 6.0 * np.sqrt(expected) + 1.0


@pytest.mark.parametrize("channel", ["tv", "", "Social", "outdoors"])
def test_unknown_channel_is_refused(channel: str) -> None:
    with pytest.raises(ValueError, match="channel"):
        campaign(channel=channel)


def test_channel_is_required() -> None:
    with pytest.raises(TypeError):
        Campaign("c", 0, DAY, 100.0)  # type: ignore[call-arg]


@pytest.mark.parametrize("channel", CHANNELS)
def test_missing_cpm_falls_back_to_the_channel_default_and_explicit_cpm_wins(
    channel: str,
) -> None:
    targeted: dict[str, object] = {"district_weights": (1.0, 2.0)} if channel == "outdoor" else {}
    default = campaign(channel=channel, budget_per_day=250.0, **targeted)
    assert default.cost_per_mille == CHANNEL_CPM_INR[channel]
    assert default.impressions_per_day == pytest.approx(250.0 / CHANNEL_CPM_INR[channel] * 1000)
    explicit = campaign(channel=channel, budget_per_day=250.0, cpm=77.0, **targeted)
    assert explicit.cost_per_mille == 77.0
    assert explicit.impressions_per_day == pytest.approx(250.0 / 77.0 * 1000)


def test_default_cpms_are_the_sourced_table() -> None:
    assert CHANNEL_CPM_INR == {
        "social": 130.0,
        "video": 110.0,
        "creators": 2500.0,
        "in_app": 125.0,
        "outdoor": 40.0,
    }


@pytest.mark.parametrize("weights", [(), (1.0,), (3.0, 3.0, 3.0), (0.0, 0.0)])
def test_outdoor_without_district_targeting_is_refused(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="outdoor"):
        campaign(channel="outdoor", district_weights=weights)


def test_outdoor_with_uneven_district_weights_is_accepted() -> None:
    assert campaign(channel="outdoor", district_weights=(1.0, 0.0, 1.0)).channel == "outdoor"
    assert campaign(channel="outdoor", district_weights=(1.0, 2.0)).channel == "outdoor"


def test_non_outdoor_channels_need_no_district_targeting() -> None:
    for channel in CHANNELS[:4]:
        assert campaign(channel=channel).district_weights == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"channel_kappa": ((1.0,) * SEGMENTS,) * 4},
        {"channel_kappa": ((1.0,) * SEGMENTS,) * 6},
        {"channel_kappa": ((1.0, 1.0, -0.1, 1.0, 1.0),) + ((1.0,) * SEGMENTS,) * 4},
        {"channel_presence": (((1.0,) * 24,) * SEGMENTS,) * 4},
        {"channel_presence": ((),) * 3 + (((1.0,) * 24,) * SEGMENTS,) + ((),)},
        {"channel_presence": ((((-1.0,) * 24),) * SEGMENTS,) + ((),) * 4},
    ],
)
def test_malformed_channel_tables_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Params(**overrides)  # type: ignore[arg-type]


def test_neutral_channel_tables_are_accepted() -> None:
    assert Params(channel_kappa=((),) * 5, channel_presence=((),) * 5)
    assert Params(channel_kappa=((1.0,) * SEGMENTS,) * 5)


def test_equal_values_of_different_number_types_hash_equally_with_channel_fields() -> None:
    ints = Scenario(
        campaigns=[campaign(budget_per_day=100, cpm=50, district_weights=[1, 2], channel="outdoor")]  # type: ignore[arg-type]
    )
    floats = Scenario(
        campaigns=(
            campaign(
                budget_per_day=100.0, cpm=50.0, district_weights=(1.0, 2.0), channel="outdoor"
            ),
        )
    )
    assert ints.hash() == floats.hash()
    loose = Params(
        channel_kappa=[[1, 2, 1, 1, 1]] * 5,  # type: ignore[arg-type]
        channel_presence=[[[1] * 24] * 5, [], [], [], []],  # type: ignore[arg-type]
    )
    exact = Params(
        channel_kappa=((1.0, 2.0, 1.0, 1.0, 1.0),) * 5,
        channel_presence=(((1.0,) * 24,) * 5, (), (), (), ()),
    )
    assert loose == exact and loose.hash() == exact.hash()


def test_channel_changes_the_scenario_hash() -> None:
    assert (
        Scenario(campaigns=(campaign(channel="social"),)).hash()
        != Scenario(campaigns=(campaign(channel="video"),)).hash()
    )


def test_default_cpm_buys_impressions_in_inverse_proportion_to_price(city: ToyCity) -> None:
    targeting = uneven_districts(districts(city))
    campaigns = tuple(
        campaign(days=2, name=ch, channel=ch, district_weights=targeting, budget_per_day=50_000.0)
        for ch in CHANNELS
    )
    plan = plan_for(city, campaigns, Params(), days=2)
    bought = plan.per_tick.sum(axis=1)
    per_agent = agents_per_impression(city, Params())
    for c, ch in enumerate(CHANNELS):
        assert bought[c] == pytest.approx(2 * 50_000.0 / CHANNEL_CPM_INR[ch] * 1000 * per_agent)
    assert bought[CHANNELS.index("social")] / bought[CHANNELS.index("creators")] == pytest.approx(
        2500.0 / 130.0
    )


def test_channel_kappa_scales_gain_for_that_campaign_and_segment_only(city: ToyCity) -> None:
    campaigns = (campaign(name="a", channel="video"), campaign(name="b", channel="social"))
    neutral = plan_for(city, campaigns, Params())
    row = (1.0, 2.0, 1.0, 1.0, 1.0)
    boosted_params = Params(channel_kappa=((), row, (), (), ()))
    boosted = plan_for(city, campaigns, boosted_params)
    segment = city.residents.segment
    hit = segment == 1
    assert hit.any() and (~hit).any()
    assert np.array_equal(boosted.gain[0][hit], 2.0 * neutral.gain[0][hit])
    assert np.array_equal(boosted.gain[0][~hit], neutral.gain[0][~hit])
    assert np.array_equal(boosted.gain[1], neutral.gain[1])
    assert np.array_equal(boosted.per_tick, neutral.per_tick)
    assert np.array_equal(boosted.cum_target, neutral.cum_target)


def test_neutral_channel_kappa_gain_is_the_pre_channel_formula_bitwise(city: ToyCity) -> None:
    params = Params(impression_gain=0.37)
    expected = params.impression_gain * np.asarray(params.kappa)[city.residents.segment]
    for ch in CHANNELS:
        targeted = (
            {"district_weights": uneven_districts(districts(city))} if ch == "outdoor" else {}
        )
        plan = plan_for(city, (campaign(channel=ch, **targeted),), params)
        assert np.array_equal(plan.gain[0], expected)
    explicit = Params(impression_gain=0.37, channel_kappa=((1.0,) * SEGMENTS,) * 5)
    plan = plan_for(city, (campaign(),), explicit)
    assert np.array_equal(plan.gain[0], expected)
    assert np.array_equal(channel_kappa(campaign(), Params(), SEGMENTS), np.ones(SEGMENTS))


def test_doubled_channel_kappa_makes_one_impression_enough() -> None:
    uniform = build_uniform_city(20_000)
    scenario = Scenario(
        days=1, campaigns=(campaign(budget_per_day=0.5 * 20_000 * CPM / 1000, cpm=CPM),)
    )

    def aware_share(params: Params) -> float:
        tape = run_seed(RunInputs(uniform, bytes(32), scenario, params, 1))
        return float((tape.residents.aware_at != NEVER).mean())

    neutral = aware_share(PAID_ONLY)
    doubled = aware_share(
        dataclasses.replace(PAID_ONLY, **{"channel_kappa": ((2.0,) * SEGMENTS,) * 5})
    )
    assert neutral == pytest.approx(1 - 1.5 * np.exp(-0.5), abs=0.02)
    assert doubled == pytest.approx(1 - np.exp(-0.5), abs=0.02)


def morning_evening_params(own: tuple[int, int], other: tuple[int, int]) -> Params:
    def hours(window: tuple[int, int]) -> tuple[float, ...]:
        return tuple(1.0 if window[0] <= h < window[1] else 0.0 for h in range(24))

    rows = (hours(own), hours(other)) + ((hours((0, 0)),) * 3)
    return dataclasses.replace(PAID_ONLY, **{"channel_presence": (rows, (), (), (), ())})


def test_presence_that_differs_by_segment_steers_impressions_by_time_of_day(
    city: ToyCity,
) -> None:
    morning, evening = (6, 12), (18, 24)
    params = morning_evening_params(morning, evening)
    params = dataclasses.replace(params, theta_aware=1e9)
    weights = (1.0, 2.0, 0.0, 0.0, 0.0)
    scenario = Scenario(
        days=1,
        campaigns=(
            campaign(segment_weights=weights, frequency_cap=None, budget_per_day=30_000.0, cpm=CPM),
        ),
    )
    sim, deltas = drive(city, scenario, params)
    segment = city.residents.segment
    n_a, n_b = (int((segment == s).sum()) for s in (0, 1))
    audience_a, audience_b = n_a * 1.0, n_b * 2.0
    total = scenario.campaigns[0].impressions_per_day * agents_per_impression(city, params)
    expected_a = total * audience_a / (audience_a + audience_b)

    per_tick = np.array([impressions_by_segment(city, d) for d in deltas])
    hour = np.arange(96) // 4
    in_morning = (hour >= 6) & (hour < 12)
    in_evening = hour >= 18
    assert per_tick[:, 2:].sum() == pytest.approx(0.0, abs=1e-6)
    assert per_tick[in_morning, 1].sum() == pytest.approx(0.0, abs=1e-6)
    assert per_tick[in_evening, 0].sum() == pytest.approx(0.0, abs=1e-6)
    assert per_tick[~in_morning & ~in_evening].sum() == pytest.approx(0.0, abs=1e-6)
    got_a = per_tick[in_morning, 0].sum()
    assert abs(got_a - expected_a) < six_sigma(expected_a)
    got_b = per_tick[in_evening, 1].sum()
    assert abs(got_b - (total - expected_a)) < six_sigma(total - expected_a)
    assert sim.residents.awareness.sum() > 0


def test_same_presence_for_every_segment_leaves_landing_split_to_targeting(
    city: ToyCity,
) -> None:
    daytime = tuple(0.2 if h < 8 else (1.0 if h < 20 else 3.0) for h in range(24))
    params = dataclasses.replace(
        PAID_ONLY, **{"channel_presence": ((daytime,) * SEGMENTS, (), (), (), ())}
    )
    params = dataclasses.replace(params, theta_aware=1e9)
    weights = (1.0, 1.0, 3.0, 0.0, 0.5)
    scenario = Scenario(
        days=1,
        campaigns=(
            campaign(segment_weights=weights, frequency_cap=None, budget_per_day=40_000.0, cpm=CPM),
        ),
    )
    plan = build_paid_plan(city, scenario, params, Clock(DAY, 15))
    assert not plan.two_stage.any()
    _, deltas = drive(city, scenario, params)
    segment = city.residents.segment
    audience = np.array([(segment == s).sum() * weights[s] for s in range(SEGMENTS)])
    total = scenario.campaigns[0].impressions_per_day * agents_per_impression(city, params)
    got = np.sum([impressions_by_segment(city, d) for d in deltas], axis=0)
    expected = total * audience / audience.sum()
    assert got[3] == pytest.approx(0.0, abs=1e-6)
    for s in range(SEGMENTS):
        assert abs(got[s] - expected[s]) < six_sigma(expected[s])
    shares = plan.per_tick[0] / plan.per_tick[0].sum()
    hours = np.arange(96) // 4
    assert shares[hours >= 20].sum() == pytest.approx(4 * 3.0 / (8 * 0.2 + 12 * 1.0 + 4 * 3.0))


def test_in_app_spreads_a_day_over_ticks_by_the_occasion_profile(city: ToyCity) -> None:
    profile = tuple(float(1 + (h % 5) ** 2) for h in range(24))
    params = Params(occasion_profile=profile)
    item = campaign(days=2, channel="in_app", budget_per_day=10_000.0)
    plan = plan_for(city, (item,), params, days=2)
    hours = np.arange(2 * 96) % 96 // 4
    per_day = item.impressions_per_day * agents_per_impression(city, params)
    expected = per_day * np.asarray(profile)[hours] / (4 * sum(profile))
    assert np.allclose(plan.per_tick[0], expected, rtol=1e-12, atol=0)
    assert plan.per_tick[0][:96].sum() == pytest.approx(per_day)


def test_in_app_own_presence_multiplies_the_occasion_profile(city: ToyCity) -> None:
    profile = tuple(float(1 + (h % 5) ** 2) for h in range(24))
    own = tuple(0.0 if h < 12 else float(h - 10) for h in range(24))
    params = Params(occasion_profile=profile)
    item = campaign(channel="in_app", budget_per_day=10_000.0, presence=own)
    plan = plan_for(city, (item,), params)
    weight = np.asarray(own) * np.asarray(profile)
    expected = np.repeat(weight, 4) / (4 * weight.sum())
    per_day = item.impressions_per_day * agents_per_impression(city, params)
    assert np.allclose(plan.per_tick[0], per_day * expected, rtol=1e-12, atol=0)
    assert plan.per_tick[0][:48].sum() == 0.0
    assert np.allclose(campaign_presence(item, params, SEGMENTS), weight[None, :] * np.ones((5, 1)))


def test_in_app_ignores_other_channels_presence_tables(city: ToyCity) -> None:
    quiet_night = tuple(0.0 if h < 6 else 1.0 for h in range(24))
    tables = Params(channel_presence=((quiet_night,) * SEGMENTS, (), (), (), ()))
    item = campaign(channel="in_app")
    assert np.array_equal(
        plan_for(city, (item,), tables).per_tick, plan_for(city, (item,), Params()).per_tick
    )


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    caps=st.tuples(st.integers(1, 5), st.integers(1, 5)),
    per_resident=st.tuples(st.floats(0.2, 15.0), st.floats(0.2, 15.0)),
    channels=st.tuples(st.sampled_from(CHANNELS), st.sampled_from(CHANNELS)),
    seed=st.integers(0, 2**32 - 1),
)
def test_daily_frequency_cap_holds_per_campaign(
    caps: tuple[int, int],
    per_resident: tuple[float, float],
    channels: tuple[str, str],
    seed: int,
) -> None:
    small = build_toy_city(seed=49, residents=600)
    targeting = uneven_districts(districts(small))
    items = tuple(
        campaign(
            name=f"c{i}",
            channel=channels[i],
            frequency_cap=caps[i],
            budget_per_day=per_resident[i] * 600 * CPM / 1000,
            cpm=CPM,
            district_weights=targeting,
        )
        for i in range(2)
    )
    scenario = Scenario(days=1, campaigns=items)
    params = dataclasses.replace(PAID_ONLY, theta_aware=1e9)
    sim, _ = drive(small, scenario, params, seed=seed % 1000)
    seen = sim.residents.impressions_today
    for i in range(2):
        assert seen[i].max() <= caps[i]
        assert seen[i].sum() <= 600 * caps[i]
    if min(per_resident) >= 12.0 and max(caps) <= 2:
        assert all(seen[i].max() == caps[i] for i in range(2))


def test_cap_counts_each_campaign_separately(city: ToyCity) -> None:
    items = (
        campaign(name="a", frequency_cap=1, budget_per_day=60_000.0, cpm=CPM),
        campaign(name="b", frequency_cap=2, budget_per_day=60_000.0, cpm=CPM),
    )
    params = dataclasses.replace(PAID_ONLY, theta_aware=1e9)
    sim, _ = drive(city, Scenario(days=1, campaigns=items), params)
    seen = sim.residents.impressions_today
    assert seen[0].max() == 1 and seen[1].max() == 2
    assert (seen[0] == 1).mean() > 0.9 and (seen[1] == 2).mean() > 0.9


def section_fields(tape: Tape) -> list[tuple[str, np.ndarray]]:
    out = []
    for name in ("residents", "chatter", "deliveries", "stores", "aggregates"):
        section = getattr(tape, name)
        out += [(f"{name}.{f.name}", getattr(section, f.name)) for f in dataclasses.fields(section)]
    return out


def test_equal_cpm_social_video_and_creators_give_identical_tapes_but_for_the_header() -> None:
    small = build_toy_city(seed=49, residents=4_000)
    params = Params()

    def run(channel: str) -> Tape:
        item = campaign(channel=channel, budget_per_day=4_000 * 2.0 * CPM / 1000, cpm=CPM)
        scenario = Scenario(days=6, campaigns=(item,))
        return run_seed(RunInputs(small, bytes(32), scenario, params, 5))

    tapes = {channel: run(channel) for channel in ("social", "video", "creators")}
    base = tapes["social"]
    assert (base.residents.aware_at != NEVER).any()
    for channel in ("video", "creators"):
        other = tapes[channel]
        assert other.header.scenario_hash != base.header.scenario_hash
        assert other.header.run_id != base.header.run_id
        same_header = dataclasses.replace(
            other.header, scenario_hash=base.header.scenario_hash, run_id=base.header.run_id
        )
        assert same_header == base.header
        for (name, left), (_, right) in zip(
            section_fields(base), section_fields(other), strict=True
        ):
            assert np.array_equal(left, right), name
