from functools import lru_cache

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.orders import NUDGE_WINDOW_MINUTES
from twl_engine.run import RunInputs, run_seed, run_seed_bytes
from twl_engine.scenario import Campaign, Params, Scenario
from twl_tape import (
    CHATTER_HEARD,
    CHATTER_NUDGED,
    NEVER,
    SOURCE_NONE,
    SOURCE_PAID,
    SOURCE_WOM,
    read_tape,
)

RESIDENTS = 600
DAYS = 3


@lru_cache(maxsize=1)
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=RESIDENTS, stores=2)


campaigns = st.builds(
    Campaign,
    name=st.just("burst"),
    start_minute=st.sampled_from([0, 300, 1440]),
    end_minute=st.sampled_from([1500, 2880, 4320]),
    budget_per_day=st.floats(0.0, 3.0).map(lambda per_resident: per_resident * RESIDENTS * 0.1),
    cpm=st.just(100.0),
    frequency_cap=st.sampled_from([None, 1, 3]),
)
scenarios = st.builds(
    Scenario,
    days=st.just(DAYS),
    price=st.floats(10.0, 150.0),
    campaigns=st.lists(campaigns, min_size=1, max_size=2).map(tuple),
)
params = st.builds(
    Params,
    q=st.floats(0.0, 0.5),
    hearsay_factor=st.floats(0.0, 1.0),
    talk_gain=st.floats(0.3, 1.5),
    talk_decay_hours=st.sampled_from([None, 6.0, 72.0]),
    awareness_half_life_days=st.sampled_from([None, 0.5, 4.0]),
    alpha_0=st.floats(-6.0, 1.0),
    alpha_rep=st.floats(-4.0, 1.0),
    occasions_per_day=st.floats(0.0, 12.0),
    wom_mixing=st.sampled_from(["graph", "uniform"]),
    order_on_awareness=st.booleans(),
)


@settings(derandomize=True, max_examples=40, deadline=None)
@given(scenario=scenarios, params=params, seed=st.integers(0, 2**32))
def test_every_run_keeps_the_tape_invariants(scenario: Scenario, params: Params, seed: int) -> None:
    inputs = RunInputs(city(), bytes(32), scenario, params, seed)
    data = run_seed_bytes(inputs)
    tape = read_tape(data)
    residents, chatter, aggregates = tape.residents, tape.chatter, tape.aggregates
    aware = residents.aware_at != NEVER
    ordered = residents.first_order_at != NEVER

    assert (residents.aware_at[aware] < scenario.sim_minutes).all()
    assert not (ordered & ~aware).any()
    assert (residents.aware_at[ordered] <= residents.first_order_at[ordered]).all()
    assert (residents.first_order_at[ordered] < scenario.sim_minutes).all()
    assert np.array_equal(ordered, residents.order_count > 0)
    assert int(aggregates.orders.sum()) == int(residents.order_count.sum())
    assert (np.diff(aggregates.aware.astype(np.int64)) >= 0).all()
    assert (np.diff(aggregates.adopted.astype(np.int64)) >= 0).all()
    assert (aggregates.aware[-1], aggregates.adopted[-1]) == (aware.sum(), ordered.sum())
    assert (aggregates.adopted <= aggregates.aware).all()

    by_talk = residents.first_source == SOURCE_WOM
    assert np.array_equal(residents.first_source == SOURCE_NONE, ~aware)
    assert np.array_equal(residents.heard_from != NEVER, by_talk)
    assert np.isin(residents.first_source, [SOURCE_NONE, SOURCE_PAID, SOURCE_WOM]).all()

    heard = chatter.kind == CHATTER_HEARD
    nudged = chatter.kind == CHATTER_NUDGED
    assert (heard | nudged).all() and tape.header.chatter_rows == chatter.at.shape[0]
    assert np.array_equal(np.sort(chatter.to[heard]), np.flatnonzero(by_talk))
    assert np.array_equal(residents.aware_at[chatter.to[heard]], chatter.at[heard])
    assert np.array_equal(residents.heard_from[chatter.to[heard]], chatter.from_[heard])
    assert (chatter.from_ != chatter.to).all()
    talker_aware = residents.aware_at[chatter.from_].astype(np.int64)
    tick_start = chatter.at.astype(np.int64) // scenario.tick_minutes * scenario.tick_minutes
    assert (talker_aware < tick_start).all()

    nudged_to = chatter.to[nudged]
    wait = residents.first_order_at[nudged_to].astype(np.int64) - chatter.at[nudged]
    assert ordered[nudged_to].all() and np.unique(nudged_to).shape == nudged_to.shape
    assert (wait >= 0).all() and (wait <= NUDGE_WINDOW_MINUTES).all()

    order = np.lexsort((chatter.kind, chatter.to, chatter.at))
    assert np.array_equal(order, np.arange(order.shape[0]))
    assert data == run_seed_bytes(inputs)


@settings(derandomize=True, max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**32), days=st.integers(1, 3))
def test_a_shorter_run_is_a_prefix_of_a_longer_one(seed: int, days: int) -> None:
    campaign = Campaign("burst", 0, 2880, RESIDENTS * 0.1, 100.0)
    lively = Params(q=0.1, alpha_0=-2.0)
    short = run_seed(
        RunInputs(city(), bytes(32), Scenario(days=days, campaigns=(campaign,)), lively, seed)
    )
    full = run_seed(
        RunInputs(city(), bytes(32), Scenario(days=4, campaigns=(campaign,)), lively, seed)
    )
    ticks = days * 96
    for name in ("aware", "adopted", "orders"):
        assert np.array_equal(
            getattr(short.aggregates, name), getattr(full.aggregates, name)[:ticks]
        )
    before = full.residents.aware_at < days * 1440
    assert np.array_equal(short.residents.aware_at[before], full.residents.aware_at[before])
    assert (short.residents.aware_at[~before] == NEVER).all()
