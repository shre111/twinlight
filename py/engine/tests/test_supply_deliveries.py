import dataclasses
from functools import lru_cache

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from twl_engine.clock import Clock
from twl_engine.events import NO_EVENTS, EventQueue
from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.fixtures.types import Residents, Stores, Ties
from twl_engine.run import RunInputs, SupplyStats, run_seed_bytes, run_seed_with_stats
from twl_engine.scenario import Campaign, Params, Scenario, Supply
from twl_engine.state import (
    DELIVERY_ROWS,
    INITIAL_DELIVERY_ROWS,
    LATE,
    Sim,
    new_delivery_log,
    new_sim,
)
from twl_engine.supply import (
    SERVED_FIRST,
    AttemptDraws,
    attempts_from_occasions,
    draw_for_attempts,
    pick_and_ride,
    serve_attempts,
    store_quotes,
)
from twl_engine.world import NO_STORE, SupplyPlan, build_supply_plan, build_world
from twl_tape import (
    DELIVERY_LATE,
    DELIVERY_SPILLED,
    FLAG_GAVE_UP,
    FLAG_SPILLED,
    FLAG_STOCKOUT,
    NEVER,
    Tape,
    read_tape,
)

TICK = 15
FOUR_PER_STORE = 2.0
WALKING_PACE_KMH = 6.0
TOWN_RESIDENTS = 1200
TOWN_STORES = 14


@lru_cache(maxsize=1)
def town() -> ToyCity:
    return build_toy_city(seed=49, residents=TOWN_RESIDENTS, stores=TOWN_STORES)


def line_city() -> ToyCity:
    """Stores at x = 0 and 1000 with a 600 m radius. Residents 0 and 1 are 100 and 200 m from
    store 0; residents 2 and 3 are 100 and 500 m from store 1. At 6 km/h that is 1, 2, 1 and 5
    minutes, and 9 minutes for residents 0 and 2 from their spill stores."""
    resident_x = [100.0, 200.0, 900.0, 1500.0]
    n = len(resident_x)
    residents = Residents(
        x=np.array(resident_x, dtype=np.float32),
        y=np.zeros(n, dtype=np.float32),
        z=np.zeros(n, dtype=np.float32),
        building=np.zeros(n, dtype=np.uint32),
        hex=np.zeros(n, dtype=np.uint32),
        segment=np.zeros(n, dtype=np.uint16),
        persona_ref=np.zeros(n, dtype=np.uint32),
        weight=np.ones(n, dtype=np.float32),
    )
    stores = Stores(
        x=np.array([0.0, 1000.0], dtype=np.float32),
        y=np.zeros(2, dtype=np.float32),
        hex=np.zeros(2, dtype=np.uint32),
        radius_m=np.full(2, 600.0, dtype=np.float32),
        district=np.zeros(2, dtype=np.uint16),
    )
    ties = Ties(
        offsets=np.zeros(n + 1, dtype=np.uint32),
        targets=np.zeros(0, dtype=np.uint32),
        kind=np.zeros(0, dtype=np.uint8),
        strength=np.zeros(0, dtype=np.float32),
    )
    return dataclasses.replace(town(), residents=residents, stores=stores, ties=ties)


def bench(params: Params | None = None, days: int = 2, **supply: float) -> tuple[Sim, EventQueue]:
    fields = {"initial_stock": FOUR_PER_STORE, "rider_speed_kmh": WALKING_PACE_KMH} | supply
    scenario = Scenario(days=days, supply=Supply(**fields))  # type: ignore[arg-type]
    clock = Clock(sim_minutes=scenario.sim_minutes, tick_minutes=TICK)
    world = build_world(line_city(), scenario, params or Params(), clock, np.random.default_rng(0))
    return new_sim(world), EventQueue()


def run_tick(
    sim: Sim,
    queue: EventQueue,
    start: int,
    attempts: list[tuple[int, int]] | None = None,
    draws: list[float] | None = None,
    pick_minutes: list[int] | None = None,
    travel_noise: list[float] | None = None,
) -> list[int]:
    """One tick's supply work for (resident, minute) attempts, as the run loop drives it.

    Unless given, picking takes a minute and travel has no noise.
    """
    attempts = attempts or []
    events = queue.pop_until(start + TICK - 1)
    who = np.array([resident for resident, _ in attempts], dtype=np.int64)
    minutes = np.array([minute for _, minute in attempts], dtype=np.int64)
    attempt_draws = AttemptDraws(
        hit=np.array(draws or [0.0] * len(attempts), dtype=np.float64),
        pick_minutes=np.array(pick_minutes or [1] * len(attempts), dtype=np.int64),
        travel_noise=np.array(travel_noise or [1.0] * len(attempts), dtype=np.float64),
    )
    served = serve_attempts(sim, queue, start, start + TICK, events, who, minutes, attempt_draws)
    return served.tolist()


Row = tuple[int, int, int, int, int, int]


def rows(sim: Sim) -> list[Row]:
    """(store, resident, ordered_at, picked_at, delivered_at, outcome) per logged delivery."""
    state = sim.supply
    assert state is not None
    log = state.deliveries.section(int(state.tally[DELIVERY_ROWS]))
    columns = (log.store, log.resident, log.ordered_at, log.picked_at, log.delivered_at)
    return list(zip(*(c.tolist() for c in (*columns, log.outcome)), strict=True))


def test_an_order_is_picked_ridden_and_logged_when_it_is_placed() -> None:
    sim, queue = bench()
    state = sim.supply
    assert state is not None
    assert run_tick(sim, queue, 0, [(1, 3)], pick_minutes=[2]) == [SERVED_FIRST]
    assert rows(sim) == [(0, 1, 3, 5, 7, 0)]
    assert state.picker_free.tolist() == [5, 0]
    assert state.rider_free[0].tolist() == [9, 0, 0, 0, 0, 0]
    assert len(queue) == 0, "the delivery fell inside the tick and was handled there"


def test_one_picker_per_store_serves_in_order() -> None:
    sim, queue = bench()
    state = sim.supply
    assert state is not None
    run_tick(sim, queue, 0, [(0, 3), (1, 4), (2, 4)], pick_minutes=[4, 1, 1])
    assert rows(sim) == [(0, 0, 3, 7, 8, 0), (0, 1, 4, 8, 10, 0), (1, 2, 4, 5, 6, 0)]
    assert state.picker_free.tolist() == [8, 5]
    assert state.rider_free[0, :3].tolist() == [9, 12, 0]


def test_a_picked_order_waits_for_the_only_rider_to_come_back() -> None:
    sim, queue = bench(riders_per_store=1)
    state = sim.supply
    assert state is not None
    run_tick(sim, queue, 0, [(1, 0), (0, 1)])
    assert rows(sim) == [(0, 1, 0, 1, 3, 0), (0, 0, 1, 2, 6, 0)]
    assert state.rider_free.tolist() == [[7], [0]]


def test_the_rider_free_earliest_takes_the_order_and_the_lowest_index_wins_a_tie() -> None:
    picker_free = np.zeros(1, dtype=np.int64)
    rider_free = np.array([[30, 12, 12, 40]], dtype=np.int64)
    assert pick_and_ride(0, 5, 2, 4, picker_free, rider_free) == (7, 16)
    assert rider_free.tolist() == [[30, 20, 12, 40]]
    assert pick_and_ride(0, 6, 1, 3, picker_free, rider_free) == (8, 15)
    assert rider_free.tolist() == [[30, 20, 18, 40]]
    assert picker_free.tolist() == [8]


def test_travel_is_the_noisy_straight_line_rounded_up_and_never_under_a_minute() -> None:
    sim, queue = bench(rider_speed_kmh=20.0)
    run_tick(sim, queue, 0, [(0, 0), (3, 0)], travel_noise=[1.0, 1.0])
    run_tick(sim, queue, 15, [(3, 20)], travel_noise=[2.01])
    travel = [delivered - picked for _, _, _, picked, delivered, _ in rows(sim)]
    assert travel == [1, 2, 4], "100 m is under a minute; 500 m is 1.5 and, with noise, 3.015"


def test_a_spilled_order_rides_from_the_spill_store_and_is_marked() -> None:
    sim, queue = bench()
    state = sim.supply
    assert state is not None
    state.stock[:] = [0, 4]
    run_tick(sim, queue, 0, [(0, 2)], draws=[0.1])
    assert rows(sim) == [(1, 0, 2, 3, 12, DELIVERY_SPILLED)]
    assert sim.residents.outcome_flags[0] == FLAG_STOCKOUT | FLAG_SPILLED


def test_a_late_delivery_costs_satisfaction_at_the_delivery_minute() -> None:
    sim, queue = bench(promise_minutes=5)
    state = sim.supply
    assert state is not None
    run_tick(sim, queue, 0, [(2, 10), (3, 10)])
    assert rows(sim) == [(1, 2, 10, 11, 12, 0), (1, 3, 10, 12, 17, DELIVERY_LATE)]
    assert (state.satisfaction.tolist(), state.tally[LATE]) == ([0.0] * 4, 0)
    assert queue.peek_minute() == 17

    run_tick(sim, queue, 15)
    assert state.satisfaction.tolist() == [0.0, 0.0, 0.0, -0.8]
    assert (state.tally[LATE], len(queue)) == (1, 0)


def test_a_delivery_that_takes_exactly_the_promise_is_on_time() -> None:
    sim, queue = bench(promise_minutes=6)
    run_tick(sim, queue, 0, [(3, 0)])
    run_tick(sim, queue, 15, [(3, 20)], pick_minutes=[2])
    assert [row[3:] for row in rows(sim)] == [(1, 6, 0), (22, 27, DELIVERY_LATE)]


def test_a_delivery_late_inside_the_tick_it_was_ordered_in_counts_in_that_tick() -> None:
    sim, queue = bench(promise_minutes=1)
    state = sim.supply
    assert state is not None
    run_tick(sim, queue, 0, [(1, 2)])
    assert rows(sim) == [(0, 1, 2, 3, 5, DELIVERY_LATE)]
    assert (state.satisfaction[1], state.tally[LATE], len(queue)) == (-0.8, 1, 0)


def test_what_falls_after_the_end_of_the_run_is_never_and_never_late() -> None:
    sim, queue = bench(days=1, promise_minutes=1)
    state = sim.supply
    assert state is not None
    run_tick(sim, queue, 1425, [(3, 1433), (3, 1438), (2, 1439)])
    assert rows(sim) == [
        (1, 3, 1433, 1434, 1439, DELIVERY_LATE),
        (1, 3, 1438, 1439, NEVER, 0),
        (1, 2, 1439, NEVER, NEVER, 0),
    ]
    assert (state.tally[LATE], len(queue)) == (1, 0)
    assert state.satisfaction.tolist() == [0.0, 0.0, 0.0, -0.8]


def test_the_quote_adds_the_picker_backlog_mean_picking_and_the_wait_for_a_rider() -> None:
    sim, _ = bench(picks_per_hour=30.0, riders_per_store=2)
    state, supply = sim.supply, sim.world.scenario.supply
    assert state is not None and supply is not None
    assert store_quotes(state, supply, 60).tolist() == [2.0, 2.0]
    state.picker_free[:] = [63, 40]
    state.rider_free[:] = [[70, 75], [61, 90]]
    assert store_quotes(state, supply, 60).tolist() == [10.0, 2.0]


def test_a_quote_over_the_promise_takes_delta_off_like_an_empty_store() -> None:
    params = Params(occasions_per_day=1e6, affinity_sd=0.0, rho_sd=0.0, alpha_0=-50.0, delta=100.0)
    sim, _ = bench(params, promise_minutes=10, riders_per_store=1)
    plan, state = sim.world.supply, sim.supply
    assert plan is not None and state is not None
    sim.residents.aware_at[:] = 0

    def deciding_to_order() -> list[int]:
        rng = np.random.default_rng(1)
        who, _ = attempts_from_occasions(sim, plan, state, rng, 60, 75, NO_EVENTS)
        return who.tolist()

    assert deciding_to_order() == [0, 1, 2, 3]
    state.rider_free[0, 0] = 69
    assert deciding_to_order() == [0, 2, 3], "9 minutes for a rider and 2 of travel is over 10"
    state.rider_free[0, 0] = 68
    assert deciding_to_order() == [0, 1, 2, 3]


def test_draws_per_attempt_come_from_the_supply_and_rider_streams() -> None:
    supply = Supply(picks_per_hour=6.0, travel_noise_sigma=0.15)
    draws = draw_for_attempts(supply, np.random.default_rng(1), np.random.default_rng(2), 500)
    again = draw_for_attempts(supply, np.random.default_rng(1), np.random.default_rng(2), 500)
    other_riders = draw_for_attempts(
        supply, np.random.default_rng(1), np.random.default_rng(3), 500
    )
    assert draws.hit.min() >= 0.0 and draws.hit.max() < 1.0
    assert draws.pick_minutes.min() >= 1 and 8.0 < draws.pick_minutes.mean() < 13.0
    assert 0.13 < np.log(draws.travel_noise).std() < 0.17
    assert np.array_equal(draws.pick_minutes, again.pick_minutes)
    assert np.array_equal(draws.pick_minutes, other_riders.pick_minutes)
    assert not np.array_equal(draws.travel_noise, other_riders.travel_noise)
    none = draw_for_attempts(supply, np.random.default_rng(1), np.random.default_rng(2), 0)
    assert none.hit.size == none.pick_minutes.size == none.travel_noise.size == 0


def test_the_delivery_log_grows_and_keeps_its_rows() -> None:
    log = new_delivery_log(capacity=2)
    log.ordered_at[:] = [7, 9]
    log.reserve(2)
    assert log.ordered_at.shape == (2,)
    log.reserve(5)
    assert log.ordered_at.shape == log.outcome.shape == (8,)
    assert log.ordered_at[:3].tolist() == [7, 9, 0]
    assert log.section(2).ordered_at.tolist() == [7, 9]


def town_run(
    supply: Supply, params: Params | None = None, seed: int = 3, days: int = 6
) -> RunInputs:
    launch = Campaign("launch", 0, 2880, budget_per_day=300.0, cpm=100.0, channel="social")
    scenario = Scenario(days=days, campaigns=(launch,), supply=supply)
    params = params or Params(alpha_0=-1.0, alpha_rep=-0.5, q=0.05)
    return RunInputs(town(), bytes(32), scenario, params, seed)


def check_supply_invariants(inputs: RunInputs, tape: Tape, stats: SupplyStats) -> None:
    supply = inputs.scenario.supply
    assert supply is not None
    plan: SupplyPlan = build_supply_plan(inputs.city, supply)
    header, residents, d, stores = tape.header, tape.residents, tape.deliveries, tape.stores
    n, k, ticks = header.residents, header.stores, header.ticks
    ordered = d.ordered_at.astype(np.int64)
    picked, delivered = d.picked_at.astype(np.int64), d.delivered_at.astype(np.int64)

    assert stats.attempts == stats.served + stats.deferred + stats.lost
    assert header.delivery_rows == ordered.shape[0] == stats.served
    assert stats.served == int(tape.aggregates.orders.sum()) == int(residents.order_count.sum())
    stock_out = stats.initial_stock + stats.restocked - stats.stock_end
    assert stock_out.tolist() == stats.served_by_store.tolist()
    assert np.bincount(d.store, minlength=k).tolist() == stats.served_by_store.tolist()
    assert np.bincount(d.resident, minlength=n).tolist() == residents.order_count.tolist()
    stock = stores.stock.reshape(k, ticks)
    assert stock.min() >= 0 and (stock == np.rint(stock)).all()
    assert tape.aggregates.stores_out.tolist() == (stock < 1).sum(axis=0).tolist()
    assert np.array_equal(
        np.bincount(ordered // header.tick_minutes, minlength=ticks), tape.aggregates.orders
    )

    assert (np.lexsort((d.resident, ordered)) == np.arange(ordered.shape[0])).all()
    assert (ordered < header.sim_minutes).all()
    done = d.delivered_at != NEVER
    assert (picked >= ordered).all() and (delivered[done] > picked[done]).all()
    assert not ((d.picked_at == NEVER) & done).any()
    assert (delivered[done] < header.sim_minutes).all()
    late = (d.outcome & DELIVERY_LATE) != 0
    assert np.array_equal(late, done & (delivered - ordered > supply.promise_minutes))
    assert stats.late == int(late.sum())
    spilled = (d.outcome & DELIVERY_SPILLED) != 0
    assert stats.spilled == int(spilled.sum())
    assert np.array_equal(spilled, d.store != plan.own_store[d.resident])
    assert (d.store[spilled] == plan.spill_store[d.resident[spilled]]).all()

    first = np.full(n, NEVER, dtype=np.int64)
    np.minimum.at(first, d.resident, ordered)
    assert np.array_equal(first, residents.first_order_at)
    flags = residents.outcome_flags
    hit = (flags & FLAG_STOCKOUT) != 0
    assert not (flags[~hit] != 0).any()
    assert set(d.resident[spilled].tolist()) == set(np.flatnonzero(flags & FLAG_SPILLED).tolist())
    assert np.count_nonzero(flags & FLAG_GAVE_UP) <= stats.lost
    assert (stats.lost == 0) == (np.count_nonzero(flags & FLAG_GAVE_UP) == 0)
    assert (residents.order_count[plan.own_store == NO_STORE] == 0).all()
    assert stats.unserviceable_residents == np.count_nonzero(plan.own_store == NO_STORE)
    assert stats.catchment.tolist() == plan.catchment.tolist()

    offsets = stores.stockout_offsets
    assert header.stockout_pairs == offsets[-1]
    last_minute = (np.arange(ticks) + 1) * header.tick_minutes - 1
    for store in range(k):
        pairs = stores.stockout_intervals[2 * offsets[store] : 2 * offsets[store + 1]]
        begin, end = pairs[0::2].astype(np.int64), pairs[1::2].astype(np.int64)
        assert (begin < end).all() and (end <= header.sim_minutes).all()
        assert (begin[1:] > end[:-1]).all()
        covered = (begin[:, None] <= last_minute[None, :]) & (last_minute[None, :] < end[:, None])
        assert np.array_equal(covered.any(axis=0), stock[store] < 1)


CONGESTED = Supply(
    initial_stock=0.5,
    reorder_point=0.2,
    order_up_to=0.5,
    lead_minutes=240,
    picks_per_hour=6.0,
    riders_per_store=1,
    rider_speed_kmh=3.0,
)


def test_a_congested_run_delivers_late_and_keeps_every_invariant() -> None:
    inputs = town_run(CONGESTED)
    tape, stats = run_seed_with_stats(inputs)
    check_supply_invariants(inputs, tape, stats)
    assert stats.served > INITIAL_DELIVERY_ROWS, "the delivery log had to grow"
    assert 0 < stats.late < stats.served
    assert stats.spilled > 0 and stats.lost > 0 and tape.header.stockout_pairs > 0
    assert run_seed_bytes(inputs) == run_seed_bytes(inputs)


def test_lateness_lowers_repeat_orders_through_satisfaction() -> None:
    late_matters = town_run(CONGESTED, Params(alpha_0=-1.0, alpha_rep=1.0, q=0.05))
    forgiving = dataclasses.replace(
        late_matters, params=dataclasses.replace(late_matters.params, satisfaction_drop=0.0)
    )
    orders = [
        int(read_tape(run_seed_bytes(inputs)).aggregates.orders.sum())
        for inputs in (late_matters, forgiving)
    ]
    assert orders[0] < 0.9 * orders[1]


supplies = st.builds(
    Supply,
    initial_stock=st.sampled_from([0.0, 0.02, 0.2, 5.0]),
    reorder_point=st.sampled_from([0.0, 0.01, 0.1]),
    order_up_to=st.sampled_from([0.0, 0.02, 0.2]),
    review_minutes=st.sampled_from([1, 7, 60, 600]),
    lead_minutes=st.sampled_from([0, 5, 90, 1440]),
    picks_per_hour=st.sampled_from([2.0, 120.0]),
    riders_per_store=st.sampled_from([1, 6]),
    rider_speed_kmh=st.sampled_from([1.0, 20.0]),
    travel_noise_sigma=st.sampled_from([0.0, 0.15, 1.0]),
    promise_minutes=st.sampled_from([0, 5, 20]),
    spill_radius_factor=st.sampled_from([0.0, 1.0, 1.67, 4.0]),
    wait_minutes=st.sampled_from([0, 45, 1440]),
)
SPILL_AND_WAIT_SHARES = [(0.42, 0.38), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.5), (0.6, 0.4)]
supply_params = st.sampled_from(SPILL_AND_WAIT_SHARES).flatmap(
    lambda shares: st.builds(
        Params,
        alpha_0=st.sampled_from([-3.0, -1.0, 1.0]),
        alpha_rep=st.sampled_from([-2.3, 0.5]),
        q=st.sampled_from([0.0, 0.05]),
        p_spill=st.just(shares[0]),
        p_wait=st.just(shares[1]),
        satisfaction_drop=st.sampled_from([0.0, 0.8]),
        occasions_per_day=st.sampled_from([0.0, 3.0, 12.0]),
        order_on_awareness=st.booleans(),
    )
)


@settings(derandomize=True, max_examples=40, deadline=None)
@given(supply=supplies, params=supply_params, seed=st.integers(0, 2**32))
def test_every_supply_run_conserves_orders_stock_and_time(
    supply: Supply, params: Params, seed: int
) -> None:
    inputs = town_run(supply, params, seed, days=3)
    tape, stats = run_seed_with_stats(inputs)
    check_supply_invariants(inputs, tape, stats)
