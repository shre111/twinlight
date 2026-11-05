import dataclasses
from functools import lru_cache

import numpy as np

from twl_engine.clock import Clock
from twl_engine.events import NO_EVENTS, RESTOCK_ARRIVAL, EventQueue
from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.fixtures.types import Residents, Stores, Ties
from twl_engine.run import RunInputs, run_seed_bytes, run_seed_with_stats
from twl_engine.scenario import Campaign, Params, Scenario, Supply
from twl_engine.state import (
    ATTEMPTS,
    DEFERRED,
    LOST,
    NOT_OUT,
    NOT_WAITING,
    SERVED,
    SPILLED,
    STOCKOUT_ROWS,
    Sim,
    new_sim,
)
from twl_engine.supply import (
    NOT_SERVED,
    SERVED_FIRST,
    SERVED_REPEAT,
    AttemptDraws,
    attempts_from_occasions,
    close_stockout,
    serve_attempts,
    stores_section,
)
from twl_engine.world import NO_STORE, build_world
from twl_tape import FLAG_GAVE_UP, FLAG_SPILLED, FLAG_STOCKOUT, NEVER, read_tape

TICK = 15
TWO_PER_STORE = 1.0
RADIUS = 600.0


TOWN_RESIDENTS = 1200
TOWN_STORES = 14


@lru_cache(maxsize=1)
def town() -> ToyCity:
    return build_toy_city(seed=49, residents=TOWN_RESIDENTS, stores=TOWN_STORES)


def line_city() -> ToyCity:
    """Stores at x = 0 and 1000 with a 600 m radius; residents 0 and 1 belong to store 0 and
    residents 2 and 3 to store 1. Resident 3 at x = 1500 is beyond store 0's spill reach."""
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
        radius_m=np.full(2, RADIUS, dtype=np.float32),
        district=np.zeros(2, dtype=np.uint16),
    )
    ties = Ties(
        offsets=np.zeros(n + 1, dtype=np.uint32),
        targets=np.zeros(0, dtype=np.uint32),
        kind=np.zeros(0, dtype=np.uint8),
        strength=np.zeros(0, dtype=np.float32),
    )
    return dataclasses.replace(town(), residents=residents, stores=stores, ties=ties)


def bench(supply: Supply, params: Params | None = None, days: int = 2) -> tuple[Sim, EventQueue]:
    scenario = Scenario(days=days, supply=supply)
    clock = Clock(sim_minutes=scenario.sim_minutes, tick_minutes=TICK)
    rng = np.random.default_rng(0)
    world = build_world(line_city(), scenario, params or Params(), clock, rng)
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


def run_ticks(sim: Sim, queue: EventQueue, first_start: int, last_start: int) -> None:
    for start in range(first_start, last_start + 1, TICK):
        run_tick(sim, queue, start)


def send_restock(sim: Sim, queue: EventQueue, minute: int, store: int, units: int) -> None:
    state = sim.supply
    assert state is not None
    state.in_transit[store] += units
    queue.push(minute, RESTOCK_ARRIVAL, store, units)


def stockouts(sim: Sim) -> list[tuple[int, int, int]]:
    state = sim.supply
    assert state is not None
    rows = int(state.tally[STOCKOUT_ROWS])
    return list(
        zip(
            state.stockout_store[:rows].tolist(),
            state.stockout_start[:rows].tolist(),
            state.stockout_end[:rows].tolist(),
            strict=True,
        )
    )


def test_line_city_has_the_catchments_the_tests_rely_on() -> None:
    sim, _ = bench(Supply(initial_stock=TWO_PER_STORE))
    plan, state = sim.world.supply, sim.supply
    assert plan is not None and state is not None
    assert plan.own_store.tolist() == [0, 0, 1, 1]
    assert plan.spill_store.tolist() == [1, 1, 0, NO_STORE]
    assert state.stock.tolist() == [2, 2]


def test_an_order_takes_stock_when_placed_and_the_last_unit_starts_a_stockout() -> None:
    supply = Supply(
        initial_stock=TWO_PER_STORE, reorder_point=0.0, order_up_to=TWO_PER_STORE, lead_minutes=120
    )
    sim, queue = bench(supply)
    state, residents, ticks = sim.supply, sim.residents, sim.world.clock.ticks
    assert state is not None

    assert run_tick(sim, queue, 0, [(0, 3), (1, 7)]) == [SERVED_FIRST, SERVED_FIRST]
    assert state.stock.tolist() == [0, 2]
    assert state.out_since.tolist() == [7, NOT_OUT]
    assert residents.first_order_at[:2].tolist() == [3, 7]
    assert residents.last_order_at[:2].tolist() == [3, 7]
    assert residents.order_count.tolist() == [1, 1, 0, 0]
    assert (state.orders_series[0], state.orders_series[ticks]) == (2, 0)
    assert (state.stock_series[0], state.stock_series[ticks]) == (0.0, 2.0)
    assert (sim.aggregates.orders[0], sim.aggregates.stores_out[0]) == (2, 1)
    assert len(queue) == 0

    run_ticks(sim, queue, 15, 165)
    assert (queue.peek_minute(), state.in_transit.tolist()) == (180, [2, 0])
    assert sim.aggregates.stores_out[1:12].tolist() == [1] * 11
    run_tick(sim, queue, 180)
    assert (state.stock.tolist(), state.in_transit.tolist()) == ([2, 2], [0, 0])
    assert state.restocked.tolist() == [2, 0]
    assert stockouts(sim) == [(0, 7, 180)]
    assert sim.aggregates.stores_out[12] == 0


def test_a_repeat_order_keeps_the_first_order_minute() -> None:
    sim, queue = bench(Supply(initial_stock=TWO_PER_STORE))
    assert run_tick(sim, queue, 0, [(0, 3)]) == [SERVED_FIRST]
    assert run_tick(sim, queue, 15, [(0, 20)]) == [SERVED_REPEAT]
    assert sim.residents.first_order_at[0] == 3
    assert sim.residents.last_order_at[0] == 20
    assert sim.residents.order_count[0] == 2


def test_a_restock_arriving_in_the_minute_of_an_attempt_comes_first() -> None:
    sim, queue = bench(Supply(initial_stock=0.0, reorder_point=0.0, order_up_to=0.0))
    state = sim.supply
    assert state is not None
    send_restock(sim, queue, 185, 0, 1)
    outcome = run_tick(sim, queue, 180, [(0, 184), (1, 185)], draws=[0.99, 0.99])
    assert outcome == [NOT_SERVED, SERVED_FIRST]
    assert stockouts(sim) == [(0, 0, 185)]
    assert state.out_since.tolist() == [185, 0]


def test_a_review_sees_its_minute_after_arrivals_and_before_orders() -> None:
    supply = Supply(
        initial_stock=TWO_PER_STORE, reorder_point=0.5, order_up_to=TWO_PER_STORE, lead_minutes=45
    )
    sim, queue = bench(supply)
    state = sim.supply
    assert state is not None
    state.stock[:] = [2, 1]
    send_restock(sim, queue, 60, 1, 1)

    run_tick(sim, queue, 60, [(0, 60)])
    assert state.stock.tolist() == [1, 2]
    assert len(queue) == 0, "store 0 was above s at the review; store 1 was restocked before it"

    run_ticks(sim, queue, 75, 120)
    assert (queue.peek_minute(), state.in_transit.tolist()) == (165, [1, 0])


def test_a_restock_with_no_lead_time_serves_an_order_in_the_review_minute() -> None:
    supply = Supply(initial_stock=0.0, reorder_point=0.0, order_up_to=TWO_PER_STORE, lead_minutes=0)
    sim, queue = bench(supply)
    state = sim.supply
    assert state is not None
    assert state.out_since.tolist() == [0, 0]
    assert run_tick(sim, queue, 0, [(0, 0)]) == [SERVED_FIRST]
    assert state.stock.tolist() == [1, 2]
    assert state.restocked.tolist() == [2, 2]
    assert stockouts(sim) == [] and state.out_since.tolist() == [NOT_OUT, NOT_OUT]


def test_a_restock_due_inside_the_tick_that_ordered_it_arrives_in_that_tick() -> None:
    supply = Supply(initial_stock=0.0, reorder_point=0.0, order_up_to=TWO_PER_STORE, lead_minutes=5)
    sim, queue = bench(supply)
    outcome = run_tick(sim, queue, 0, [(0, 4), (1, 5)], draws=[0.99, 0.99])
    assert outcome == [NOT_SERVED, SERVED_FIRST]
    assert stockouts(sim) == [(0, 0, 5), (1, 0, 5)]
    assert len(queue) == 0


def test_out_of_stock_spills_then_waits_then_loses_the_sale() -> None:
    params = Params(p_spill=0.42, p_wait=0.38, satisfaction_drop=0.8)
    sim, queue = bench(Supply(initial_stock=0.0, order_up_to=0.0, wait_minutes=1440), params)
    state, residents = sim.supply, sim.residents
    assert state is not None
    state.stock[:] = [0, 1]

    outcome = run_tick(sim, queue, 0, [(0, 1), (3, 2), (1, 3), (2, 4)], draws=[0.1, 0.1, 0.1, 0.9])

    assert outcome == [SERVED_FIRST, NOT_SERVED, NOT_SERVED, NOT_SERVED]
    assert state.stock.tolist() == [0, 0], "resident 0 was served by its spill store"
    tally = state.tally
    assert (tally[ATTEMPTS], tally[SERVED], tally[SPILLED]) == (4, 1, 1)
    assert (tally[DEFERRED], tally[LOST]) == (2, 1)
    assert residents.outcome_flags.tolist() == [
        FLAG_STOCKOUT | FLAG_SPILLED,
        FLAG_STOCKOUT,
        FLAG_STOCKOUT | FLAG_GAVE_UP,
        FLAG_STOCKOUT,
    ]
    assert state.satisfaction.tolist() == [-0.8, -0.8, -0.8, -0.8]
    assert state.waiting_until.tolist() == [NOT_WAITING, 3 + 1440, NOT_WAITING, 2 + 1440]
    assert residents.order_count.tolist() == [1, 0, 0, 0]
    assert (sim.aggregates.orders[0], state.orders_series[sim.world.clock.ticks]) == (1, 1)


def test_a_draw_between_the_spill_and_wait_shares_waits_even_with_a_stocked_spill_store() -> None:
    sim, queue = bench(Supply(initial_stock=0.0, order_up_to=0.0, wait_minutes=30))
    state = sim.supply
    assert state is not None
    state.stock[:] = [0, 1]
    assert run_tick(sim, queue, 0, [(0, 9)], draws=[0.5]) == [NOT_SERVED]
    assert state.waiting_until[0] == 39 and state.stock.tolist() == [0, 1]
    assert run_tick(sim, queue, 45, [(0, 50)], draws=[0.5]) == [NOT_SERVED]
    assert state.waiting_until[0] == 80 and state.satisfaction[0] == -1.6
    assert (state.tally[ATTEMPTS], state.tally[DEFERRED]) == (2, 2)


def test_stockouts_that_touch_are_one_interval_and_empty_ones_are_dropped() -> None:
    out_since = np.array([5, NOT_OUT], dtype=np.int64)
    last = np.full(2, -1, dtype=np.int64)
    store, start, end = (np.zeros(4, dtype=np.int64) for _ in range(3))
    tally = np.zeros(7, dtype=np.int64)

    def close(minute: int) -> None:
        close_stockout(0, minute, out_since, last, store, start, end, tally)

    close(20)
    out_since[0] = 20
    close(30)
    out_since[0] = 31
    close(31)
    out_since[0] = 31
    close(40)
    rows = int(tally[STOCKOUT_ROWS])
    assert list(zip(start[:rows].tolist(), end[:rows].tolist(), strict=True)) == [(5, 30), (31, 40)]
    assert out_since[0] == NOT_OUT


def test_a_unit_that_arrives_and_sells_in_one_minute_does_not_split_the_stockout() -> None:
    sim, queue = bench(Supply(initial_stock=0.0, order_up_to=0.0), days=1)
    state = sim.supply
    assert state is not None
    send_restock(sim, queue, 20, 0, 1)
    assert run_tick(sim, queue, 15, [(0, 20)]) == [SERVED_FIRST]
    section = stores_section(state, sim.world.clock.sim_minutes)
    assert section.stockout_offsets.tolist() == [0, 1, 2]
    assert section.stockout_intervals.tolist() == [0, 1440, 0, 1440]


def everyone_decides(**overrides: float) -> Params:
    """An occasion for every aware resident every tick; `alpha_0` and `alpha_rep` set the choice."""
    return Params(occasions_per_day=1e6, affinity_sd=0.0, rho_sd=0.0, **overrides)  # type: ignore[arg-type]


def attempts(sim: Sim, start: int, seed: int = 1) -> dict[int, int]:
    plan, state = sim.world.supply, sim.supply
    assert plan is not None and state is not None
    rng = np.random.default_rng(seed)
    who, minutes = attempts_from_occasions(sim, plan, state, rng, start, start + TICK, NO_EVENTS)
    return dict(zip(who.tolist(), minutes.tolist(), strict=True))


def test_a_waiting_resident_sits_out_occasions_then_attempts_without_deciding() -> None:
    sim, _ = bench(Supply(initial_stock=TWO_PER_STORE), everyone_decides(alpha_0=-60.0))
    state = sim.supply
    assert state is not None
    sim.residents.aware_at[:] = 0
    assert attempts(sim, 75) == {}, "nobody wants the product at this alpha"

    state.waiting_until[1] = 100
    assert attempts(sim, 75) == {}
    retry = attempts(sim, 90)
    assert list(retry) == [1] and 100 <= retry[1] < 105
    assert list(attempts(sim, 105)) == [1], "still due until an attempt clears the wait"


def test_a_waiting_resident_who_would_order_still_waits() -> None:
    sim, _ = bench(Supply(initial_stock=TWO_PER_STORE), everyone_decides(alpha_0=60.0))
    state = sim.supply
    assert state is not None
    sim.residents.aware_at[:] = 0
    state.waiting_until[1] = 100
    assert sorted(attempts(sim, 75)) == [0, 2, 3]


def test_unaware_residents_and_those_inside_no_radius_get_no_occasion() -> None:
    sim, _ = bench(Supply(initial_stock=TWO_PER_STORE), everyone_decides(alpha_0=60.0))
    plan = sim.world.supply
    assert plan is not None
    sim.residents.aware_at[[0, 2, 3]] = 0
    plan.serviceable[3] = False
    assert sorted(attempts(sim, 30)) == [0, 2]


def test_an_empty_own_store_takes_delta_off_the_first_order_utility() -> None:
    sim, queue = bench(
        Supply(initial_stock=TWO_PER_STORE), everyone_decides(alpha_0=-50.0, delta=100.0)
    )
    plan, state = sim.world.supply, sim.supply
    assert plan is not None and state is not None
    sim.residents.aware_at[:] = 0
    state.stock[:] = [0, 2]
    assert sorted(attempts(sim, 30)) == [2, 3]

    send_restock(sim, queue, 45, 0, 1)
    arriving = queue.pop_until(59)
    rng = np.random.default_rng(1)
    who, _ = attempts_from_occasions(sim, plan, state, rng, 45, 60, arriving)
    assert who.tolist() == [0, 1, 2, 3], "a restock landing on the tick's first minute counts"


def test_satisfaction_enters_the_repeat_utility() -> None:
    sim, _ = bench(Supply(initial_stock=TWO_PER_STORE), everyone_decides(alpha_rep=20.0))
    state = sim.supply
    assert state is not None
    sim.residents.aware_at[:] = 0
    sim.residents.first_order_at[:] = 5
    state.satisfaction[[1, 3]] = -100.0
    assert sorted(attempts(sim, 30)) == [0, 2]


def stocked_run(supply: Supply, seed: int = 3, city: ToyCity | None = None) -> RunInputs:
    launch = Campaign("launch", 0, 2880, budget_per_day=300.0, cpm=100.0)
    scenario = Scenario(days=6, campaigns=(launch,), supply=supply)
    params = Params(alpha_0=-1.0, alpha_rep=-0.5, q=0.05)
    return RunInputs(city or town(), bytes(32), scenario, params, seed)


LOW_STOCK = Supply(initial_stock=0.05, reorder_point=0.0125, order_up_to=0.05, lead_minutes=720)


def test_a_low_stock_run_conserves_orders_and_stock() -> None:
    tape, stats = run_seed_with_stats(stocked_run(LOW_STOCK))
    residents, stores, aggregates = tape.residents, tape.stores, tape.aggregates
    assert stats.lost > 0 and stats.deferred > 0 and stats.spilled > 0
    assert stats.attempts == stats.served + stats.deferred + stats.lost
    assert stats.served == int(aggregates.orders.sum()) == int(residents.order_count.sum())
    assert stats.served == int(stats.served_by_store.sum())
    stock_out = stats.initial_stock + stats.restocked - stats.stock_end
    assert stock_out.tolist() == stats.served_by_store.tolist()
    assert stats.catchment.sum() + stats.unserviceable_residents == TOWN_RESIDENTS
    ticks = tape.header.ticks
    stock = stores.stock.reshape(TOWN_STORES, ticks)
    assert stock.min() >= 0 and (stock == np.rint(stock)).all()
    assert stock[:, -1].tolist() == stats.stock_end.tolist()
    assert (
        stores.orders.reshape(TOWN_STORES, ticks).sum(axis=1).tolist()
        == stats.served_by_store.tolist()
    )
    assert (
        stores.orders.reshape(TOWN_STORES, ticks).sum(axis=0).tolist() == aggregates.orders.tolist()
    )
    assert np.array_equal(residents.first_order_at != NEVER, residents.order_count > 0)


def test_stockout_intervals_agree_with_the_stock_series() -> None:
    tape, _ = run_seed_with_stats(stocked_run(LOW_STOCK))
    header, stores = tape.header, tape.stores
    ticks, sim_minutes = header.ticks, header.sim_minutes
    stock = stores.stock.reshape(TOWN_STORES, ticks)
    last_minute = (np.arange(ticks) + 1) * header.tick_minutes - 1
    offsets = stores.stockout_offsets
    assert header.stockout_pairs == offsets[-1] > 0
    out_at_tick_end = np.zeros((TOWN_STORES, ticks), dtype=bool)
    for store in range(TOWN_STORES):
        pairs = stores.stockout_intervals[2 * offsets[store] : 2 * offsets[store + 1]]
        start, end = pairs[0::2].astype(np.int64), pairs[1::2].astype(np.int64)
        assert (start < end).all() and (end <= sim_minutes).all()
        assert (start[1:] > end[:-1]).all(), "sorted, disjoint and never adjacent"
        covered = (start[:, None] <= last_minute[None, :]) & (last_minute[None, :] < end[:, None])
        out_at_tick_end[store] = covered.any(axis=0)
    assert np.array_equal(out_at_tick_end, stock < 1)
    assert tape.aggregates.stores_out.tolist() == (stock < 1).sum(axis=0).tolist()


def test_flags_mark_who_met_a_stockout_and_nobody_outside_a_radius_orders() -> None:
    inputs = stocked_run(LOW_STOCK)
    tape, stats = run_seed_with_stats(inputs)
    flags = tape.residents.outcome_flags
    hit = (flags & FLAG_STOCKOUT) != 0
    assert hit.any() and not (flags[~hit] != 0).any()
    assert ((flags & FLAG_SPILLED) != 0).sum() <= stats.spilled
    assert 0 < ((flags & FLAG_GAVE_UP) != 0).sum() <= stats.lost

    scenario = inputs.scenario
    clock = Clock(scenario.sim_minutes, scenario.tick_minutes)
    world = build_world(inputs.city, scenario, inputs.params, clock, np.random.default_rng(0))
    assert world.supply is not None
    outside = ~world.supply.serviceable
    assert outside.sum() == stats.unserviceable_residents > 0
    assert (tape.residents.aware_at[outside] != NEVER).any()
    assert (tape.residents.order_count[outside] == 0).all()
    assert (flags[outside] == 0).all()


def test_a_supply_run_is_reproducible_and_differs_from_the_unlimited_run() -> None:
    inputs = stocked_run(LOW_STOCK)
    data = run_seed_bytes(inputs)
    assert run_seed_bytes(inputs) == data
    unlimited = dataclasses.replace(
        inputs, scenario=dataclasses.replace(inputs.scenario, supply=None)
    )
    limited_orders = int(read_tape(data).aggregates.orders.sum())
    assert limited_orders < int(read_tape(run_seed_bytes(unlimited)).aggregates.orders.sum())


def test_ample_supply_for_everyone_reproduces_the_unlimited_run() -> None:
    city = town()
    everywhere = dataclasses.replace(
        city.stores, radius_m=np.full(TOWN_STORES, 1e6, dtype=np.float32)
    )
    city = dataclasses.replace(city, stores=everywhere)
    ample = Supply(initial_stock=1000.0, order_up_to=1000.0, promise_minutes=10**6)
    limited, stats = run_seed_with_stats(stocked_run(ample, city=city))
    inputs = stocked_run(ample, city=city)
    unlimited, _ = run_seed_with_stats(
        dataclasses.replace(inputs, scenario=dataclasses.replace(inputs.scenario, supply=None))
    )
    assert stats.unserviceable_residents == 0 and stats.attempts == stats.served > 0
    for section in ("residents", "chatter"):
        ours, theirs = getattr(limited, section), getattr(unlimited, section)
        for field in dataclasses.fields(ours):
            assert np.array_equal(getattr(ours, field.name), getattr(theirs, field.name)), (
                f"{section}.{field.name}"
            )
    for name in ("aware", "adopted", "orders", "stores_out"):
        assert np.array_equal(
            getattr(limited.aggregates, name), getattr(unlimited.aggregates, name)
        ), name
    assert limited.header.stockout_pairs == 0
