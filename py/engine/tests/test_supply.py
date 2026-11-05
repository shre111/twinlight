import dataclasses
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from numpy.typing import NDArray

from twl_engine.goldens import (
    golden_inputs,
    low_stock_inputs,
    prototype_small_inputs,
    supply_small_inputs,
)
from twl_engine.run import RunInputs, SupplyStats, run_seed, run_seed_bytes, run_seed_with_stats
from twl_engine.scenario import Supply
from twl_tape import write_tape
from twl_tape.format import (
    DELIVERY_LATE,
    DELIVERY_SPILLED,
    FLAG_GAVE_UP,
    FLAG_SPILLED,
    FLAG_STOCKOUT,
    NEVER,
    Tape,
)

CONFIGS = {"supply_small": supply_small_inputs, "low_stock": low_stock_inputs}
SEEDS = (1, 2)
EPS = 0.01
PLENTIFUL = {"initial_stock": 5.0, "order_up_to": 5.0, "reorder_point": 2.5}


@dataclass(frozen=True)
class Result:
    inputs: RunInputs
    tape: Tape
    stats: SupplyStats

    @property
    def supply(self) -> Supply:
        assert self.inputs.scenario.supply is not None
        return self.inputs.scenario.supply

    @property
    def stock(self) -> NDArray[np.int64]:
        t = self.tape
        return t.stores.stock.reshape(t.header.stores, t.header.ticks).astype(np.int64)

    @property
    def store_orders(self) -> NDArray[np.int64]:
        t = self.tape
        return t.stores.orders.reshape(t.header.stores, t.header.ticks).astype(np.int64)

    @property
    def arrived(self) -> NDArray[np.int64]:
        stock, orders = self.stock, self.store_orders
        previous = np.concatenate([self.stats.initial_stock[:, None], stock[:, :-1]], axis=1)
        return stock - (previous - orders)

    def intervals(self, store: int) -> NDArray[np.int64]:
        offsets = self.tape.stores.stockout_offsets
        pairs = self.tape.stores.stockout_intervals.reshape(-1, 2).astype(np.int64)
        return pairs[int(offsets[store]) : int(offsets[store + 1])]


def run_of(inputs: RunInputs) -> Result:
    tape, stats = run_seed_with_stats(inputs)
    return Result(inputs, tape, stats)


@cache
def config_result(name: str, seed: int) -> Result:
    return run_of(CONFIGS[name](seed))


@cache
def variant(seed: int, **supply: float) -> Result:
    base = supply_small_inputs(seed)
    scenario = dataclasses.replace(base.scenario, supply=Supply(**supply))  # type: ignore[arg-type]
    return run_of(dataclasses.replace(base, scenario=scenario))


def every_run() -> list[tuple[str, int]]:
    return [(name, seed) for name in CONFIGS for seed in SEEDS]


def geometry(r: Result) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    city = r.inputs.city
    dx = city.residents.x.astype(np.float64)[:, None] - city.stores.x.astype(np.float64)[None, :]
    dy = city.residents.y.astype(np.float64)[:, None] - city.stores.y.astype(np.float64)[None, :]
    dist = np.hypot(dx, dy)
    radius = city.stores.radius_m.astype(np.float64)
    masked = np.where(dist <= radius[None, :], dist, np.inf)
    own = np.where(np.isfinite(masked.min(axis=1)), masked.argmin(axis=1), -1)
    return dist, radius, own


def delivered_rows(r: Result) -> NDArray[np.bool_]:
    return r.tape.deliveries.delivered_at != NEVER


def durations(r: Result) -> NDArray[np.int64]:
    d = r.tape.deliveries
    rows = delivered_rows(r)
    return d.delivered_at[rows].astype(np.int64) - d.ordered_at[rows].astype(np.int64)


def late_share(r: Result) -> float:
    return float(np.mean(durations(r) > r.supply.promise_minutes))


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_orders_are_conserved_from_attempt_to_delivery_row(name: str, seed: int) -> None:
    r = config_result(name, seed)
    s, t = r.stats, r.tape
    assert s.attempts == s.served + s.deferred + s.lost
    assert s.served > 0
    assert len(t.deliveries.store) == s.served == t.header.delivery_rows
    assert int(t.aggregates.orders.astype(np.int64).sum()) == s.served
    assert int(t.residents.order_count.astype(np.int64).sum()) == s.served
    assert s.catchment.sum() + s.unserviceable_residents == t.header.residents


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_each_store_balances_stock_against_orders_served(name: str, seed: int) -> None:
    r = config_result(name, seed)
    s = r.stats
    rows = np.bincount(r.tape.deliveries.store.astype(np.int64), minlength=len(s.catchment))
    assert np.array_equal(s.initial_stock + s.restocked - s.stock_end, s.served_by_store)
    assert np.array_equal(rows, s.served_by_store)
    assert np.array_equal(r.store_orders.sum(axis=1), s.served_by_store)
    assert int(s.served_by_store.sum()) == s.served


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_stock_series_is_whole_non_negative_and_ends_at_stock_end(name: str, seed: int) -> None:
    r = config_result(name, seed)
    raw = r.tape.stores.stock
    assert raw.min() >= 0
    assert np.array_equal(raw, np.round(raw))
    assert np.array_equal(r.stock[:, -1], r.stats.stock_end)


@pytest.mark.parametrize("seed", SEEDS)
def test_initial_stock_is_the_rate_times_the_catchment(seed: int) -> None:
    s = config_result("supply_small", seed).stats
    assert np.array_equal(s.initial_stock, np.round(0.036 * s.catchment).astype(np.int64))
    low = config_result("low_stock", seed).stats
    assert np.array_equal(low.initial_stock, np.round(0.004 * low.catchment).astype(np.int64))


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_stock_only_rises_at_restocks_and_otherwise_falls_by_the_ticks_orders(
    name: str, seed: int
) -> None:
    r = config_result(name, seed)
    arrived = r.arrived
    assert arrived.min() >= 0
    assert np.array_equal(arrived.sum(axis=1), r.stats.restocked)
    previous = np.concatenate([r.stats.initial_stock[:, None], r.stock[:, :-1]], axis=1)
    quiet = arrived == 0
    assert np.array_equal(r.stock[quiet], (previous - r.store_orders)[quiet])


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_spilled_and_late_counts_match_delivery_outcome_bits(name: str, seed: int) -> None:
    r = config_result(name, seed)
    outcome = r.tape.deliveries.outcome
    assert r.stats.spilled == int(np.count_nonzero(outcome & DELIVERY_SPILLED))
    late = (outcome & DELIVERY_LATE != 0) & delivered_rows(r)
    assert r.stats.late == int(np.count_nonzero(late))


def test_low_stock_run_has_stockout_intervals() -> None:
    r = config_result("low_stock", 1)
    counts = np.diff(r.tape.stores.stockout_offsets.astype(np.int64))
    assert counts.sum() == r.tape.header.stockout_pairs > 0
    assert r.tape.stores.stockout_intervals.shape[0] == 2 * r.tape.header.stockout_pairs


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_stockout_intervals_are_sorted_disjoint_end_exclusive_and_not_adjacent(
    name: str, seed: int
) -> None:
    r = config_result(name, seed)
    end_of_run = r.tape.header.sim_minutes
    for store in range(r.tape.header.stores):
        pairs = r.intervals(store)
        assert np.all(
            (pairs[:, 0] >= 0) & (pairs[:, 0] < pairs[:, 1]) & (pairs[:, 1] <= end_of_run)
        )
        assert np.all(pairs[1:, 0] > pairs[:-1, 1])


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_stockout_interval_begins_on_the_last_unit_and_ends_at_a_restock(
    name: str, seed: int
) -> None:
    r = config_result(name, seed)
    tick = r.tape.header.tick_minutes
    end_of_run = r.tape.header.sim_minutes
    arrived = r.arrived
    for store in range(r.tape.header.stores):
        pairs = r.intervals(store)
        for start, end in pairs:
            t = int(start // tick)
            assert r.stock[store, t] == 0 or end < (t + 1) * tick
            if end < end_of_run:
                assert arrived[store, int(end // tick)] > 0
            else:
                assert r.stock[store, -1] == 0
        below = r.stock[store] < 1
        assert bool(below[-1]) == bool(len(pairs) and pairs[-1, 1] == end_of_run)


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_store_serves_no_new_order_strictly_inside_its_stockout_interval(
    name: str, seed: int
) -> None:
    r = config_result(name, seed)
    d = r.tape.deliveries
    ordered = d.ordered_at.astype(np.int64)
    for store in range(r.tape.header.stores):
        mine = ordered[d.store == store]
        for start, end in r.intervals(store):
            assert not np.any((mine > start) & (mine < end))


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_stores_out_counts_stores_with_stock_below_one_each_tick(name: str, seed: int) -> None:
    r = config_result(name, seed)
    out = np.count_nonzero(r.stock < 1, axis=0)
    assert np.array_equal(r.tape.aggregates.stores_out.astype(np.int64), out)


@pytest.mark.parametrize("seed", SEEDS)
def test_stockout_outcomes_are_flagged_on_residents(seed: int) -> None:
    r = config_result("low_stock", seed)
    flags = r.tape.residents.outcome_flags
    d = r.tape.deliveries
    assert r.stats.lost > 0
    assert np.any(flags & FLAG_GAVE_UP)
    assert r.stats.spilled > 0
    assert np.any(flags & FLAG_SPILLED)
    spilled = d.resident[d.outcome & DELIVERY_SPILLED != 0]
    assert np.all(flags[spilled] & FLAG_SPILLED)
    assert np.all(flags[spilled] & FLAG_STOCKOUT)
    hurt = flags & (FLAG_SPILLED | FLAG_GAVE_UP) != 0
    assert np.all(flags[hurt] & FLAG_STOCKOUT)


def test_plentiful_stock_has_no_stockouts_losses_spills_or_flags() -> None:
    r = variant(1, **PLENTIFUL)
    s = r.stats
    assert s.served > 0
    assert r.tape.header.stockout_pairs == 0
    assert s.lost == s.deferred == s.spilled == 0
    assert not np.any(r.tape.residents.outcome_flags)
    assert not np.any(r.tape.aggregates.stores_out)


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_deliveries_come_from_stores_that_serve_the_resident(name: str, seed: int) -> None:
    r = config_result(name, seed)
    dist, radius, own = geometry(r)
    d = r.tape.deliveries
    store, resident = d.store.astype(np.int64), d.resident.astype(np.int64)
    distance = dist[resident, store]
    assert np.all(distance <= r.supply.spill_radius_factor * radius[store] + EPS)
    assert np.all(own[resident] >= 0)
    spilled = d.outcome & DELIVERY_SPILLED != 0
    assert np.all(store[spilled] != own[resident][spilled])
    plain = ~spilled
    assert np.all(distance[plain] <= radius[store[plain]] + EPS)
    assert np.all(store[plain] == own[resident][plain])


@pytest.mark.parametrize("seed", SEEDS)
def test_residents_inside_no_radius_never_order_and_catchments_match_geometry(seed: int) -> None:
    r = config_result("supply_small", seed)
    _, _, own = geometry(r)
    assert r.stats.unserviceable_residents == int(np.count_nonzero(own < 0))
    expected = np.bincount(own[own >= 0], minlength=r.tape.header.stores)
    assert np.array_equal(r.stats.catchment, expected)
    assert np.all(r.tape.residents.order_count[own < 0] == 0)
    assert np.all(r.tape.residents.first_order_at[own < 0] == NEVER)


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_delivery_rows_are_sorted_and_time_ordered(name: str, seed: int) -> None:
    r = config_result(name, seed)
    d = r.tape.deliveries
    end = r.tape.header.sim_minutes
    ordered = d.ordered_at.astype(np.int64)
    picked, delivered = d.picked_at.astype(np.int64), d.delivered_at.astype(np.int64)
    got_picked, got_delivered = d.picked_at != NEVER, d.delivered_at != NEVER
    assert np.all(np.diff(ordered) >= 0)
    assert np.all(ordered < end)
    assert np.all(picked[got_picked] >= ordered[got_picked])
    assert np.all(picked[got_picked] < end)
    assert np.all(delivered[got_delivered] < end)
    both = got_picked & got_delivered
    assert np.all(delivered[both] > picked[both])
    assert np.all(~got_delivered[~got_picked])


@pytest.mark.parametrize(("name", "seed"), every_run())
def test_late_rows_exceed_the_promise_and_on_time_rows_do_not(name: str, seed: int) -> None:
    r = config_result(name, seed)
    rows = delivered_rows(r)
    late = r.tape.deliveries.outcome[rows] & DELIVERY_LATE != 0
    assert np.array_equal(late, durations(r) > r.supply.promise_minutes)


def test_fast_pickers_and_many_riders_deliver_near_travel_time_and_on_time() -> None:
    r = variant(1, riders_per_store=200, picks_per_hour=6000.0)
    dist, _, _ = geometry(r)
    d = r.tape.deliveries
    rows = delivered_rows(r)
    metres = dist[d.resident[rows].astype(np.int64), d.store[rows].astype(np.int64)]
    minimum = metres / (r.supply.rider_speed_kmh * 1000 / 60)
    assert len(minimum) > 100
    assert np.all(durations(r) >= 0.5 * minimum)
    assert late_share(r) < 0.1


def test_a_one_minute_promise_makes_nearly_every_delivery_late() -> None:
    assert late_share(variant(1, promise_minutes=1)) >= 0.95


def test_one_slow_rider_per_store_lengthens_deliveries_and_raises_lateness() -> None:
    base = variant(1)
    slow = variant(1, riders_per_store=1, rider_speed_kmh=2.0)
    assert np.median(durations(slow)) > 2 * np.median(durations(base))
    assert late_share(slow) > late_share(base) + 0.2


def test_late_deliveries_reduce_repeat_orders() -> None:
    def repeats(promise: int, seed: int) -> int:
        r = variant(seed, promise_minutes=promise, **PLENTIFUL)
        adopters = int(np.count_nonzero(r.tape.residents.first_order_at != NEVER))
        return int(r.tape.residents.order_count.astype(np.int64).sum()) - adopters

    on_time = float(np.median([repeats(10_000, s) for s in (1, 2, 3)]))
    late = float(np.median([repeats(1, s) for s in (1, 2, 3)]))
    assert on_time > 0
    assert late < 0.9 * on_time


@pytest.mark.parametrize("make", [golden_inputs, prototype_small_inputs])
def test_without_a_supply_block_the_tape_is_the_p012_tape_and_stats_are_trivial(
    make: object,
) -> None:
    inputs = make(1)  # type: ignore[operator]
    assert inputs.scenario.supply is None
    tape, s = run_seed_with_stats(inputs)
    assert write_tape(tape) == run_seed_bytes(inputs)
    assert write_tape(run_seed(inputs)) == write_tape(tape)
    assert s.attempts == s.served == int(tape.aggregates.orders.astype(np.int64).sum())
    assert s.spilled == s.deferred == s.lost == s.late == s.unserviceable_residents == 0
    assert tape.header.delivery_rows == 0
    assert tape.header.stockout_pairs == 0
    assert len(tape.deliveries.store) == 0


@pytest.mark.parametrize("name", list(CONFIGS))
def test_supply_runs_are_deterministic_and_seed_sensitive(name: str) -> None:
    again = run_seed_bytes(CONFIGS[name](1))
    assert again == write_tape(config_result(name, 1).tape)
    assert again != write_tape(config_result(name, 2).tape)
