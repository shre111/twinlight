"""Supply (MODEL.md sections 4, 5 and 12): stock, restocks, picking, riders and deliveries.

Who attempts an order in a tick is array work. What becomes of each attempt, in minute order
with the tick's restock arrivals, deliveries and reviews, runs in one kernel over preallocated
arrays.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from twl_engine.events import (
    DELIVERY_DONE,
    MINUTE,
    RESTOCK_ARRIVAL,
    EventQueue,
    Events,
    heap_push,
    heap_remove_first,
)
from twl_engine.events import KIND as EVENT_KIND
from twl_engine.events import A as EVENT_A
from twl_engine.events import B as EVENT_B
from twl_engine.orders import MAX_ORDER_COUNT, earliest_order_minute, log_nudges, sigmoid
from twl_engine.scenario import Supply
from twl_engine.state import (
    ATTEMPTS,
    DEFERRED,
    DELIVERY_ROWS,
    LATE,
    LOST,
    NOT_OUT,
    NOT_WAITING,
    SERVED,
    SPILLED,
    STOCKOUT_ROWS,
    Sim,
    SupplyState,
)
from twl_engine.world import NO_STORE, SupplyPlan
from twl_tape import (
    DELIVERY_LATE,
    DELIVERY_SPILLED,
    FLAG_GAVE_UP,
    FLAG_SPILLED,
    FLAG_STOCKOUT,
    NEVER,
    StoresSection,
)

Streams = dict[str, np.random.Generator]

NOT_SERVED, SERVED_REPEAT, SERVED_FIRST = 0, 1, 2
NO_MINUTE = np.iinfo(np.int64).max
MAX_ORDERS_PER_TICK = np.iinfo(np.uint16).max
NOBODY = np.zeros(0, dtype=np.int64)
METRES_PER_KM = 1000.0


def orders_through_supply(
    sim: Sim, streams: Streams, queue: EventQueue, start: int, end: int, events: Events
) -> None:
    """Occasions and orders with supply on: the tick's attempts, then its supply events."""
    supply = sim.world.scenario.supply
    assert supply is not None
    who, minutes = order_attempts(sim, streams["occasions"], start, end, events)
    draws = draw_for_attempts(supply, streams["supply"], streams["riders"], who.shape[0])
    outcome = serve_attempts(sim, queue, start, end, events, who, minutes, draws)
    first = outcome == SERVED_FIRST
    if first.any():
        log_nudges(sim.residents, sim.chatter, who[first], minutes[first])


@dataclass(frozen=True, eq=False)
class AttemptDraws:
    """Per attempt: the out-of-stock uniform, whole picking minutes and the travel noise factor.

    Each is used only if the attempt gets that far (MODEL.md section 12).
    """

    hit: NDArray[np.float64]
    pick_minutes: NDArray[np.int64]
    travel_noise: NDArray[np.float64]


NO_DRAWS = AttemptDraws(
    hit=np.zeros(0, dtype=np.float64),
    pick_minutes=np.zeros(0, dtype=np.int64),
    travel_noise=np.zeros(0, dtype=np.float64),
)


def draw_for_attempts(
    supply: Supply, supply_rng: np.random.Generator, riders_rng: np.random.Generator, count: int
) -> AttemptDraws:
    """Picking takes max(1, ceil(Exp(60 / mu))) minutes; travel noise is LogNormal(0, sigma)."""
    if count == 0:
        return NO_DRAWS
    hit = supply_rng.random(count)
    picking = supply_rng.exponential(60.0 / supply.picks_per_hour, count)
    return AttemptDraws(
        hit=hit,
        pick_minutes=np.maximum(np.ceil(picking), 1.0).astype(np.int64),
        travel_noise=riders_rng.lognormal(0.0, supply.travel_noise_sigma, count),
    )


def metres_per_minute(supply: Supply) -> float:
    return supply.rider_speed_kmh * METRES_PER_KM / 60.0


def serve_attempts(
    sim: Sim,
    queue: EventQueue,
    start: int,
    end: int,
    events: Events,
    who: NDArray[np.int64],
    minutes: NDArray[np.int64],
    draws: AttemptDraws,
) -> NDArray[np.uint8]:
    """Run the tick's supply events around the attempts, which must be sorted by minute.

    Returns per attempt `NOT_SERVED`, `SERVED_REPEAT` or `SERVED_FIRST`.
    """
    plan, state, residents = sim.world.supply, sim.supply, sim.residents
    supply, params = sim.world.scenario.supply, sim.world.params
    assert plan is not None and state is not None and supply is not None
    count = who.shape[0]
    outcome = np.zeros(count, dtype=np.uint8)
    queue.reserve(count + plan.stores * ((end - start) // supply.review_minutes + 1))
    log = state.deliveries
    log.reserve(int(state.tally[DELIVERY_ROWS]) + count)
    heap_count, next_seq = serve_tick(
        start,
        end,
        start // (end - start),
        sim.world.clock.ticks,
        sim.world.clock.sim_minutes,
        events.minute,
        events.kind,
        events.a,
        events.b,
        queue.heap,
        queue.count,
        queue.next_seq,
        who,
        minutes,
        draws.hit,
        draws.pick_minutes,
        draws.travel_noise,
        outcome,
        supply.review_minutes,
        supply.lead_minutes,
        supply.wait_minutes,
        supply.promise_minutes,
        metres_per_minute(supply),
        params.p_spill,
        params.p_spill + params.p_wait,
        params.satisfaction_drop,
        plan.own_store,
        plan.spill_store,
        plan.own_distance,
        plan.spill_distance,
        plan.reorder_point,
        plan.order_up_to,
        state.stock,
        state.in_transit,
        state.restocked,
        state.picker_free,
        state.rider_free,
        state.out_since,
        state.last_stockout,
        state.stockout_store,
        state.stockout_start,
        state.stockout_end,
        log.store,
        log.resident,
        log.ordered_at,
        log.picked_at,
        log.delivered_at,
        log.outcome,
        state.tally,
        state.satisfaction,
        state.waiting_until,
        residents.first_order_at,
        residents.last_order_at,
        residents.order_count,
        residents.outcome_flags,
        state.stock_series,
        state.orders_series,
        sim.aggregates.orders,
        sim.aggregates.stores_out,
    )
    queue.adopt(heap_count, next_seq)
    return outcome


def order_attempts(
    sim: Sim, rng: np.random.Generator, start: int, end: int, events: Events
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The tick's attempts as (resident, minute), sorted by minute, then resident."""
    plan, state = sim.world.supply, sim.supply
    assert plan is not None and state is not None
    who, minutes = attempts_from_occasions(sim, plan, state, rng, start, end, events)
    if sim.world.params.order_on_awareness:
        new, aware_minutes = attempts_on_awareness(sim, plan, start, end)
        who, minutes = np.concatenate((new, who)), np.concatenate((aware_minutes, minutes))
    order = np.lexsort((who, minutes))
    return who[order], minutes[order]


def attempts_on_awareness(
    sim: Sim, plan: SupplyPlan, start: int, end: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Bass configuration (D-027): whoever became aware this tick attempts at that minute."""
    residents = sim.residents
    aware_at = residents.aware_at
    new = np.flatnonzero(
        (aware_at >= start)
        & (aware_at < end)
        & (residents.first_order_at == NEVER)
        & plan.serviceable
    )
    return new, aware_at[new].astype(np.int64)


def attempts_from_occasions(
    sim: Sim,
    plan: SupplyPlan,
    state: SupplyState,
    rng: np.random.Generator,
    start: int,
    end: int,
    events: Events,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Aware residents inside a store radius with an occasion in the tick (MODEL.md section 12).

    One who is waiting attempts without deciding once the wait is over and otherwise sits the
    occasion out. Everyone else decides with `avail` and `satisfaction` as they stand.
    """
    choice, residents, supply = sim.world.choice, sim.residents, sim.world.scenario.supply
    assert supply is not None
    chance = choice.occasion_probability[start // (end - start)]
    pool = np.flatnonzero((residents.aware_at != NEVER) & plan.serviceable)
    if chance <= 0.0 or pool.size == 0:
        return NOBODY, NOBODY
    count = int(rng.binomial(pool.size, chance))
    if count == 0:
        return NOBODY, NOBODY
    deciding = np.sort(pool[rng.choice(pool.size, size=count, replace=False, shuffle=False)])
    draws = rng.random((count, 2))
    adopted = residents.first_order_at[deciding] != NEVER
    own = plan.own_store[deciding]
    travel = plan.own_distance[deciding] / metres_per_minute(supply)
    in_promise = store_quotes(state, supply, start)[own] + travel <= supply.promise_minutes
    available = stores_in_stock(plan, state, start, events)[own] & in_promise
    first_logit = (
        choice.first_logit[deciding]
        + choice.gamma * np.log1p(residents.awareness[deciding])
        - np.where(available, 0.0, choice.delta)
    )
    repeat_logit = choice.repeat_logit[deciding] + state.satisfaction[deciding]
    wants = draws[:, 0] < sigmoid(np.where(adopted, repeat_logit, first_logit))
    retry_from = state.waiting_until[deciding]
    attempts = np.where(retry_from == NOT_WAITING, wants, retry_from < end)
    who = deciding[attempts]
    earliest = np.maximum(earliest_order_minute(residents, who, start), retry_from[attempts])
    minutes = np.minimum(
        earliest + (draws[attempts, 1] * (end - earliest)).astype(np.int64), end - 1
    )
    return who, minutes


def store_quotes(state: SupplyState, supply: Supply, start: int) -> NDArray[np.float64]:
    """Per store the quoted minutes before travel for an order placed at the start of the tick:
    the wait for the picker, the mean picking time, then the wait for a rider."""
    picked = np.maximum(state.picker_free - start, 0) + 60.0 / supply.picks_per_hour
    rider_wait = np.maximum(state.rider_free.min(axis=1) - start - picked, 0.0)
    return picked + rider_wait


def stores_in_stock(
    plan: SupplyPlan, state: SupplyState, start: int, events: Events
) -> NDArray[np.bool_]:
    """Per store: stock at the start of the tick, counting a restock that arrives that minute."""
    stock = state.stock
    arriving = (events.kind == RESTOCK_ARRIVAL) & (events.minute == start)
    if arriving.any():
        landed = np.bincount(events.a[arriving], weights=events.b[arriving], minlength=plan.stores)
        stock = stock + landed.astype(np.int64)
    return stock >= 1


@njit(cache=True)
def serve_tick(
    start: int,
    end: int,
    tick: int,
    ticks: int,
    sim_minutes: int,
    due_minute: NDArray[np.int64],
    due_kind: NDArray[np.int64],
    due_a: NDArray[np.int64],
    due_b: NDArray[np.int64],
    heap: NDArray[np.int64],
    heap_count: int,
    next_seq: int,
    who: NDArray[np.int64],
    minute: NDArray[np.int64],
    hit_draw: NDArray[np.float64],
    pick_minutes: NDArray[np.int64],
    travel_noise: NDArray[np.float64],
    outcome: NDArray[np.uint8],
    review_minutes: int,
    lead_minutes: int,
    wait_minutes: int,
    promise_minutes: int,
    metres_per_minute: float,
    p_spill: float,
    p_spill_or_wait: float,
    satisfaction_drop: float,
    own_store: NDArray[np.int64],
    spill_store: NDArray[np.int64],
    own_distance: NDArray[np.float64],
    spill_distance: NDArray[np.float64],
    reorder_point: NDArray[np.int64],
    order_up_to: NDArray[np.int64],
    stock: NDArray[np.int64],
    in_transit: NDArray[np.int64],
    restocked: NDArray[np.int64],
    picker_free: NDArray[np.int64],
    rider_free: NDArray[np.int64],
    out_since: NDArray[np.int64],
    last_stockout: NDArray[np.int64],
    stockout_store: NDArray[np.int64],
    stockout_start: NDArray[np.int64],
    stockout_end: NDArray[np.int64],
    delivery_store: NDArray[np.uint16],
    delivery_resident: NDArray[np.uint32],
    delivery_ordered_at: NDArray[np.uint32],
    delivery_picked_at: NDArray[np.uint32],
    delivery_delivered_at: NDArray[np.uint32],
    delivery_outcome: NDArray[np.uint8],
    tally: NDArray[np.int64],
    satisfaction: NDArray[np.float64],
    waiting_until: NDArray[np.int64],
    first_order_at: NDArray[np.uint32],
    last_order_at: NDArray[np.uint32],
    order_count: NDArray[np.uint16],
    outcome_flags: NDArray[np.uint8],
    stock_series: NDArray[np.float32],
    orders_series: NDArray[np.uint16],
    orders_in_tick: NDArray[np.uint32],
    stores_out: NDArray[np.uint16],
) -> tuple[int, int]:
    """Everything supply does in `[start, end)`, in minute order (MODEL.md section 12).

    At one minute events come first (the ones already due, then any the tick itself queued),
    then the review, then order attempts. Stock is taken when an order is placed, and a served
    order is picked, given a rider and logged at once; its delivery is queued as an event so a
    late one costs satisfaction at the delivery minute. Returns the queue's new size and next
    seq.
    """
    due = 0
    attempt = 0
    review = -(-start // review_minutes) * review_minutes
    while True:
        due_at = due_minute[due] if due < due_minute.shape[0] else NO_MINUTE
        queued_at = heap[0, MINUTE] if heap_count > 0 and heap[0, MINUTE] < end else NO_MINUTE
        event_at = min(due_at, queued_at)
        review_at = review if review < end else NO_MINUTE
        attempt_at = minute[attempt] if attempt < who.shape[0] else NO_MINUTE
        if event_at <= review_at and event_at <= attempt_at:
            if event_at == NO_MINUTE:
                break
            if due_at <= queued_at:
                kind, a, b = due_kind[due], due_a[due], due_b[due]
                due += 1
            else:
                kind, a, b = heap[0, EVENT_KIND], heap[0, EVENT_A], heap[0, EVENT_B]
                heap_count = heap_remove_first(heap, heap_count)
            if kind == RESTOCK_ARRIVAL:
                stock[a] += b
                in_transit[a] -= b
                restocked[a] += b
                if out_since[a] != NOT_OUT and stock[a] >= 1:
                    close_stockout(
                        a,
                        event_at,
                        out_since,
                        last_stockout,
                        stockout_store,
                        stockout_start,
                        stockout_end,
                        tally,
                    )
            elif kind == DELIVERY_DONE and delivery_outcome[a] & DELIVERY_LATE:
                satisfaction[delivery_resident[a]] -= satisfaction_drop
                tally[LATE] += 1
        elif review_at <= attempt_at:
            pushed = review_stores(
                review,
                lead_minutes,
                reorder_point,
                order_up_to,
                stock,
                in_transit,
                heap,
                heap_count,
                next_seq,
            )
            heap_count += pushed
            next_seq += pushed
            review += review_minutes
        else:
            resident = who[attempt]
            store = place_attempt(
                resident,
                attempt_at,
                hit_draw[attempt],
                wait_minutes,
                p_spill,
                p_spill_or_wait,
                satisfaction_drop,
                own_store,
                spill_store,
                stock,
                tally,
                satisfaction,
                waiting_until,
                outcome_flags,
            )
            if store != NO_STORE:
                stock[store] -= 1
                if stock[store] < 1:
                    out_since[store] = attempt_at
                if orders_series[store * ticks + tick] < MAX_ORDERS_PER_TICK:
                    orders_series[store * ticks + tick] += 1
                orders_in_tick[tick] += 1
                outcome[attempt] = count_order(
                    resident, attempt_at, first_order_at, last_order_at, order_count
                )
                spilled = store != own_store[resident]
                distance = spill_distance[resident] if spilled else own_distance[resident]
                travel = max(1, math.ceil(distance / metres_per_minute * travel_noise[attempt]))
                picked, delivered = pick_and_ride(
                    store, attempt_at, pick_minutes[attempt], travel, picker_free, rider_free
                )
                row = tally[DELIVERY_ROWS]
                tally[DELIVERY_ROWS] = row + 1
                delivery_store[row] = store
                delivery_resident[row] = resident
                delivery_ordered_at[row] = attempt_at
                delivery_picked_at[row] = picked if picked < sim_minutes else NEVER
                delivery_delivered_at[row] = delivered if delivered < sim_minutes else NEVER
                delivery_outcome[row] = DELIVERY_SPILLED if spilled else 0
                if delivered < sim_minutes and delivered - attempt_at > promise_minutes:
                    delivery_outcome[row] |= DELIVERY_LATE
                    heap_push(heap, heap_count, delivered, next_seq, DELIVERY_DONE, row, 0)
                    heap_count += 1
                    next_seq += 1
            attempt += 1
    out = 0
    for store in range(stock.shape[0]):
        stock_series[store * ticks + tick] = stock[store]
        if stock[store] < 1:
            out += 1
    stores_out[tick] = out
    return heap_count, next_seq


@njit(cache=True)
def place_attempt(
    resident: int,
    minute: int,
    hit_draw: float,
    wait_minutes: int,
    p_spill: float,
    p_spill_or_wait: float,
    satisfaction_drop: float,
    own_store: NDArray[np.int64],
    spill_store: NDArray[np.int64],
    stock: NDArray[np.int64],
    tally: NDArray[np.int64],
    satisfaction: NDArray[np.float64],
    waiting_until: NDArray[np.int64],
    outcome_flags: NDArray[np.uint8],
) -> int:
    """The store that serves the attempt, or `NO_STORE` when it is deferred or lost.

    Out of stock at the own store (MODEL.md section 12): one draw sends the order to the spill
    store if that has stock, else makes the resident wait, else loses the sale; every such hit
    costs satisfaction and sets the stockout flag.
    """
    tally[ATTEMPTS] += 1
    waiting_until[resident] = NOT_WAITING
    store = own_store[resident]
    if stock[store] >= 1:
        tally[SERVED] += 1
        return store
    outcome_flags[resident] |= FLAG_STOCKOUT
    satisfaction[resident] -= satisfaction_drop
    spill = spill_store[resident]
    if hit_draw < p_spill and spill != NO_STORE and stock[spill] >= 1:
        outcome_flags[resident] |= FLAG_SPILLED
        tally[SERVED] += 1
        tally[SPILLED] += 1
        return spill
    if hit_draw < p_spill_or_wait:
        waiting_until[resident] = minute + wait_minutes
        tally[DEFERRED] += 1
    else:
        outcome_flags[resident] |= FLAG_GAVE_UP
        tally[LOST] += 1
    return NO_STORE


@njit(cache=True)
def pick_and_ride(
    store: int,
    minute: int,
    pick_minutes: int,
    travel: int,
    picker_free: NDArray[np.int64],
    rider_free: NDArray[np.int64],
) -> tuple[int, int]:
    """Minutes the order is picked and delivered (MODEL.md section 12).

    One picker per store serves in order; the order leaves with the rider free earliest (lowest
    index on a tie), who is free again after twice the travel time.
    """
    picked = max(minute, picker_free[store]) + pick_minutes
    picker_free[store] = picked
    rider = 0
    for candidate in range(1, rider_free.shape[1]):
        if rider_free[store, candidate] < rider_free[store, rider]:
            rider = candidate
    leaves = max(picked, rider_free[store, rider])
    rider_free[store, rider] = leaves + 2 * travel
    return picked, leaves + travel


@njit(cache=True)
def count_order(
    resident: int,
    minute: int,
    first_order_at: NDArray[np.uint32],
    last_order_at: NDArray[np.uint32],
    order_count: NDArray[np.uint16],
) -> int:
    first = first_order_at[resident] == NEVER
    if first:
        first_order_at[resident] = minute
    last_order_at[resident] = minute
    if order_count[resident] < MAX_ORDER_COUNT:
        order_count[resident] += 1
    return SERVED_FIRST if first else SERVED_REPEAT


@njit(cache=True)
def review_stores(
    minute: int,
    lead_minutes: int,
    reorder_point: NDArray[np.int64],
    order_up_to: NDArray[np.int64],
    stock: NDArray[np.int64],
    in_transit: NDArray[np.int64],
    heap: NDArray[np.int64],
    heap_count: int,
    next_seq: int,
) -> int:
    """(s, S) review: a store at or below s orders up to S less what is already on its way.

    Each order is queued as a restock arrival `lead_minutes` later; returns how many were.
    """
    pushed = 0
    for store in range(stock.shape[0]):
        if stock[store] > reorder_point[store]:
            continue
        units = order_up_to[store] - stock[store] - in_transit[store]
        if units > 0:
            in_transit[store] += units
            heap_push(
                heap,
                heap_count + pushed,
                minute + lead_minutes,
                next_seq + pushed,
                RESTOCK_ARRIVAL,
                store,
                units,
            )
            pushed += 1
    return pushed


@njit(cache=True)
def close_stockout(
    store: int,
    minute: int,
    out_since: NDArray[np.int64],
    last_stockout: NDArray[np.int64],
    stockout_store: NDArray[np.int64],
    stockout_start: NDArray[np.int64],
    stockout_end: NDArray[np.int64],
    tally: NDArray[np.int64],
) -> None:
    """Log the store's stockout as the end-exclusive `[out_since, minute)`.

    One that starts the minute the store's previous one ended extends that row, and an empty
    one is dropped (ARCHITECTURE.md section 4).
    """
    began = out_since[store]
    out_since[store] = NOT_OUT
    if minute <= began:
        return
    row = last_stockout[store]
    if row >= 0 and stockout_end[row] == began:
        stockout_end[row] = minute
        return
    row = tally[STOCKOUT_ROWS]
    stockout_store[row] = store
    stockout_start[row] = began
    stockout_end[row] = minute
    last_stockout[store] = row
    tally[STOCKOUT_ROWS] = row + 1


@njit(cache=True)
def close_open_stockouts(
    sim_minutes: int,
    out_since: NDArray[np.int64],
    last_stockout: NDArray[np.int64],
    stockout_store: NDArray[np.int64],
    stockout_start: NDArray[np.int64],
    stockout_end: NDArray[np.int64],
    tally: NDArray[np.int64],
) -> None:
    """A stockout still open when the run ends is closed at `sim_minutes`."""
    for store in range(out_since.shape[0]):
        if out_since[store] != NOT_OUT:
            close_stockout(
                store,
                sim_minutes,
                out_since,
                last_stockout,
                stockout_store,
                stockout_start,
                stockout_end,
                tally,
            )


def stores_section(state: SupplyState, sim_minutes: int) -> StoresSection:
    """Per-store series and stockouts as CSR pairs, each store's sorted by start."""
    close_open_stockouts(
        sim_minutes,
        state.out_since,
        state.last_stockout,
        state.stockout_store,
        state.stockout_start,
        state.stockout_end,
        state.tally,
    )
    rows = int(state.tally[STOCKOUT_ROWS])
    store = state.stockout_store[:rows]
    by_store = np.argsort(store, kind="stable")
    pairs = np.stack((state.stockout_start[:rows], state.stockout_end[:rows]), axis=1)[by_store]
    offsets = np.zeros(state.stock.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(store, minlength=state.stock.shape[0]), out=offsets[1:])
    return StoresSection(
        stock=state.stock_series,
        orders=state.orders_series,
        stockout_offsets=offsets.astype(np.uint32),
        stockout_intervals=pairs.reshape(-1).astype(np.uint32),
    )
