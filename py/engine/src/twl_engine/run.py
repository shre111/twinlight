from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.clock import Clock
from twl_engine.events import EventQueue
from twl_engine.fixtures import ToyCity
from twl_engine.rng import spawn_streams
from twl_engine.scenario import Params, Scenario
from twl_engine.state import ResidentState, new_sim
from twl_engine.tick import advance_tick
from twl_engine.world import build_world
from twl_tape import (
    NEVER,
    AggregatesSection,
    DeliveriesSection,
    ResidentsSection,
    StoresSection,
    Tape,
    TapeHeader,
    derive_run_id,
    write_tape,
)

ENGINE_VERSION = "twl-engine 1"
STREAMS = ("campaign", "wom", "occasions", "supply", "riders", "params")


@dataclass(frozen=True)
class RunInputs:
    city: ToyCity
    city_pack_hash: bytes
    scenario: Scenario
    params: Params
    seed: int


@dataclass(frozen=True, eq=False)
class SupplyStats:
    """What supply did with the run's order attempts (MODEL.md section 12 conservation).

    attempts = served + deferred + lost; spilled orders are among the served and late ones among
    those delivered before the end. Per store: initial_stock + restocked - stock_end =
    served_by_store. Without supply every order is an attempt that was served and the rest is 0.
    """

    attempts: int
    served: int
    spilled: int
    deferred: int
    lost: int
    late: int
    unserviceable_residents: int
    catchment: NDArray[np.int64]
    initial_stock: NDArray[np.int64]
    restocked: NDArray[np.int64]
    stock_end: NDArray[np.int64]
    served_by_store: NDArray[np.int64]


def run_seed(inputs: RunInputs) -> Tape:
    """Run one seed to the end of the scenario and return its tape."""
    return run_seed_with_stats(inputs)[0]


def run_seed_with_stats(inputs: RunInputs) -> tuple[Tape, SupplyStats]:
    scenario = inputs.scenario
    clock = Clock(sim_minutes=scenario.sim_minutes, tick_minutes=scenario.tick_minutes)
    streams = spawn_streams(inputs.seed, STREAMS)
    sim = new_sim(build_world(inputs.city, scenario, inputs.params, clock, streams["params"]))
    queue = EventQueue()
    for tick in range(clock.ticks):
        start = clock.tick_start(tick)
        end = start + clock.tick_minutes
        events = queue.pop_until(end - 1)
        advance_tick(sim, streams, queue, start, end, events)
        record_aggregates(sim.residents, sim.aggregates, tick)
    chatter = sim.chatter.section()
    stores = inputs.city.stores.x.shape[0]
    tape = Tape(
        header=tape_header(inputs, clock, chatter_rows=chatter.at.shape[0]),
        residents=residents_section(sim.residents),
        chatter=chatter,
        deliveries=DeliveriesSection(
            store=np.zeros(0, np.uint16),
            resident=np.zeros(0, np.uint32),
            ordered_at=np.zeros(0, np.uint32),
            picked_at=np.zeros(0, np.uint32),
            delivered_at=np.zeros(0, np.uint32),
            outcome=np.zeros(0, np.uint8),
        ),
        stores=empty_stores_section(stores, clock.ticks),
        aggregates=sim.aggregates,
    )
    return tape, unlimited_supply_stats(int(sim.aggregates.orders.sum()), stores)


def unlimited_supply_stats(orders: int, stores: int) -> SupplyStats:
    none = np.zeros(stores, dtype=np.int64)
    return SupplyStats(
        attempts=orders,
        served=orders,
        spilled=0,
        deferred=0,
        lost=0,
        late=0,
        unserviceable_residents=0,
        catchment=none,
        initial_stock=none.copy(),
        restocked=none.copy(),
        stock_end=none.copy(),
        served_by_store=none.copy(),
    )


def run_seed_bytes(inputs: RunInputs) -> bytes:
    return write_tape(run_seed(inputs))


def tape_header(inputs: RunInputs, clock: Clock, chatter_rows: int) -> TapeHeader:
    scenario_hash = inputs.scenario.hash()
    return TapeHeader(
        run_id=derive_run_id(
            inputs.city_pack_hash, scenario_hash, inputs.params.hash(), inputs.seed
        ),
        seed=inputs.seed,
        engine_version=ENGINE_VERSION,
        city_pack_hash=inputs.city_pack_hash,
        scenario_hash=scenario_hash,
        sim_minutes=clock.sim_minutes,
        tick_minutes=clock.tick_minutes,
        residents=inputs.city.residents.x.shape[0],
        stores=inputs.city.stores.x.shape[0],
        chatter_rows=chatter_rows,
        delivery_rows=0,
        stockout_pairs=0,
        flags=0,
    )


def record_aggregates(state: ResidentState, aggregates: AggregatesSection, tick: int) -> None:
    aggregates.aware[tick] = np.count_nonzero(state.aware_at != NEVER)
    aggregates.adopted[tick] = np.count_nonzero(state.first_order_at != NEVER)


def residents_section(state: ResidentState) -> ResidentsSection:
    return ResidentsSection(
        aware_at=state.aware_at,
        first_order_at=state.first_order_at,
        order_count=state.order_count,
        heard_from=state.heard_from,
        first_source=state.first_source,
        outcome_flags=state.outcome_flags,
    )


def empty_stores_section(stores: int, ticks: int) -> StoresSection:
    return StoresSection(
        stock=np.zeros(stores * ticks, np.float32),
        orders=np.zeros(stores * ticks, np.uint16),
        stockout_offsets=np.zeros(stores + 1, np.uint32),
        stockout_intervals=np.zeros(0, np.uint32),
    )
