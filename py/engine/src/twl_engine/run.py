from dataclasses import dataclass

import numpy as np

from twl_engine.clock import Clock
from twl_engine.events import EventQueue
from twl_engine.fixtures import ToyCity
from twl_engine.rng import spawn_streams
from twl_engine.scenario import Params, Scenario
from twl_engine.state import ResidentState, new_resident_state
from twl_engine.tick import advance_tick
from twl_tape import (
    NEVER,
    AggregatesSection,
    ChatterSection,
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


def run_seed(inputs: RunInputs) -> Tape:
    """Run one seed to the end of the scenario and return its tape."""
    scenario = inputs.scenario
    clock = Clock(sim_minutes=scenario.sim_minutes, tick_minutes=scenario.tick_minutes)
    streams = spawn_streams(inputs.seed, STREAMS)
    state = new_resident_state(inputs.city.residents.x.shape[0])
    queue = EventQueue()
    aggregates = new_aggregates(clock.ticks)
    for tick in range(clock.ticks):
        start = clock.tick_start(tick)
        end = start + clock.tick_minutes
        events = queue.pop_until(end - 1)
        advance_tick(state, streams, queue, start, end, events)
        record_aggregates(state, aggregates, tick)
    return Tape(
        header=tape_header(inputs, clock),
        residents=residents_section(state),
        chatter=ChatterSection(
            from_=np.zeros(0, np.uint32), to=np.zeros(0, np.uint32), at=np.zeros(0, np.uint32)
        ),
        deliveries=DeliveriesSection(
            store=np.zeros(0, np.uint16),
            resident=np.zeros(0, np.uint32),
            ordered_at=np.zeros(0, np.uint32),
            picked_at=np.zeros(0, np.uint32),
            delivered_at=np.zeros(0, np.uint32),
            outcome=np.zeros(0, np.uint8),
        ),
        stores=empty_stores_section(inputs.city.stores.x.shape[0], clock.ticks),
        aggregates=aggregates,
    )


def run_seed_bytes(inputs: RunInputs) -> bytes:
    return write_tape(run_seed(inputs))


def tape_header(inputs: RunInputs, clock: Clock) -> TapeHeader:
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
        chatter_rows=0,
        delivery_rows=0,
        stockout_pairs=0,
        flags=0,
    )


def new_aggregates(ticks: int) -> AggregatesSection:
    return AggregatesSection(
        aware=np.zeros(ticks, np.uint32),
        adopted=np.zeros(ticks, np.uint32),
        orders=np.zeros(ticks, np.uint32),
        stores_out=np.zeros(ticks, np.uint16),
        rain_intensity=np.zeros(ticks, np.float32),
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
