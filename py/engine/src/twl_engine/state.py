import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.world import SupplyPlan, World
from twl_tape import NEVER, AggregatesSection, ChatterSection, DeliveriesSection

NOT_OUT = -1
NOT_WAITING = -1
ATTEMPTS, SERVED, SPILLED, DEFERRED, LOST, LATE, STOCKOUT_ROWS, DELIVERY_ROWS = range(8)
TALLIES = 8
INITIAL_DELIVERY_ROWS = 1024


@dataclass(eq=False)
class ResidentState:
    """Per-resident struct of arrays; the first six columns are written to the tape as they are.

    `awareness` is A_r (MODEL.md section 3). `impressions_today` is [campaign, resident] for the
    frequency cap. `last_talk_*` is the last talk received, kept for nudged rows (D-028).
    `talk_rate_cum` is scratch: the running sum of this tick's talker rates.
    """

    aware_at: NDArray[np.uint32]
    first_order_at: NDArray[np.uint32]
    order_count: NDArray[np.uint16]
    heard_from: NDArray[np.uint32]
    first_source: NDArray[np.uint8]
    outcome_flags: NDArray[np.uint8]
    awareness: NDArray[np.float64]
    last_order_at: NDArray[np.uint32]
    last_talk_from: NDArray[np.uint32]
    last_talk_at: NDArray[np.uint32]
    impressions_today: NDArray[np.int32]
    talk_rate_cum: NDArray[np.float64]

    @property
    def count(self) -> int:
        return self.aware_at.shape[0]


def new_resident_state(n: int, campaigns: int = 0) -> ResidentState:
    """Nobody aware, nobody ordered, awareness A_r = 0 (MODEL.md section 3)."""
    return ResidentState(
        aware_at=np.full(n, NEVER, dtype=np.uint32),
        first_order_at=np.full(n, NEVER, dtype=np.uint32),
        order_count=np.zeros(n, dtype=np.uint16),
        heard_from=np.full(n, NEVER, dtype=np.uint32),
        first_source=np.zeros(n, dtype=np.uint8),
        outcome_flags=np.zeros(n, dtype=np.uint8),
        awareness=np.zeros(n, dtype=np.float64),
        last_order_at=np.full(n, NEVER, dtype=np.uint32),
        last_talk_from=np.full(n, NEVER, dtype=np.uint32),
        last_talk_at=np.full(n, NEVER, dtype=np.uint32),
        impressions_today=np.zeros((campaigns, n), dtype=np.int32),
        talk_rate_cum=np.zeros(n, dtype=np.float64),
    )


@dataclass(eq=False)
class ChatterLog:
    """Recorded talks in the order they were found: at most one heard and one nudged row each."""

    from_: NDArray[np.uint32]
    to: NDArray[np.uint32]
    at: NDArray[np.uint32]
    kind: NDArray[np.uint8]
    rows: int = 0

    def section(self) -> ChatterSection:
        """Rows sorted by `at`, then `to`, then `kind` (stable)."""
        m = self.rows
        order = np.lexsort((self.kind[:m], self.to[:m], self.at[:m]))
        return ChatterSection(
            from_=self.from_[:m][order],
            to=self.to[:m][order],
            at=self.at[:m][order],
            kind=self.kind[:m][order],
        )


def new_chatter_log(residents: int) -> ChatterLog:
    capacity = 2 * residents
    return ChatterLog(
        from_=np.zeros(capacity, dtype=np.uint32),
        to=np.zeros(capacity, dtype=np.uint32),
        at=np.zeros(capacity, dtype=np.uint32),
        kind=np.zeros(capacity, dtype=np.uint8),
    )


@dataclass(eq=False)
class DeliveryLog:
    """The tape's delivery columns, one row per served order in the order they were placed.

    The row count is `SupplyState.tally[DELIVERY_ROWS]`; the arrays are capacity.
    """

    store: NDArray[np.uint16]
    resident: NDArray[np.uint32]
    ordered_at: NDArray[np.uint32]
    picked_at: NDArray[np.uint32]
    delivered_at: NDArray[np.uint32]
    outcome: NDArray[np.uint8]

    def reserve(self, rows: int) -> None:
        """Grow by doubling until `rows` fit."""
        capacity = max(self.store.shape[0], 1)
        if rows <= self.store.shape[0]:
            return
        while capacity < rows:
            capacity *= 2
        for field in dataclasses.fields(self):
            column = getattr(self, field.name)
            grown = np.zeros(capacity, dtype=column.dtype)
            grown[: column.shape[0]] = column
            setattr(self, field.name, grown)

    def section(self, rows: int) -> DeliveriesSection:
        return DeliveriesSection(
            store=self.store[:rows].copy(),
            resident=self.resident[:rows].copy(),
            ordered_at=self.ordered_at[:rows].copy(),
            picked_at=self.picked_at[:rows].copy(),
            delivered_at=self.delivered_at[:rows].copy(),
            outcome=self.outcome[:rows].copy(),
        )


def new_delivery_log(capacity: int = INITIAL_DELIVERY_ROWS) -> DeliveryLog:
    return DeliveryLog(
        store=np.zeros(capacity, dtype=np.uint16),
        resident=np.zeros(capacity, dtype=np.uint32),
        ordered_at=np.zeros(capacity, dtype=np.uint32),
        picked_at=np.zeros(capacity, dtype=np.uint32),
        delivered_at=np.zeros(capacity, dtype=np.uint32),
        outcome=np.zeros(capacity, dtype=np.uint8),
    )


@dataclass(eq=False)
class SupplyState:
    """Stores, residents' supply memory and the logs the tape is cut from (MODEL.md section 12).

    `out_since` is the minute a store ran out (`NOT_OUT` while it has stock) and `waiting_until`
    the minute from which a resident tries again (`NOT_WAITING` otherwise). `picker_free` and
    `rider_free` ([store, rider]) are the minutes from which each is free. Stockouts that ended
    are rows of `stockout_*`; `last_stockout` is each store's latest row. `tally` counts attempts
    and their outcomes and holds the row count of each log.
    """

    stock: NDArray[np.int64]
    in_transit: NDArray[np.int64]
    restocked: NDArray[np.int64]
    picker_free: NDArray[np.int64]
    rider_free: NDArray[np.int64]
    out_since: NDArray[np.int64]
    last_stockout: NDArray[np.int64]
    satisfaction: NDArray[np.float64]
    waiting_until: NDArray[np.int64]
    stock_series: NDArray[np.float32]
    orders_series: NDArray[np.uint16]
    stockout_store: NDArray[np.int64]
    stockout_start: NDArray[np.int64]
    stockout_end: NDArray[np.int64]
    deliveries: DeliveryLog
    tally: NDArray[np.int64]


def new_supply_state(plan: SupplyPlan, world: World) -> SupplyState:
    """Stores at their initial stock, nobody waiting; a store that starts empty is out from 0."""
    supply = world.scenario.supply
    assert supply is not None
    stores, ticks = plan.stores, world.clock.ticks
    stockouts = stores * (world.clock.sim_minutes // supply.review_minutes + 2)
    return SupplyState(
        stock=plan.initial_stock.copy(),
        in_transit=np.zeros(stores, dtype=np.int64),
        restocked=np.zeros(stores, dtype=np.int64),
        picker_free=np.zeros(stores, dtype=np.int64),
        rider_free=np.zeros((stores, supply.riders_per_store), dtype=np.int64),
        out_since=np.where(plan.initial_stock < 1, 0, NOT_OUT).astype(np.int64),
        last_stockout=np.full(stores, -1, dtype=np.int64),
        satisfaction=np.zeros(world.residents, dtype=np.float64),
        waiting_until=np.full(world.residents, NOT_WAITING, dtype=np.int64),
        stock_series=np.zeros(stores * ticks, dtype=np.float32),
        orders_series=np.zeros(stores * ticks, dtype=np.uint16),
        stockout_store=np.zeros(stockouts, dtype=np.int64),
        stockout_start=np.zeros(stockouts, dtype=np.int64),
        stockout_end=np.zeros(stockouts, dtype=np.int64),
        deliveries=new_delivery_log(),
        tally=np.zeros(TALLIES, dtype=np.int64),
    )


@dataclass(eq=False)
class Sim:
    """What every tick hook works on: the static world and the run's mutable state."""

    world: World
    residents: ResidentState
    chatter: ChatterLog
    aggregates: AggregatesSection
    supply: SupplyState | None


def new_sim(world: World) -> Sim:
    ticks = world.clock.ticks
    return Sim(
        world=world,
        residents=new_resident_state(world.residents, len(world.scenario.campaigns)),
        chatter=new_chatter_log(world.residents),
        supply=None if world.supply is None else new_supply_state(world.supply, world),
        aggregates=AggregatesSection(
            aware=np.zeros(ticks, np.uint32),
            adopted=np.zeros(ticks, np.uint32),
            orders=np.zeros(ticks, np.uint32),
            stores_out=np.zeros(ticks, np.uint16),
            rain_intensity=np.zeros(ticks, np.float32),
        ),
    )
