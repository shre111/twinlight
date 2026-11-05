from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.world import World
from twl_tape import NEVER, AggregatesSection, ChatterSection


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
class Sim:
    """What every tick hook works on: the static world and the run's mutable state."""

    world: World
    residents: ResidentState
    chatter: ChatterLog
    aggregates: AggregatesSection


def new_sim(world: World) -> Sim:
    ticks = world.clock.ticks
    return Sim(
        world=world,
        residents=new_resident_state(world.residents, len(world.scenario.campaigns)),
        chatter=new_chatter_log(world.residents),
        aggregates=AggregatesSection(
            aware=np.zeros(ticks, np.uint32),
            adopted=np.zeros(ticks, np.uint32),
            orders=np.zeros(ticks, np.uint32),
            stores_out=np.zeros(ticks, np.uint16),
            rain_intensity=np.zeros(ticks, np.float32),
        ),
    )
