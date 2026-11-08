"""Awareness decay and paid exposure (MODEL.md sections 3 and 11)."""

import numpy as np
from numpy.typing import NDArray

from twl_engine.events import EventQueue, Events
from twl_engine.scenario import MINUTES_PER_DAY
from twl_engine.state import Sim
from twl_engine.world import NO_CAP, PaidPlan
from twl_tape import NEVER, SOURCE_PAID

Streams = dict[str, np.random.Generator]


def decay_awareness(
    sim: Sim, streams: Streams, queue: EventQueue, start: int, end: int, events: Events
) -> None:
    if sim.world.decay_per_tick != 1.0:
        sim.residents.awareness *= sim.world.decay_per_tick


def paid_exposure(
    sim: Sim, streams: Streams, queue: EventQueue, start: int, end: int, events: Events
) -> None:
    """Poisson impressions per campaign, landed by targeting weight up to the daily cap.

    Impressions on a resident already at the cap are lost. A resident taken over `theta_aware`
    becomes aware at a minute drawn inside the tick.
    """
    plan, residents = sim.world.paid, sim.residents
    tick_minutes = end - start
    if start // MINUTES_PER_DAY != (start - tick_minutes) // MINUTES_PER_DAY:
        residents.impressions_today[:] = 0
    tick = start // tick_minutes
    rng = streams["campaign"]
    for c in range(plan.per_tick.shape[0]):
        expected = plan.per_tick[c, tick]
        if expected <= 0.0:
            continue
        count = int(rng.poisson(expected))
        if count == 0:
            continue
        if plan.two_stage[c]:
            hour = start % MINUTES_PER_DAY // 60
            landing = _land_by_segment(
                plan, c, rng.multinomial(count, plan.segment_share[c, hour]), rng
            )
        else:
            cum = plan.cum_target[c]
            landing = np.minimum(
                np.searchsorted(cum, rng.random(count) * cum[-1], side="right"), plan.last_target[c]
            )
        who, landed = np.unique(landing, return_counts=True)
        if plan.cap[c] != NO_CAP:
            seen = residents.impressions_today[c]
            landed = np.minimum(landed, plan.cap[c] - seen[who])
            seen[who] += landed.astype(np.int32)
        residents.awareness[who] += landed * plan.gain[c, who]
        unaware = residents.aware_at[who] == NEVER
        crossed = who[unaware & (residents.awareness[who] >= sim.world.params.theta_aware)]
        residents.aware_at[crossed] = rng.integers(start, end, size=crossed.size, dtype=np.uint32)
        residents.first_source[crossed] = SOURCE_PAID


def _land_by_segment(
    plan: PaidPlan, c: int, per_segment: NDArray[np.int64], rng: np.random.Generator
) -> NDArray[np.int64]:
    """Residents hit by each segment's share of the tick's impressions, by targeting within it."""
    landed: list[NDArray[np.int64]] = []
    for s in np.flatnonzero(per_segment):
        lo, hi = int(plan.segment_start[s]), int(plan.segment_start[s + 1])
        cum = plan.segment_cum[c, lo:hi]
        slot = lo + np.searchsorted(cum, rng.random(int(per_segment[s])) * cum[-1], side="right")
        landed.append(plan.by_segment[np.minimum(slot, plan.segment_last[c, s])])
    return np.concatenate(landed) if landed else np.zeros(0, dtype=np.int64)
