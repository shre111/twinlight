"""Awareness decay and paid exposure (MODEL.md sections 3 and 11)."""

import numpy as np

from twl_engine.events import EventQueue, Events
from twl_engine.scenario import MINUTES_PER_DAY
from twl_engine.state import Sim
from twl_engine.world import NO_CAP
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
        cum = plan.cum_target[c]
        landing = np.searchsorted(cum, rng.random(count) * cum[-1], side="right")
        who, landed = np.unique(np.minimum(landing, plan.last_target[c]), return_counts=True)
        if plan.cap[c] != NO_CAP:
            seen = residents.impressions_today[c]
            landed = np.minimum(landed, plan.cap[c] - seen[who])
            seen[who] += landed.astype(np.int32)
        residents.awareness[who] += landed * plan.gain[who]
        unaware = residents.aware_at[who] == NEVER
        crossed = who[unaware & (residents.awareness[who] >= sim.world.params.theta_aware)]
        residents.aware_at[crossed] = rng.integers(start, end, size=crossed.size, dtype=np.uint32)
        residents.first_source[crossed] = SOURCE_PAID
