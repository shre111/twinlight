"""Occasions, first and repeat orders (MODEL.md sections 4 and 11) and nudged rows (D-028).

Until P-013 an order is counted at its occasion: no stock, no delivery.
"""

import numpy as np
from numpy.typing import NDArray

from twl_engine.events import EventQueue, Events
from twl_engine.state import ChatterLog, ResidentState, Sim
from twl_engine.world import ChoiceModel
from twl_tape import CHATTER_NUDGED, NEVER, SOURCE_WOM

Streams = dict[str, np.random.Generator]

NUDGE_WINDOW_MINUTES = 72 * 60
MAX_ORDER_COUNT = np.iinfo(np.uint16).max
NOBODY = np.zeros(0, dtype=np.int64)


def occasions_and_orders(
    sim: Sim, streams: Streams, queue: EventQueue, start: int, end: int, events: Events
) -> None:
    """One decision for every aware resident with an occasion in the tick."""
    choice, residents = sim.world.choice, sim.residents
    tick = start // (end - start)
    if sim.world.params.order_on_awareness:
        order_on_awareness(sim, start, end, tick)
    rng = streams["occasions"]
    pool = np.flatnonzero(residents.aware_at != NEVER)
    deciding = residents_with_occasions(choice, pool, tick, rng)
    count = deciding.size
    if count == 0:
        return
    draws = rng.random((count, 2))
    adopted = residents.first_order_at[deciding] != NEVER
    first_logit = choice.first_logit[deciding] + choice.gamma * np.log1p(
        residents.awareness[deciding]
    )
    logit = np.where(adopted, choice.repeat_logit[deciding], first_logit)
    orders = draws[:, 0] < sigmoid(logit)
    who = deciding[orders]
    earliest = earliest_order_minute(residents, who, start)
    minutes = np.minimum(earliest + (draws[orders, 1] * (end - earliest)).astype(np.int64), end - 1)
    place_orders(sim, who, minutes, tick)


def residents_with_occasions(
    choice: ChoiceModel, pool: NDArray[np.int64], tick: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """The residents of `pool` (ascending) with an occasion in the tick, ascending.

    Residents who share an occasion probability have their occasions as a binomial count drawn
    without replacement, the same law as an independent draw each: one count over the whole pool
    when every segment shares the tick's probability, otherwise one per segment in segment order
    (MODEL.md section 11, D-036).
    """
    chances = choice.occasion_probability[tick]
    if choice.occasion_shared[tick]:
        return np.sort(_draw_occasions(pool, chances[0], rng))
    segment = choice.segment[pool]
    groups = [_draw_occasions(pool[segment == s], chances[s], rng) for s in range(chances.size)]
    return np.sort(np.concatenate(groups))


def _draw_occasions(
    group: NDArray[np.int64], chance: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    if chance <= 0.0 or group.size == 0:
        return NOBODY
    count = int(rng.binomial(group.size, chance))
    if count == 0:
        return NOBODY
    return group[rng.choice(group.size, size=count, replace=False, shuffle=False)]


def order_on_awareness(sim: Sim, start: int, end: int, tick: int) -> None:
    """Bass configuration (D-027): whoever became aware this tick orders at that minute."""
    residents = sim.residents
    aware_at = residents.aware_at
    new = np.flatnonzero(
        (aware_at >= start) & (aware_at < end) & (residents.first_order_at == NEVER)
    )
    place_orders(sim, new, aware_at[new].astype(np.int64), tick)


def place_orders(sim: Sim, who: NDArray[np.int64], minutes: NDArray[np.int64], tick: int) -> None:
    """Count one order each for the distinct residents `who`; first orders may reveal a nudge."""
    residents = sim.residents
    first = residents.first_order_at[who] == NEVER
    log_nudges(residents, sim.chatter, who[first], minutes[first])
    residents.first_order_at[who[first]] = minutes[first]
    residents.last_order_at[who] = minutes
    residents.order_count[who] = np.minimum(
        residents.order_count[who].astype(np.int64) + 1, MAX_ORDER_COUNT
    )
    sim.aggregates.orders[tick] += who.size


def log_nudges(
    residents: ResidentState, log: ChatterLog, who: NDArray[np.int64], minutes: NDArray[np.int64]
) -> None:
    """Log the last talk each first-time buyer received in the 72 hours before ordering,
    unless it is the talk that made them aware (already logged as heard)."""
    talk_at = residents.last_talk_at[who].astype(np.int64)
    talk_from = residents.last_talk_from[who]
    received = residents.last_talk_at[who] != NEVER
    recent = received & (talk_at <= minutes) & (minutes - talk_at <= NUDGE_WINDOW_MINUTES)
    heard = (
        (residents.first_source[who] == SOURCE_WOM)
        & (residents.aware_at[who] == residents.last_talk_at[who])
        & (residents.heard_from[who] == talk_from)
    )
    nudged = recent & ~heard
    rows = slice(log.rows, log.rows + int(nudged.sum()))
    log.from_[rows] = talk_from[nudged]
    log.to[rows] = who[nudged]
    log.at[rows] = talk_at[nudged]
    log.kind[rows] = CHATTER_NUDGED
    log.rows = rows.stop


def earliest_order_minute(
    residents: ResidentState, who: NDArray[np.int64], start: int
) -> NDArray[np.int64]:
    """An order follows everything that fed its decision this tick: awareness and the last talk."""
    aware_at = residents.aware_at[who].astype(np.int64)
    talk_at = residents.last_talk_at[who]
    talked = np.where(talk_at == NEVER, 0, talk_at).astype(np.int64)
    return np.maximum(start, np.maximum(aware_at, talked))


def sigmoid(logit: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-np.logaddexp(0.0, -logit))
