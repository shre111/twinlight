"""Word of mouth (MODEL.md sections 3 and 11) and the heard rows of the tape (D-028).

Per-tie talk counts are independent Poisson draws, so the tick's talks are sampled as one Poisson
count from the summed rate, each then given a talker and a tie in proportion to their rates.
"""

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from twl_engine.events import EventQueue, Events
from twl_engine.state import Sim
from twl_tape import CHATTER_HEARD, NEVER, SOURCE_WOM

Streams = dict[str, np.random.Generator]

TALKER, LISTENER, MINUTE = 0, 1, 2


def word_of_mouth(
    sim: Sim, streams: Streams, queue: EventQueue, start: int, end: int, events: Events
) -> None:
    """Talks from residents aware before the tick started, applied in (minute, talker) order."""
    talk, residents, log = sim.world.talk, sim.residents, sim.chatter
    if talk.talks_per_pull <= 0.0:
        return
    total = talker_rates(
        residents.aware_at,
        residents.first_order_at,
        residents.last_order_at,
        talk.pull,
        start,
        talk.hearsay,
        talk.floor,
        talk.decay_per_minute,
        residents.talk_rate_cum,
    )
    if total <= 0.0:
        return
    rng = streams["wom"]
    count = int(rng.poisson(total * talk.talks_per_pull))
    if count == 0:
        return
    log.rows = apply_talks(
        rng.random((count, 3)),
        residents.talk_rate_cum,
        start,
        end - start,
        talk.uniform,
        talk.offsets,
        talk.targets,
        talk.tie_weight,
        talk.tie_gain,
        talk.pull,
        talk.uniform_gain,
        sim.world.params.theta_aware,
        residents.awareness,
        residents.aware_at,
        residents.first_source,
        residents.heard_from,
        residents.last_talk_from,
        residents.last_talk_at,
        log.from_,
        log.to,
        log.at,
        log.kind,
        log.rows,
    )


@njit(cache=True)
def talker_rates(
    aware_at: NDArray[np.uint32],
    first_order_at: NDArray[np.uint32],
    last_order_at: NDArray[np.uint32],
    pull: NDArray[np.float64],
    start: int,
    hearsay: float,
    floor: float,
    decay_per_minute: float,
    cum: NDArray[np.float64],
) -> float:
    """Fill `cum` with the running sum of lambda_u / q at tick start and return the total.

    lambda_u = q * pull_u * (floor + (1 - floor) * exp(-(t - t_o) / decay)) for adopters, with
    t_o the latest order; aware non-adopters talk at the hearsay factor, fading to zero from
    `aware_at` (MODEL.md section 3, D-029).
    """
    total = 0.0
    for u in range(aware_at.shape[0]):
        if aware_at[u] < start and pull[u] > 0.0:
            adopter = first_order_at[u] != NEVER
            if adopter:
                rate = pull[u]
                since = np.int64(last_order_at[u])
            else:
                rate = hearsay * pull[u]
                since = np.int64(aware_at[u])
            if decay_per_minute > 0.0:
                fade = math.exp(-(start - since) * decay_per_minute)
                rate *= floor + (1.0 - floor) * fade if adopter else fade
            total += rate
        cum[u] = total
    return total


@njit(cache=True)
def apply_talks(
    draws: NDArray[np.float64],
    cum: NDArray[np.float64],
    start: int,
    tick_minutes: int,
    uniform: bool,
    offsets: NDArray[np.int64],
    targets: NDArray[np.int64],
    tie_weight: NDArray[np.float64],
    tie_gain: NDArray[np.float64],
    pull: NDArray[np.float64],
    uniform_gain: float,
    theta_aware: float,
    awareness: NDArray[np.float64],
    aware_at: NDArray[np.uint32],
    first_source: NDArray[np.uint8],
    heard_from: NDArray[np.uint32],
    last_talk_from: NDArray[np.uint32],
    last_talk_at: NDArray[np.uint32],
    log_from: NDArray[np.uint32],
    log_to: NDArray[np.uint32],
    log_at: NDArray[np.uint32],
    log_kind: NDArray[np.uint8],
    rows: int,
) -> int:
    """Give each talk a talker, a minute and a listener, then apply them in (minute, talker) order.

    The first talk to take a listener over `theta_aware` sets `aware_at`, `first_source` and
    `heard_from` and is logged as a heard row. Returns the new row count of the log.
    """
    n = cum.shape[0]
    count = draws.shape[0]
    total = cum[n - 1]
    talker = np.empty(count, dtype=np.int64)
    minute = np.empty(count, dtype=np.int64)
    for i in range(count):
        u = min(np.searchsorted(cum, draws[i, TALKER] * total, side="right"), n - 1)
        while u > 0 and cum[u] == cum[u - 1]:
            u -= 1
        talker[i] = u
        minute[i] = start + min(int(draws[i, MINUTE] * tick_minutes), tick_minutes - 1)
    order = np.argsort(minute * n + talker, kind="mergesort")
    for i in order:
        u = talker[i]
        if uniform:
            v = min(int(draws[i, LISTENER] * (n - 1)), n - 2)
            if v >= u:
                v += 1
            gain = uniform_gain
        else:
            point = draws[i, LISTENER] * pull[u]
            tie = pick_tie(int(offsets[u]), int(offsets[u + 1]), tie_weight, point)
            v = targets[tie]
            gain = tie_gain[tie]
        awareness[v] += gain
        last_talk_from[v] = u
        last_talk_at[v] = minute[i]
        if aware_at[v] == NEVER and awareness[v] >= theta_aware:
            aware_at[v] = minute[i]
            first_source[v] = SOURCE_WOM
            heard_from[v] = u
            log_from[rows] = u
            log_to[rows] = v
            log_at[rows] = minute[i]
            log_kind[rows] = CHATTER_HEARD
            rows += 1
    return rows


@njit(cache=True)
def pick_tie(first: int, end: int, tie_weight: NDArray[np.float64], point: float) -> int:
    """The tie whose share of the talker's pull contains `point`; never a zero-weight tie."""
    chosen = first
    reached = 0.0
    for tie in range(first, end):
        if tie_weight[tie] > 0.0:
            chosen = tie
            reached += tie_weight[tie]
            if reached > point:
                break
    return chosen
