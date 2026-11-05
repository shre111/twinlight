"""Per-tick behaviour hooks, run in MODEL.md section 11 order.

Each hook is a plain function `(sim, streams, queue, start, end, events)`: `sim` holds the static
world and the mutable state, `events` are the ones due in `[start, end)`, already drained, and
hooks schedule later ones on `queue`. P-013 adds the supply coupling here.
"""

from collections.abc import Callable

import numpy as np

from twl_engine.awareness import decay_awareness, paid_exposure
from twl_engine.events import EventQueue, Events
from twl_engine.state import Sim
from twl_engine.wom import word_of_mouth

Streams = dict[str, np.random.Generator]
TickHook = Callable[[Sim, Streams, EventQueue, int, int, Events], None]

HOOKS: tuple[TickHook, ...] = (decay_awareness, paid_exposure, word_of_mouth)


def advance_tick(
    sim: Sim,
    streams: Streams,
    queue: EventQueue,
    start: int,
    end: int,
    events: Events,
) -> None:
    """Apply every hook for the tick window `[start, end)` after its events were drained."""
    for hook in HOOKS:
        hook(sim, streams, queue, start, end, events)
