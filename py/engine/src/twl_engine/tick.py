"""Per-tick behaviour hooks over the resident struct of arrays.

Each hook is a plain function `(state, streams, queue, start, end, events)`: `events` are the ones
due in `[start, end)`, already drained, and hooks schedule later ones on `queue`. P-012 adds
awareness and purchase updates here, P-013 the supply coupling. Nothing runs yet.
"""

from collections.abc import Callable

import numpy as np

from twl_engine.events import EventQueue, Events
from twl_engine.state import ResidentState

Streams = dict[str, np.random.Generator]
TickHook = Callable[[ResidentState, Streams, EventQueue, int, int, Events], None]

HOOKS: tuple[TickHook, ...] = ()


def advance_tick(
    state: ResidentState,
    streams: Streams,
    queue: EventQueue,
    start: int,
    end: int,
    events: Events,
) -> None:
    """Apply every hook for the tick window `[start, end)` after its events were drained."""
    for hook in HOOKS:
        hook(state, streams, queue, start, end, events)
