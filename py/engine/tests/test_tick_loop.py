import numpy as np
import pytest

from twl_engine import tick
from twl_engine.events import EventQueue, Events
from twl_engine.fixtures import build_toy_city
from twl_engine.run import STREAMS, RunInputs, run_seed
from twl_engine.scenario import Params, Scenario
from twl_engine.state import ResidentState

EVENT_MINUTE = 100
EVENT_KIND = 7


def test_tick_loop_covers_every_window_once_and_delivers_events_in_their_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows: list[tuple[int, int]] = []
    deliveries: list[tuple[int, int, int]] = []
    stream_names: list[tuple[str, ...]] = []

    def recorder(
        state: ResidentState,
        streams: dict[str, np.random.Generator],
        queue: EventQueue,
        start: int,
        end: int,
        events: Events,
    ) -> None:
        if start == 0:
            queue.push(EVENT_MINUTE, EVENT_KIND, 11, 22)
            stream_names.append(tuple(streams))
        windows.append((start, end))
        pairs = zip(events.minute, events.kind, strict=True)
        deliveries.extend((start, int(m), int(k)) for m, k in pairs)

    monkeypatch.setattr(tick, "HOOKS", (recorder,))
    scenario = Scenario(days=2)
    city = build_toy_city(seed=49, residents=200, stores=2)
    run_seed(RunInputs(city, bytes(32), scenario, Params(), seed=3))

    starts = [start for start, _ in windows]
    assert starts == list(range(0, scenario.sim_minutes, scenario.tick_minutes))
    assert all(end - start == scenario.tick_minutes for start, end in windows)
    assert deliveries == [(90, EVENT_MINUTE, EVENT_KIND)]
    assert stream_names == [STREAMS]
