import heapq

import numpy as np
import pytest

from twl_engine.events import EventQueue


def test_pop_until_returns_events_in_minute_then_seq_order() -> None:
    queue = EventQueue(capacity=2)
    rng = np.random.Generator(np.random.PCG64(3))
    minutes = rng.integers(0, 60, size=3000).tolist()
    reference: list[tuple[int, int]] = []
    for seq, minute in enumerate(minutes):
        assert queue.push(minute, kind=1, a=seq, b=-seq) == seq
        heapq.heappush(reference, (minute, seq))
    assert len(queue) == 3000

    first = queue.pop_until(29)
    expected = [heapq.heappop(reference) for _ in range(sum(m <= 29 for m in minutes))]
    assert list(zip(first.minute.tolist(), first.seq.tolist(), strict=True)) == expected
    assert first.a.tolist() == first.seq.tolist()
    assert first.b.tolist() == (-first.seq).tolist()

    rest = queue.pop_until(10_000)
    assert len(first) + len(rest) == 3000
    assert len(queue) == 0
    assert rest.minute.min() >= 30
    assert np.all(np.diff(rest.minute) >= 0)


def test_events_pushed_later_at_the_same_minute_come_later() -> None:
    queue = EventQueue()
    for _ in range(50):
        queue.push(15, kind=0)
    popped = queue.pop_until(15)
    assert popped.seq.tolist() == list(range(50))


def test_pop_until_leaves_future_events_and_peek_reports_the_next_minute() -> None:
    queue = EventQueue()
    queue.push(40, kind=0)
    queue.push(10, kind=0)
    assert queue.peek_minute() == 10
    assert len(queue.pop_until(9)) == 0
    assert queue.pop_until(10).minute.tolist() == [10]
    assert queue.peek_minute() == 40


def test_negative_minutes_are_rejected() -> None:
    with pytest.raises(ValueError):
        EventQueue().push(-1, kind=0)
