from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest

from twl_minds.elicitation.guard import Budget, CostCapExceeded, CostGuard
from twl_minds.elicitation.keys import CacheKey
from twl_minds.elicitation.pricing import Usage
from twl_minds.elicitation.store import ElicitationStore

NOW = datetime(2026, 11, 11, 15, 0, 0, tzinfo=UTC)
MIDNIGHT = datetime(2026, 11, 11, 0, 0, 0, tzinfo=UTC)
KEY = CacheKey("p:aaaa", "s1", "elicit-v1", "claude-haiku-5-5")
USAGE = Usage(
    input_tokens=1,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
    output_tokens=1,
)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _spend(store: ElicitationStore, clock: Clock, call_id: str, at: datetime, cost: float) -> None:
    clock.now = at
    store.record(call_id, KEY, "errored", USAGE, cost, False)


@pytest.fixture
def clock() -> Clock:
    return Clock(NOW)


@pytest.fixture
def store(clock: Clock) -> Iterator[ElicitationStore]:
    with ElicitationStore(":memory:", clock=clock) as opened:
        yield opened


def _guard(
    store: ElicitationStore, clock: Clock, run: float = 1.0, daily: float = 5.0
) -> CostGuard:
    return CostGuard(run_cap_usd=run, daily_cap_usd=daily, store=store, clock=clock)


def test_an_estimate_under_both_caps_returns_the_budget(
    store: ElicitationStore, clock: Clock
) -> None:
    _spend(store, clock, "c1", MIDNIGHT + timedelta(hours=1), 2.0)
    clock.now = NOW
    budget = _guard(store, clock).check(0.5)
    assert budget == Budget(
        estimate_usd=0.5, spent_today_usd=2.0, run_cap_usd=1.0, daily_cap_usd=5.0
    )


def test_an_estimate_equal_to_the_run_cap_passes(store: ElicitationStore, clock: Clock) -> None:
    assert _guard(store, clock).check(1.0).estimate_usd == 1.0


def test_an_estimate_a_cent_over_the_run_cap_fails_as_run(
    store: ElicitationStore, clock: Clock
) -> None:
    with pytest.raises(CostCapExceeded) as raised:
        _guard(store, clock).check(1.01)
    assert raised.value.cap == "run"
    assert raised.value.budget.estimate_usd == 1.01
    assert isinstance(raised.value, ValueError)


def test_spend_plus_estimate_equal_to_the_daily_cap_passes(
    store: ElicitationStore, clock: Clock
) -> None:
    _spend(store, clock, "c1", MIDNIGHT + timedelta(hours=1), 4.5)
    clock.now = NOW
    assert _guard(store, clock).check(0.5).spent_today_usd == 4.5


def test_spend_plus_estimate_a_cent_over_the_daily_cap_fails_as_daily(
    store: ElicitationStore, clock: Clock
) -> None:
    _spend(store, clock, "c1", MIDNIGHT + timedelta(hours=1), 4.5)
    clock.now = NOW
    with pytest.raises(CostCapExceeded) as raised:
        _guard(store, clock).check(0.51)
    assert raised.value.cap == "daily"
    assert raised.value.budget == Budget(0.51, 4.5, 1.0, 5.0)


def test_run_cap_is_reported_before_the_daily_cap_when_both_trip(
    store: ElicitationStore, clock: Clock
) -> None:
    _spend(store, clock, "c1", MIDNIGHT + timedelta(hours=1), 4.9)
    clock.now = NOW
    with pytest.raises(CostCapExceeded) as raised:
        _guard(store, clock).check(1.5)
    assert raised.value.cap == "run"


def test_daily_window_starts_at_utc_midnight(store: ElicitationStore, clock: Clock) -> None:
    _spend(store, clock, "yesterday", MIDNIGHT - timedelta(seconds=1), 3.0)
    _spend(store, clock, "midnight", MIDNIGHT, 2.0)
    _spend(store, clock, "morning", MIDNIGHT + timedelta(hours=9), 2.0)
    clock.now = NOW
    guard = _guard(store, clock)
    assert guard.check(0.5).spent_today_usd == pytest.approx(4.0)
    with pytest.raises(CostCapExceeded) as raised:
        guard.check(1.0 + 0.01)
    assert raised.value.cap == "run"


def test_spend_before_midnight_does_not_count_toward_today(
    store: ElicitationStore, clock: Clock
) -> None:
    _spend(store, clock, "yesterday", MIDNIGHT - timedelta(seconds=1), 4.9)
    clock.now = NOW
    assert _guard(store, clock).check(1.0).spent_today_usd == 0.0


def test_the_window_rolls_over_with_the_guard_clock(store: ElicitationStore, clock: Clock) -> None:
    _spend(store, clock, "c1", MIDNIGHT + timedelta(hours=1), 4.5)
    guard = _guard(store, clock)
    clock.now = MIDNIGHT + timedelta(hours=23, minutes=59, seconds=59)
    with pytest.raises(CostCapExceeded):
        guard.check(0.51)
    clock.now = MIDNIGHT + timedelta(days=1)
    assert guard.check(0.51).spent_today_usd == 0.0


def test_the_day_is_a_utc_day_even_when_the_guard_clock_is_offset(
    store: ElicitationStore, clock: Clock
) -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    _spend(store, clock, "c1", MIDNIGHT - timedelta(minutes=30), 4.5)
    guard_clock = Clock(datetime(2026, 11, 11, 5, 45, tzinfo=ist))
    guard = CostGuard(run_cap_usd=1.0, daily_cap_usd=5.0, store=store, clock=guard_clock)
    assert guard.check(0.51).spent_today_usd == 0.0


def test_check_records_nothing(store: ElicitationStore, clock: Clock) -> None:
    guard = _guard(store, clock)
    guard.check(0.5)
    with pytest.raises(CostCapExceeded):
        guard.check(2.0)
    assert store.spent_since(MIDNIGHT - timedelta(days=1)) == 0.0
    assert store.status_counts("s1") == {}


def test_a_zero_estimate_passes(store: ElicitationStore, clock: Clock) -> None:
    assert _guard(store, clock).check(0.0).estimate_usd == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_run_cap_must_be_finite_and_positive(
    store: ElicitationStore, clock: Clock, bad: float
) -> None:
    with pytest.raises(ValueError):
        CostGuard(run_cap_usd=bad, daily_cap_usd=5.0, store=store, clock=clock)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_daily_cap_must_be_finite_and_positive(
    store: ElicitationStore, clock: Clock, bad: float
) -> None:
    with pytest.raises(ValueError):
        CostGuard(run_cap_usd=1.0, daily_cap_usd=bad, store=store, clock=clock)


def test_run_cap_above_daily_cap_is_refused(store: ElicitationStore, clock: Clock) -> None:
    with pytest.raises(ValueError):
        CostGuard(run_cap_usd=5.01, daily_cap_usd=5.0, store=store, clock=clock)


def test_run_cap_equal_to_daily_cap_is_allowed(store: ElicitationStore, clock: Clock) -> None:
    CostGuard(run_cap_usd=5.0, daily_cap_usd=5.0, store=store, clock=clock)
