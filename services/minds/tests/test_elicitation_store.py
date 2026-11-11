from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from twl_minds.elicitation.keys import CacheKey
from twl_minds.elicitation.pricing import Usage
from twl_minds.elicitation.store import ElicitationStore

T0 = datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC)
KEY = CacheKey("p1:aaaa", "s1", "elicit-v1", "claude-haiku-5-5")
OTHER = CacheKey("p2:bbbb", "s1", "elicit-v1", "claude-haiku-5-5")
USAGE = Usage(
    input_tokens=100,
    cache_creation_input_tokens=200,
    cache_read_input_tokens=300,
    output_tokens=50,
)
ANSWER = '{"purchase_intent":4}'


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _valid(
    store: ElicitationStore,
    call_id: str,
    key: CacheKey = KEY,
    answer: str = ANSWER,
    cost: float = 0.25,
) -> None:
    store.record(call_id, key, "valid", USAGE, cost, True, answer_json=answer)


def _fail(store: ElicitationStore, call_id: str, status: str, cost: float = 0.1) -> None:
    store.record(call_id, KEY, status, USAGE, cost, True, reason="why")


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> Iterator[ElicitationStore]:
    with ElicitationStore(":memory:", clock=clock) as opened:
        yield opened


def test_cached_returns_nothing_for_an_empty_store(store: ElicitationStore) -> None:
    assert store.cached([KEY]) == {}
    assert store.cached([]) == {}


def test_cached_round_trips_a_valid_answer(store: ElicitationStore) -> None:
    _valid(store, "c1")
    assert store.cached([KEY]) == {KEY: ANSWER}


def test_cached_returns_only_the_keys_that_have_answers(store: ElicitationStore) -> None:
    _valid(store, "c1")
    assert store.cached([KEY, OTHER]) == {KEY: ANSWER}


def test_cached_distinguishes_every_key_column(store: ElicitationStore) -> None:
    _valid(store, "c1")
    variants = [
        CacheKey("x", KEY.scenario_hash, KEY.prompt_version, KEY.model),
        CacheKey(KEY.persona_id, "x", KEY.prompt_version, KEY.model),
        CacheKey(KEY.persona_id, KEY.scenario_hash, "elicit-v2", KEY.model),
        CacheKey(KEY.persona_id, KEY.scenario_hash, KEY.prompt_version, "other-model"),
    ]
    assert store.cached(variants) == {}


def test_cached_accepts_a_generator_and_many_keys(store: ElicitationStore) -> None:
    keys = [CacheKey(f"p{i}:x", "s", "v", "m") for i in range(300)]
    for i, key in enumerate(keys):
        _valid(store, f"c{i}", key, answer=f'{{"i":{i}}}')
    found = store.cached(k for k in keys)
    assert len(found) == 300
    assert found[keys[7]] == '{"i":7}'


def test_first_valid_answer_wins(store: ElicitationStore) -> None:
    _valid(store, "c1", answer='{"first":1}')
    _valid(store, "c2", answer='{"second":2}')
    assert store.cached([KEY]) == {KEY: '{"first":1}'}


def test_second_valid_for_a_key_is_still_logged_as_a_call(store: ElicitationStore) -> None:
    _valid(store, "c1")
    _valid(store, "c2", answer='{"second":2}')
    assert store.status_counts("s1") == {"valid": 2}


@pytest.mark.parametrize(
    "status", ["invalid", "refused", "truncated", "errored", "expired", "canceled"]
)
def test_non_valid_statuses_are_recorded_but_not_cached(
    store: ElicitationStore, status: str
) -> None:
    _fail(store, "c1", status)
    assert store.cached([KEY]) == {}
    assert store.status_counts("s1") == {status: 1}
    assert store.spent_since(T0) == pytest.approx(0.1)


def test_a_failed_call_does_not_block_a_later_valid_answer(store: ElicitationStore) -> None:
    _fail(store, "c1", "invalid")
    _valid(store, "c2")
    assert store.cached([KEY]) == {KEY: ANSWER}


def test_unknown_status_is_refused_and_nothing_is_recorded(store: ElicitationStore) -> None:
    with pytest.raises(ValueError):
        store.record("c1", KEY, "maybe", USAGE, 0.1, False)
    assert store.spent_since(T0 - timedelta(days=1)) == 0.0
    assert store.status_counts("s1") == {}


def test_duplicate_call_id_is_refused_and_leaves_state_unchanged(
    store: ElicitationStore,
) -> None:
    _valid(store, "c1", cost=0.25)
    with pytest.raises(ValueError):
        _valid(store, "c1", OTHER, cost=0.5)
    assert store.cached([OTHER]) == {}
    assert store.spent_since(T0) == pytest.approx(0.25)
    assert store.status_counts("s1") == {"valid": 1}


def test_valid_status_without_answer_json_is_refused(store: ElicitationStore) -> None:
    with pytest.raises(ValueError):
        store.record("c1", KEY, "valid", USAGE, 0.1, False)
    assert store.status_counts("s1") == {}


@pytest.mark.parametrize("status", ["invalid", "refused", "truncated", "errored"])
def test_non_valid_status_with_answer_json_is_refused(store: ElicitationStore, status: str) -> None:
    with pytest.raises(ValueError):
        store.record("c1", KEY, status, USAGE, 0.1, False, answer_json=ANSWER)
    assert store.cached([KEY]) == {}
    assert store.status_counts("s1") == {}


def test_spent_since_sums_calls_of_every_status(store: ElicitationStore) -> None:
    _valid(store, "c1", cost=0.25)
    _fail(store, "c2", "refused", cost=0.5)
    _fail(store, "c3", "errored", cost=0.0)
    assert store.spent_since(T0) == pytest.approx(0.75)


def test_spent_since_is_empty_zero(store: ElicitationStore) -> None:
    assert store.spent_since(T0) == 0.0


def test_spent_since_includes_a_call_at_exactly_since_and_excludes_earlier(
    store: ElicitationStore, clock: Clock
) -> None:
    clock.now = T0 - timedelta(microseconds=1)
    _fail(store, "before", "errored", cost=1.0)
    clock.now = T0
    _fail(store, "at", "errored", cost=2.0)
    clock.now = T0 + timedelta(seconds=1)
    _fail(store, "after", "errored", cost=4.0)
    assert store.spent_since(T0) == pytest.approx(6.0)
    assert store.spent_since(T0 + timedelta(microseconds=1)) == pytest.approx(4.0)
    assert store.spent_since(T0 + timedelta(seconds=2)) == 0.0
    assert store.spent_since(T0 - timedelta(days=1)) == pytest.approx(7.0)


def test_spent_since_does_not_depend_on_the_batch_flag(store: ElicitationStore) -> None:
    store.record("a", KEY, "errored", USAGE, 0.3, True)
    store.record("b", KEY, "errored", USAGE, 0.2, False)
    assert store.spent_since(T0) == pytest.approx(0.5)


def test_status_counts_groups_by_status_for_one_scenario(store: ElicitationStore) -> None:
    other_scenario = CacheKey("p9:cccc", "s2", "elicit-v1", "claude-haiku-5-5")
    _valid(store, "c1")
    _valid(store, "c2", OTHER)
    _fail(store, "c3", "invalid")
    _fail(store, "c4", "invalid")
    _fail(store, "c5", "expired")
    _valid(store, "c6", other_scenario)
    assert store.status_counts("s1") == {"valid": 2, "invalid": 2, "expired": 1}
    assert store.status_counts("s2") == {"valid": 1}
    assert store.status_counts("missing") == {}


def test_a_naive_clock_is_refused(tmp_path: Path) -> None:
    with (
        pytest.raises(ValueError),
        ElicitationStore(
            tmp_path / "n.duckdb", clock=lambda: datetime(2026, 11, 11, 12, 0, 0)
        ) as naive,
    ):
        _fail(naive, "c1", "errored")
    with ElicitationStore(tmp_path / "n.duckdb") as reopened:
        assert reopened.status_counts("s1") == {}


def test_a_reopened_file_keeps_answers_and_calls(tmp_path: Path, clock: Clock) -> None:
    path = tmp_path / "e.duckdb"
    with ElicitationStore(path, clock=clock) as first:
        _valid(first, "c1", cost=0.25)
        _fail(first, "c2", "refused", cost=0.5)
    with ElicitationStore(str(path), clock=clock) as second:
        assert second.cached([KEY]) == {KEY: ANSWER}
        assert second.status_counts("s1") == {"valid": 1, "refused": 1}
        assert second.spent_since(T0) == pytest.approx(0.75)
        with pytest.raises(ValueError):
            _valid(second, "c1", OTHER)


def test_a_new_path_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "fresh.duckdb"
    with ElicitationStore(path):
        pass
    assert path.exists()


def test_close_is_explicit_and_stores_do_not_share_state(clock: Clock) -> None:
    a = ElicitationStore(":memory:", clock=clock)
    b = ElicitationStore(":memory:", clock=clock)
    _valid(a, "c1")
    assert b.cached([KEY]) == {}
    a.close()
    b.close()
