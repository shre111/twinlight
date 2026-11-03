import dataclasses
import json
import tomllib
from pathlib import Path

import blake3
import numpy as np
import pytest

from twl_engine.clock import Clock
from twl_engine.events import EventQueue, Events
from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.run import ENGINE_VERSION, RunInputs, run_seed, run_seed_bytes
from twl_engine.scenario import EMPTY, Params, Scenario
from twl_engine.state import new_resident_state
from twl_tape.format import NEVER, Tape, derive_run_id
from twl_tape.reader import read_tape
from twl_tape.writer import write_tape

GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "golden" / "hashes.toml"
RESIDENTS = 5_000
STORES = 14
TICKS_30D = 2_880
HEADER_END = 160 + 6 * 32
FOOTER_SIZE = 40


def pack_hash() -> bytes:
    table = tomllib.loads(GOLDEN.read_text())["toy_pack"]["seed49_residents5000"]
    return bytes.fromhex(table["hash"])


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=RESIDENTS)


def inputs(city: ToyCity, seed: int = 1, scenario: Scenario = EMPTY) -> RunInputs:
    return RunInputs(
        city=city, city_pack_hash=pack_hash(), scenario=scenario, params=Params(), seed=seed
    )


@pytest.fixture(scope="module")
def empty_bytes(city: ToyCity) -> bytes:
    return run_seed_bytes(inputs(city))


@pytest.fixture(scope="module")
def empty_tape(empty_bytes: bytes) -> Tape:
    return read_tape(empty_bytes)


def rows(popped: Events) -> list[tuple[int, ...]]:
    columns = (popped.minute, popped.seq, popped.kind, popped.a, popped.b)
    return [tuple(int(v) for v in r) for r in zip(*columns, strict=True)]


def test_clock_counts_ticks_and_tick_start_minutes() -> None:
    clock = Clock(43_200, 15)
    assert clock.ticks == 2_880
    assert clock.tick_start(0) == 0
    assert clock.tick_start(7) == 105
    assert clock.tick_start(clock.ticks - 1) == 43_200 - 15


def test_clock_rejects_sim_minutes_not_divisible_by_tick() -> None:
    with pytest.raises(ValueError):
        Clock(100, 15)


def test_new_resident_state_starts_unaware_with_documented_dtypes() -> None:
    s = new_resident_state(6)
    for name in ("aware_at", "first_order_at", "heard_from"):
        col = getattr(s, name)
        assert col.dtype == np.uint32 and col.shape == (6,), name
        assert (col == NEVER).all(), name
    assert s.order_count.dtype == np.uint16 and not s.order_count.any()
    assert s.first_source.dtype == np.uint8 and not s.first_source.any()
    assert s.outcome_flags.dtype == np.uint8 and not s.outcome_flags.any()
    assert s.awareness.dtype == np.float32 and not s.awareness.any()


def test_new_resident_state_columns_are_independent_arrays() -> None:
    s = new_resident_state(4)
    s.aware_at[0] = 5
    assert s.first_order_at[0] == NEVER and s.heard_from[0] == NEVER
    assert new_resident_state(4).aware_at[0] == NEVER


def test_event_queue_pops_in_minute_then_push_order() -> None:
    q = EventQueue()
    pushes = [(30, 1, 100, 0), (10, 2, 101, 0), (20, 3, 102, 0), (10, 4, 103, 0), (5, 5, 104, 0)]
    pushes.append((30, 6, 105, 0))
    for minute, kind, a, b in pushes:
        q.push(minute, kind, a, b)
    assert len(q) == 6
    got = rows(q.pop_until(20))
    assert [(m, k, a, b) for m, _, k, a, b in got] == [
        (5, 5, 104, 0),
        (10, 2, 101, 0),
        (10, 4, 103, 0),
        (20, 3, 102, 0),
    ]
    assert [(m, s) for m, s, *_ in got] == sorted((m, s) for m, s, *_ in got)
    assert len(q) == 2


def test_event_queue_same_minute_events_pop_in_push_order_with_increasing_seq() -> None:
    q = EventQueue()
    for a in range(7):
        q.push(40, 9, a, a * 2)
    got = rows(q.pop_until(40))
    assert [r[3] for r in got] == list(range(7))
    seqs = [r[1] for r in got]
    assert seqs == sorted(seqs) and len(set(seqs)) == 7


def test_event_queue_pop_until_returns_nothing_past_the_bound() -> None:
    q = EventQueue()
    q.push(30, 1, 1, 1)
    q.push(31, 1, 2, 2)
    assert rows(q.pop_until(29)) == []
    assert len(q) == 2
    assert [r[0] for r in rows(q.pop_until(30))] == [30]
    assert [r[0] for r in rows(q.pop_until(30))] == []
    assert len(q) == 1
    assert [r[0] for r in rows(q.pop_until(10_000))] == [31]
    assert len(q) == 0


def test_event_queue_on_empty_pops_nothing() -> None:
    assert rows(EventQueue().pop_until(100)) == []


def test_event_queue_grows_past_its_initial_capacity_and_keeps_order() -> None:
    q = EventQueue()
    minutes = np.random.default_rng(3).integers(0, 600, size=10_000)
    for i, m in enumerate(minutes):
        q.push(int(m), i % 7, i, 0)
    assert len(q) == 10_000
    got = rows(q.pop_until(600))
    assert len(got) == 10_000 and len(q) == 0
    assert [r[0] for r in got] == sorted(int(m) for m in minutes)
    assert sorted(r[3] for r in got) == list(range(10_000))
    expected_payload_order = sorted(range(10_000), key=lambda i: (int(minutes[i]), i))
    assert [r[3] for r in got] == expected_payload_order


def test_event_queue_pops_in_two_windows_without_losing_or_repeating_events() -> None:
    q = EventQueue()
    for i in range(50):
        q.push((i * 37) % 100, 0, i, 0)
    first = rows(q.pop_until(49))
    second = rows(q.pop_until(99))
    assert len(first) + len(second) == 50
    assert all(r[0] <= 49 for r in first) and all(50 <= r[0] <= 99 for r in second)
    assert sorted(r[3] for r in first + second) == list(range(50))


def test_scenario_canonical_json_is_sorted_and_compact() -> None:
    raw = EMPTY.canonical_json()
    assert isinstance(raw, bytes)
    parsed = json.loads(raw)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() == raw
    assert parsed["days"] == 30 and parsed["tick_minutes"] == 15


def test_scenario_hash_is_blake3_of_the_canonical_json() -> None:
    assert EMPTY.hash() == blake3.blake3(EMPTY.canonical_json()).digest()
    assert len(EMPTY.hash()) == 32


def test_scenario_hash_changes_with_days_and_is_stable_for_equal_scenarios() -> None:
    assert Scenario(days=10).hash() != EMPTY.hash()
    assert Scenario().hash() == EMPTY.hash()
    assert Scenario(days=10).hash() == Scenario(days=10).hash()


def test_scenario_and_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        EMPTY.days = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Params().__setattr__("anything", 1)


def test_empty_scenario_is_thirty_days_of_fifteen_minute_ticks() -> None:
    assert EMPTY.days == 30
    assert EMPTY.tick_minutes == 15
    assert Scenario() == EMPTY


def test_params_hash_is_32_bytes_and_stable() -> None:
    assert len(Params().hash()) == 32
    assert Params().hash() == Params().hash()


def test_empty_run_bytes_round_trip_through_the_reader(empty_bytes: bytes) -> None:
    tape = read_tape(empty_bytes)
    assert tape.header.flags & 1 == 1


def test_empty_run_header_pins_inputs_and_clock(empty_tape: Tape) -> None:
    h = empty_tape.header
    assert h.sim_minutes == 43_200
    assert h.tick_minutes == 15
    assert h.seed == 1
    assert h.engine_version == ENGINE_VERSION
    assert h.scenario_hash == EMPTY.hash()
    assert h.city_pack_hash == pack_hash()
    assert h.run_id == derive_run_id(pack_hash(), EMPTY.hash(), Params().hash(), 1)
    assert h.flags & 1 == 1
    assert (h.residents, h.stores, h.chatter_rows, h.delivery_rows, h.stockout_pairs) == (
        RESIDENTS,
        STORES,
        0,
        0,
        0,
    )


def test_empty_run_has_every_resident_unaware_and_unordered(empty_tape: Tape) -> None:
    r = empty_tape.residents
    for name in ("aware_at", "first_order_at", "heard_from"):
        col = getattr(r, name)
        assert len(col) == RESIDENTS and (col == NEVER).all(), name
    for name in ("order_count", "first_source", "outcome_flags"):
        col = getattr(r, name)
        assert len(col) == RESIDENTS and not col.any(), name


def test_empty_run_has_no_chatter_and_no_deliveries(empty_tape: Tape) -> None:
    assert len(empty_tape.chatter.from_) == len(empty_tape.chatter.to) == 0
    assert len(empty_tape.chatter.at) == 0
    d = empty_tape.deliveries
    for name in ("store", "resident", "ordered_at", "picked_at", "delivered_at", "outcome"):
        assert len(getattr(d, name)) == 0, name


def test_empty_run_stores_section_is_sized_stores_by_ticks_and_all_zero(
    city: ToyCity, empty_tape: Tape
) -> None:
    assert len(city.stores.x) == STORES
    s = empty_tape.stores
    assert len(s.stock) == STORES * TICKS_30D and not s.stock.any()
    assert len(s.orders) == STORES * TICKS_30D and not s.orders.any()
    assert len(s.stockout_offsets) == STORES + 1 and not s.stockout_offsets.any()
    assert len(s.stockout_intervals) == 0


def test_empty_run_aggregates_are_2880_zero_ticks(empty_tape: Tape) -> None:
    a = empty_tape.aggregates
    for name in ("aware", "adopted", "orders", "stores_out", "rain_intensity"):
        col = getattr(a, name)
        assert len(col) == TICKS_30D and not col.any(), name


def test_same_inputs_twice_give_identical_bytes(city: ToyCity, empty_bytes: bytes) -> None:
    assert run_seed_bytes(inputs(city)) == empty_bytes


def test_run_seed_tape_matches_run_seed_bytes(city: ToyCity, empty_bytes: bytes) -> None:
    assert write_tape(run_seed(inputs(city))) == empty_bytes


def test_different_seed_changes_only_the_header_of_an_empty_run(
    city: ToyCity, empty_bytes: bytes
) -> None:
    other_bytes = run_seed_bytes(inputs(city, seed=2))
    other = read_tape(other_bytes)
    first = read_tape(empty_bytes)
    assert other.header.seed == 2
    assert other.header.run_id != first.header.run_id
    assert other.header.run_id == derive_run_id(pack_hash(), EMPTY.hash(), Params().hash(), 2)
    assert len(other_bytes) == len(empty_bytes)
    assert other_bytes[HEADER_END:-FOOTER_SIZE] == empty_bytes[HEADER_END:-FOOTER_SIZE]
    assert other_bytes[-32:] != empty_bytes[-32:]


def test_ten_day_scenario_gives_960_ticks(city: ToyCity) -> None:
    tape = run_seed(inputs(city, scenario=Scenario(days=10)))
    assert tape.header.sim_minutes == 14_400
    assert tape.header.scenario_hash == Scenario(days=10).hash()
    assert len(tape.aggregates.aware) == 960
    assert len(tape.stores.stock) == STORES * 960
    assert len(tape.residents.aware_at) == RESIDENTS
    assert (tape.header.residents, tape.header.stores) == (RESIDENTS, STORES)
