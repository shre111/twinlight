import dataclasses
import os
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import pytest

from twl_engine.goldens import (
    golden_inputs,
    golden_run_pid,
    low_stock_inputs,
    prototype_small_inputs,
    supply_small_inputs,
)
from twl_engine.montecarlo import run_seeds
from twl_engine.run import RunInputs, run_seed_bytes, run_seed_with_stats
from twl_tape import write_tape
from twl_tape.format import SectionKind
from twl_tape.reader import read_tape

GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "golden" / "hashes.toml"
STRICT = os.environ.get("TWL_GOLDEN_STRICT") == "1"
PINNED_SEEDS = [1, 2]
WORKER_SEEDS = list(range(1, 9))
PROBE_KEYS = ["aware_day30", "adopted_day30", "orders_total", "tape_bytes", "chatter_rows"]
SUPPLY_PROBE_KEYS = ["deliveries", "late", "stockout_pairs", "lost"]
PINNED: dict[str, Callable[[int], RunInputs]] = {
    "empty": golden_inputs,
    "prototype_small": prototype_small_inputs,
    "supply_small": supply_small_inputs,
    "low_stock": low_stock_inputs,
}
WORKER_INPUTS: dict[str, Callable[[int], RunInputs]] = {
    "prototype_small": prototype_small_inputs,
    "supply_small": supply_small_inputs,
}

pytestmark = pytest.mark.golden


@lru_cache(maxsize=1)
def golden() -> dict[str, Any]:
    return tomllib.loads(GOLDEN.read_text())


def digest(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def probes(data: bytes) -> dict[str, int]:
    tape = read_tape(data)
    return {
        "aware_day30": int(tape.aggregates.aware[-1]) if len(tape.aggregates.aware) else 0,
        "adopted_day30": int(tape.aggregates.adopted[-1]) if len(tape.aggregates.adopted) else 0,
        "orders_total": int(tape.aggregates.orders.astype(np.int64).sum()),
        "tape_bytes": len(data),
        "chatter_rows": len(tape.chatter.at),
    }


def first_difference(a: bytes, b: bytes) -> str:
    if a == b:
        return "identical"
    ta, tb = read_tape(a), read_tape(b)
    sections = [
        (SectionKind.RESIDENTS, ta.residents, tb.residents),
        (SectionKind.CHATTER, ta.chatter, tb.chatter),
        (SectionKind.DELIVERIES, ta.deliveries, tb.deliveries),
        (SectionKind.STORES, ta.stores, tb.stores),
        (SectionKind.AGGREGATES, ta.aggregates, tb.aggregates),
    ]
    if ta.header != tb.header:
        return f"header differs: {ta.header} vs {tb.header}"
    for kind, sa, sb in sections:
        for field in dataclasses.fields(sa):
            name = field.name
            ca, cb = getattr(sa, name), getattr(sb, name)
            if ca.shape != cb.shape:
                return f"{kind.name}.{name} length {ca.shape} vs {cb.shape}"
            bad = np.flatnonzero(ca != cb)
            if len(bad):
                i = int(bad[0])
                return f"{kind.name}.{name} first differing row {i}: {ca[i]} vs {cb[i]}"
    offset = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
    return f"bytes differ at offset {offset}"


def golden_table(name: str, seed: int) -> dict[str, Any]:
    key = f"{name}_seed{seed}"
    table = golden().get("engine", {}).get(key)
    if table is None:
        pytest.fail(
            f"fixtures/golden/hashes.toml has no [engine.{key}] table; "
            "run `just goldens` for an intended output and check it in"
        )
    return table


@pytest.fixture(scope="module")
def in_process() -> dict[int, bytes]:
    return {seed: run_seed_bytes(prototype_small_inputs(seed)) for seed in WORKER_SEEDS}


@pytest.fixture(scope="module")
def in_process_supply() -> dict[int, bytes]:
    return {seed: run_seed_bytes(supply_small_inputs(seed)) for seed in WORKER_SEEDS}


@pytest.mark.parametrize("workers", [1, 4, 8])
@pytest.mark.parametrize("name", list(WORKER_INPUTS))
def test_hashes_are_identical_across_worker_counts_and_in_process_runs(
    name: str,
    workers: int,
    in_process: dict[int, bytes],
    in_process_supply: dict[int, bytes],
) -> None:
    reference = in_process if name == "prototype_small" else in_process_supply
    tapes = run_seeds(WORKER_INPUTS[name], WORKER_SEEDS, workers=workers)
    assert len(tapes) == len(WORKER_SEEDS)
    for seed, got in zip(WORKER_SEEDS, tapes, strict=True):
        want = reference[seed]
        assert digest(got) == digest(want), (
            f"{name} seed {seed} with {workers} workers: {first_difference(got, want)}"
        )


@pytest.mark.parametrize("workers", [4, 8])
def test_worker_pools_spread_seeds_over_that_many_processes(workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pids = list(pool.map(golden_run_pid, WORKER_SEEDS))
    assert len(set(pids)) == workers, f"{workers} workers used {len(set(pids))} processes"


def test_two_in_process_runs_of_a_seed_are_identical(in_process: dict[int, bytes]) -> None:
    again = run_seed_bytes(prototype_small_inputs(1))
    assert digest(again) == digest(in_process[1]), first_difference(again, in_process[1])


def test_prototype_small_run_consumes_the_streams(in_process: dict[int, bytes]) -> None:
    assert probes(in_process[1])["adopted_day30"] > 0
    assert digest(in_process[1]) != digest(in_process[2])


@pytest.mark.parametrize("name", list(PINNED))
@pytest.mark.parametrize("seed", PINNED_SEEDS)
def test_run_matches_the_checked_in_golden(
    name: str, seed: int, in_process: dict[int, bytes]
) -> None:
    expected = golden_table(name, seed)
    label = f"[engine.{name}_seed{seed}]"
    keys = PROBE_KEYS
    if name in ("supply_small", "low_stock"):
        tape, stats = run_seed_with_stats(PINNED[name](seed))
        data = write_tape(tape)
        actual = probes(data) | {
            "deliveries": tape.header.delivery_rows,
            "late": stats.late,
            "stockout_pairs": tape.header.stockout_pairs,
            "lost": stats.lost,
        }
        keys = PROBE_KEYS + SUPPLY_PROBE_KEYS
    else:
        data = in_process[seed] if name == "prototype_small" else run_seed_bytes(PINNED[name](seed))
        actual = probes(data)
    if STRICT:
        assert digest(data) == expected["hash"], (
            f"{label} hash is {digest(data)}, golden {expected['hash']}; probes {actual}"
        )
        return
    for key in keys:
        assert key in expected, f"{label} is missing probe {key}"
        assert actual[key] == expected[key], (
            f"{label} {key}: {actual[key]} vs golden {expected[key]}"
        )
