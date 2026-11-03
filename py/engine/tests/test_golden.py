import dataclasses
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import pytest

from twl_engine.goldens import golden_inputs, golden_run_pid
from twl_engine.montecarlo import run_seeds
from twl_engine.run import run_seed_bytes
from twl_tape.format import SectionKind
from twl_tape.reader import read_tape

GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "golden" / "hashes.toml"
STRICT = os.environ.get("TWL_GOLDEN_STRICT") == "1"
PINNED_SEEDS = [1, 2]
WORKER_SEEDS = list(range(1, 9))
PROBE_KEYS = ["aware_day30", "adopted_day30", "orders_total", "tape_bytes"]

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


def golden_table(seed: int) -> dict[str, Any]:
    key = f"empty_seed{seed}"
    table = golden().get("engine", {}).get(key)
    if table is None:
        pytest.fail(
            f"fixtures/golden/hashes.toml has no [engine.{key}] table; "
            "run `just goldens` for an intended output and check it in"
        )
    return table


@pytest.fixture(scope="module")
def in_process() -> dict[int, bytes]:
    return {seed: run_seed_bytes(golden_inputs(seed)) for seed in WORKER_SEEDS}


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_hashes_are_identical_across_worker_counts_and_in_process_runs(
    workers: int, in_process: dict[int, bytes]
) -> None:
    tapes = run_seeds(golden_inputs, WORKER_SEEDS, workers=workers)
    assert len(tapes) == len(WORKER_SEEDS)
    for seed, got in zip(WORKER_SEEDS, tapes, strict=True):
        want = in_process[seed]
        assert digest(got) == digest(want), (
            f"seed {seed} with {workers} workers: {first_difference(got, want)}"
        )


@pytest.mark.parametrize("workers", [4, 8])
def test_worker_pools_spread_seeds_over_that_many_processes(workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pids = list(pool.map(golden_run_pid, WORKER_SEEDS))
    assert len(set(pids)) == workers, f"{workers} workers used {len(set(pids))} processes"


def test_two_in_process_runs_of_a_seed_are_identical(in_process: dict[int, bytes]) -> None:
    again = run_seed_bytes(golden_inputs(1))
    assert digest(again) == digest(in_process[1]), first_difference(again, in_process[1])


@pytest.mark.parametrize("seed", PINNED_SEEDS)
def test_empty_run_matches_the_checked_in_golden(seed: int, in_process: dict[int, bytes]) -> None:
    expected = golden_table(seed)
    data = in_process[seed]
    actual = probes(data)
    if STRICT:
        assert digest(data) == expected["hash"], (
            f"empty seed {seed} hash is {digest(data)}, golden {expected['hash']}; probes {actual}"
        )
        return
    for key in PROBE_KEYS:
        assert key in expected, f"[engine.empty_seed{seed}] is missing probe {key}"
        assert actual[key] == expected[key], f"{key}: {actual[key]} vs golden {expected[key]}"
