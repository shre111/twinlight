import dataclasses
import os
import sys
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.montecarlo import run_seeds
from twl_engine.run import RunInputs, run_seed_bytes
from twl_engine.scenario import EMPTY, Params
from twl_tape.format import SectionKind
from twl_tape.reader import read_tape

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault("test_golden", sys.modules[__name__])

GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "golden" / "hashes.toml"
STRICT = os.environ.get("TWL_GOLDEN_STRICT") == "1"
SEEDS = [1, 2]
PROBE_KEYS = ["aware_day30", "adopted_day30", "orders_total", "tape_bytes"]

pytestmark = pytest.mark.golden


@lru_cache(maxsize=1)
def toy_city() -> ToyCity:
    return build_toy_city(seed=49, residents=5_000)


@lru_cache(maxsize=1)
def golden() -> dict[str, Any]:
    return tomllib.loads(GOLDEN.read_text())


def inputs_for(seed: int) -> RunInputs:
    pack_hash = bytes.fromhex(golden()["toy_pack"]["seed49_residents5000"]["hash"])
    return RunInputs(
        city=toy_city(), city_pack_hash=pack_hash, scenario=EMPTY, params=Params(), seed=seed
    )


inputs_for.__module__ = "test_golden"


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
def in_process() -> list[bytes]:
    return [run_seed_bytes(inputs_for(s)) for s in SEEDS]


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_hashes_are_identical_across_worker_counts_and_in_process_runs(
    workers: int, in_process: list[bytes]
) -> None:
    tapes = run_seeds(inputs_for, SEEDS, workers=workers)
    assert len(tapes) == len(SEEDS)
    for seed, got, want in zip(SEEDS, tapes, in_process, strict=True):
        assert digest(got) == digest(want), (
            f"seed {seed} with {workers} workers: {first_difference(got, want)}"
        )


def test_two_in_process_runs_of_a_seed_are_identical(in_process: list[bytes]) -> None:
    again = run_seed_bytes(inputs_for(1))
    assert digest(again) == digest(in_process[0]), first_difference(again, in_process[0])


@pytest.mark.parametrize("seed", SEEDS)
def test_empty_run_matches_the_checked_in_golden(seed: int, in_process: list[bytes]) -> None:
    expected = golden_table(seed)
    data = in_process[SEEDS.index(seed)]
    actual = probes(data)
    if STRICT:
        assert digest(data) == expected["hash"], (
            f"empty seed {seed} hash is {digest(data)}, golden {expected['hash']}; probes {actual}"
        )
        return
    for key in PROBE_KEYS:
        assert key in expected, f"[engine.empty_seed{seed}] is missing probe {key}"
        assert actual[key] == expected[key], f"{key}: {actual[key]} vs golden {expected[key]}"
