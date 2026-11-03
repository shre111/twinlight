"""Golden hashes and probes for the engine scenarios (`python -m twl_engine.goldens [--write]`).

Strict hashes are only meaningful on the pinned platform (D-023); `just goldens` sets the
environment, and CI confirms the values. Tables other than `engine` are kept as they are.
"""

import argparse
import json
import os
import tomllib
from collections.abc import Sequence
from pathlib import Path

from twl_engine.fixtures import build_toy_city
from twl_engine.run import RunInputs, run_seed_bytes
from twl_engine.scenario import EMPTY, Params, Scenario
from twl_tape import read_tape, tape_hash

GOLDEN_PATH = Path(__file__).resolve().parents[4] / "fixtures" / "golden" / "hashes.toml"
TOY_PACK_KEY = "seed49_residents5000"
GOLDEN_RESIDENTS = 5_000
GOLDEN_RUNS: tuple[tuple[str, Scenario, int], ...] = (
    ("empty_seed1", EMPTY, 1),
    ("empty_seed2", EMPTY, 2),
)
HEADER = (
    "# Strict hashes are checked on the pinned CI platform (TWL_GOLDEN_STRICT=1); "
    "elsewhere the probes are compared with tolerance.\n"
    "# Change these only for an intended output change, with the reason in the PR body."
)

Probes = dict[str, str | int | float]


def read_goldens(path: Path = GOLDEN_PATH) -> dict[str, dict[str, Probes]]:
    return tomllib.loads(path.read_text())


def toy_pack_hash(path: Path = GOLDEN_PATH) -> bytes:
    """The toy pack's golden `pack_hash`, which the golden tapes carry as `city_pack_hash`."""
    return bytes.fromhex(str(read_goldens(path)["toy_pack"][TOY_PACK_KEY]["hash"]))


def golden_inputs(seed: int, scenario: Scenario = EMPTY) -> RunInputs:
    return RunInputs(
        city=build_toy_city(seed=49, residents=GOLDEN_RESIDENTS),
        city_pack_hash=toy_pack_hash(),
        scenario=scenario,
        params=Params(),
        seed=seed,
    )


def tape_probes(data: bytes) -> Probes:
    tape = read_tape(data)
    aggregates = tape.aggregates
    return {
        "hash": tape_hash(data),
        "aware_day30": int(aggregates.aware[-1]),
        "adopted_day30": int(aggregates.adopted[-1]),
        "orders_total": int(aggregates.orders.sum()),
        "tape_bytes": len(data),
    }


def compute_engine_goldens() -> dict[str, Probes]:
    return {
        name: tape_probes(run_seed_bytes(golden_inputs(seed, scenario)))
        for name, scenario, seed in GOLDEN_RUNS
    }


def render_toml(tables: dict[str, dict[str, Probes]]) -> str:
    """Two-level tables of scalars only; anything else raises rather than being rewritten."""
    lines = [HEADER]
    for table, entries in tables.items():
        if not isinstance(entries, dict):
            raise TypeError(f"[{table}] must be a table of tables")
        for name, values in entries.items():
            if not isinstance(values, dict):
                raise TypeError(f"[{table}.{name}] must be a table")
            lines.append(f"\n[{table}.{name}]")
            lines.extend(
                f"{key} = {_toml_value(f'{table}.{name}.{key}', value)}"
                for key, value in values.items()
            )
    return "\n".join(lines) + "\n"


def _toml_value(path: str, value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if type(value) in (int, float):
        return repr(value)
    raise TypeError(f"{path}: cannot round-trip a {type(value).__name__} through render_toml")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m twl_engine.goldens")
    parser.add_argument("--write", action="store_true", help=f"rewrite {GOLDEN_PATH}")
    args = parser.parse_args(argv)
    if args.write and os.environ.get("TWL_GOLDEN_STRICT") != "1":
        raise SystemExit(
            "refusing to write goldens without TWL_GOLDEN_STRICT=1; "
            "use `just goldens`, which sets the pinned environment (D-023)"
        )
    tables = read_goldens()
    tables["engine"] = compute_engine_goldens()
    text = render_toml(tables)
    if args.write:
        GOLDEN_PATH.write_text(text)
        print(f"wrote {GOLDEN_PATH}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
