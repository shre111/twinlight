"""Golden hashes and probes for the engine scenarios (`python -m twl_engine.goldens [--write]`).

Strict hashes are only meaningful on the pinned platform (D-023); `just goldens` sets the
environment, and CI confirms the values. Tables other than `engine` are kept as they are.
"""

import argparse
import dataclasses
import json
import os
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

from twl_engine.fixtures import build_toy_city, pack_hash
from twl_engine.run import RunInputs, SupplyStats, run_seed_bytes, run_seed_with_stats
from twl_engine.scenario import EMPTY, Params, Scenario, Supply, load_scenario
from twl_tape import Tape, read_tape, tape_hash, write_tape

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures"
GOLDEN_PATH = FIXTURES / "golden" / "hashes.toml"
PROTOTYPE_SCENARIO_PATH = FIXTURES / "scenarios" / "prototype_launch.toml"
PROTOTYPE_RESIDENTS = 48_000
TOY_PACK_KEY = "seed49_residents5000"
GOLDEN_RESIDENTS = 5_000
LOW_STOCK = Supply(initial_stock=0.004, reorder_point=0.001, order_up_to=0.004, lead_minutes=2880)
SLOW_RIDERS = Supply(riders_per_store=1, rider_speed_kmh=3.0)
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


def golden_inputs(seed: int, scenario: Scenario = EMPTY, params: Params | None = None) -> RunInputs:
    return RunInputs(
        city=build_toy_city(seed=49, residents=GOLDEN_RESIDENTS),
        city_pack_hash=toy_pack_hash(),
        scenario=scenario,
        params=Params() if params is None else params,
        seed=seed,
    )


def prototype_inputs(seed: int) -> RunInputs:
    """The prototype's launch on the 48,000-resident toy city (D-027 parity fixture)."""
    city = build_toy_city(seed=49, residents=PROTOTYPE_RESIDENTS)
    scenario, params = load_scenario(PROTOTYPE_SCENARIO_PATH)
    return RunInputs(city, bytes.fromhex(pack_hash(city)), scenario, params, seed)


def prototype_small_inputs(seed: int) -> RunInputs:
    """The prototype launch on the 5,000-resident toy city: goldens and fast tests.

    Budgets are absolute, so paid reach is scaled by the resident ratio to keep the same
    impressions per resident as the 48,000-resident run.
    """
    scenario, params = load_scenario(PROTOTYPE_SCENARIO_PATH)
    reach = params.paid_scale * GOLDEN_RESIDENTS / PROTOTYPE_RESIDENTS
    return golden_inputs(seed, scenario, dataclasses.replace(params, paid_scale=reach))


def supply_small_inputs(seed: int) -> RunInputs:
    """The small prototype launch with supply on at the MODEL.md section 12 defaults."""
    return _with_supply(prototype_small_inputs(seed), Supply())


def low_stock_inputs(seed: int) -> RunInputs:
    """The same launch with a ninth of the default stock and a two-day lead time, so stores
    stock out (P-013 acceptance)."""
    return _with_supply(prototype_small_inputs(seed), LOW_STOCK)


def slow_riders_inputs(seed: int) -> RunInputs:
    """The same launch with one slow rider per store, so deliveries queue and run late."""
    return _with_supply(prototype_small_inputs(seed), SLOW_RIDERS)


def _with_supply(inputs: RunInputs, supply: Supply) -> RunInputs:
    scenario = dataclasses.replace(inputs.scenario, supply=supply)
    return dataclasses.replace(inputs, scenario=scenario)


GOLDEN_RUNS: tuple[tuple[str, Callable[[int], RunInputs], int], ...] = (
    ("empty_seed1", golden_inputs, 1),
    ("empty_seed2", golden_inputs, 2),
    ("prototype_small_seed1", prototype_small_inputs, 1),
    ("prototype_small_seed2", prototype_small_inputs, 2),
    ("supply_small_seed1", supply_small_inputs, 1),
    ("supply_small_seed2", supply_small_inputs, 2),
    ("low_stock_seed1", low_stock_inputs, 1),
    ("low_stock_seed2", low_stock_inputs, 2),
    ("slow_riders_seed1", slow_riders_inputs, 1),
    ("slow_riders_seed2", slow_riders_inputs, 2),
)
SUPPLY_RUNS = ("supply_small", "low_stock", "slow_riders")


def golden_run_pid(seed: int) -> int:
    """Run a golden seed and report the worker's pid; lets tests see how a pool spreads work."""
    run_seed_bytes(golden_inputs(seed))
    return os.getpid()


def tape_probes(data: bytes) -> Probes:
    tape = read_tape(data)
    aggregates = tape.aggregates
    return {
        "hash": tape_hash(data),
        "aware_day30": int(aggregates.aware[-1]),
        "adopted_day30": int(aggregates.adopted[-1]),
        "orders_total": int(aggregates.orders.sum()),
        "tape_bytes": len(data),
        "chatter_rows": tape.header.chatter_rows,
    }


def supply_probes(tape: Tape, stats: SupplyStats) -> Probes:
    """Deliveries and stockouts from the tape; late deliveries and lost sales from the stats."""
    return {
        "deliveries": tape.header.delivery_rows,
        "late": stats.late,
        "stockout_pairs": tape.header.stockout_pairs,
        "lost": stats.lost,
    }


def compute_engine_goldens() -> dict[str, Probes]:
    goldens: dict[str, Probes] = {}
    for name, inputs_for, seed in GOLDEN_RUNS:
        tape, stats = run_seed_with_stats(inputs_for(seed))
        goldens[name] = tape_probes(write_tape(tape))
        if name.startswith(SUPPLY_RUNS):
            goldens[name] |= supply_probes(tape, stats)
    return goldens


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
