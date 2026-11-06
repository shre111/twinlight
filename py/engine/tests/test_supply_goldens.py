import pickle
from collections.abc import Callable

import pytest

from twl_engine.goldens import (
    GOLDEN_RUNS,
    LOW_STOCK,
    low_stock_inputs,
    prototype_small_inputs,
    read_goldens,
    supply_probes,
    supply_small_inputs,
    tape_probes,
)
from twl_engine.run import RunInputs, run_seed_with_stats
from twl_engine.scenario import Supply
from twl_tape import write_tape

SUPPLY_PROBES = ("deliveries", "late", "stockout_pairs", "lost")


def test_supply_golden_inputs_are_the_small_prototype_launch_with_supply_on() -> None:
    base, stocked, low = prototype_small_inputs(1), supply_small_inputs(1), low_stock_inputs(1)
    assert base.scenario.supply is None
    assert stocked.scenario.supply == Supply()
    assert low.scenario.supply == LOW_STOCK
    assert (
        Supply(initial_stock=0.004, reorder_point=0.001, order_up_to=0.004, lead_minutes=2880)
        == LOW_STOCK
    )
    for inputs in (stocked, low):
        assert inputs.params == base.params
        assert inputs.city_pack_hash == base.city_pack_hash
        assert inputs.scenario.campaigns == base.scenario.campaigns
        assert inputs.seed == 1


def test_supply_golden_inputs_can_be_sent_to_worker_processes() -> None:
    for inputs_for in (supply_small_inputs, low_stock_inputs):
        assert pickle.loads(pickle.dumps(inputs_for)) is inputs_for


def test_every_supply_fixture_is_pinned_at_seeds_1_and_2() -> None:
    names = [name for name, _, _ in GOLDEN_RUNS]
    for fixture in ("supply_small", "low_stock"):
        assert f"{fixture}_seed1" in names and f"{fixture}_seed2" in names
    engine = read_goldens()["engine"]
    for name in names:
        assert name in engine, f"run `just goldens`: [engine.{name}] is missing"
        has_supply_probes = all(key in engine[name] for key in SUPPLY_PROBES)
        assert has_supply_probes == name.startswith(("supply_small", "low_stock")), name


@pytest.mark.golden
@pytest.mark.parametrize(
    ("name", "inputs_for"),
    [("supply_small_seed1", supply_small_inputs), ("low_stock_seed1", low_stock_inputs)],
)
def test_a_fresh_supply_run_gives_the_checked_in_probes(
    name: str, inputs_for: Callable[[int], RunInputs]
) -> None:
    tape, stats = run_seed_with_stats(inputs_for(1))
    actual = tape_probes(write_tape(tape)) | supply_probes(tape, stats)
    expected = read_goldens()["engine"][name]
    for key, value in expected.items():
        if key != "hash":
            assert actual[key] == value, f"[engine.{name}] {key}: {actual[key]} vs {value}"
    assert actual["deliveries"] == actual["orders_total"]


def test_the_low_stock_fixture_stocks_out_and_loses_sales() -> None:
    probes = supply_probes(*run_seed_with_stats(low_stock_inputs(1)))
    assert int(probes["stockout_pairs"]) > 0 and int(probes["lost"]) > 0
