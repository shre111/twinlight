import pytest

from twl_engine.goldens import golden_inputs
from twl_engine.montecarlo import run_seeds
from twl_tape import read_tape


def test_tapes_come_back_in_seed_order_from_worker_processes() -> None:
    seeds = [5, 1, 3]
    tapes = run_seeds(golden_inputs, seeds, workers=2)
    assert [read_tape(data).header.seed for data in tapes] == seeds
    assert tapes == run_seeds(golden_inputs, seeds, workers=1)


def test_run_seeds_rejects_zero_workers() -> None:
    with pytest.raises(ValueError):
        run_seeds(golden_inputs, [1], workers=0)
