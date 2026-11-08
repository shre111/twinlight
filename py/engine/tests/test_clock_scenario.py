import json

import pytest

from twl_engine.clock import Clock
from twl_engine.scenario import EMPTY, Params, Scenario


def test_thirty_days_of_fifteen_minute_ticks() -> None:
    clock = Clock(sim_minutes=43_200, tick_minutes=15)
    assert clock.ticks == 2880
    assert clock.tick_start(0) == 0
    assert clock.tick_start(2879) == 43_185


def test_clock_rejects_partial_ticks_and_out_of_range_ticks() -> None:
    with pytest.raises(ValueError):
        Clock(sim_minutes=100, tick_minutes=15)
    with pytest.raises(ValueError):
        Clock(sim_minutes=30, tick_minutes=15).tick_start(2)


def test_empty_scenario_canonical_form_is_stable() -> None:
    assert json.loads(EMPTY.canonical_json()) == {
        "campaigns": [],
        "days": 30,
        "price": 49.0,
        "ref_price": 49.0,
        "start_date": None,
        "supply": None,
        "tick_minutes": 15,
    }
    assert EMPTY.sim_minutes == 43_200
    assert EMPTY.hash() == Scenario().hash()
    assert Scenario(days=1).hash() != EMPTY.hash()
    assert len(Params().hash()) == 32
