import dataclasses
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.fixtures.types import Residents, Stores
from twl_engine.run import RunInputs, run_seed_with_stats
from twl_engine.scenario import Campaign, Params, Scenario, Supply, load_scenario
from twl_engine.world import NO_STORE, build_supply_plan, stock_level


@lru_cache(maxsize=1)
def toy() -> ToyCity:
    return build_toy_city(seed=49, residents=600, stores=3)


def line_city(resident_x: list[float], store_x: list[float], radius: list[float]) -> ToyCity:
    """Residents and stores on the x axis, so distances can be read off the test."""
    base = toy()
    n, k = len(resident_x), len(store_x)
    residents = Residents(
        x=np.array(resident_x, dtype=np.float32),
        y=np.zeros(n, dtype=np.float32),
        z=np.zeros(n, dtype=np.float32),
        building=np.zeros(n, dtype=np.uint32),
        hex=np.zeros(n, dtype=np.uint32),
        segment=np.zeros(n, dtype=np.uint16),
        persona_ref=np.zeros(n, dtype=np.uint32),
        tier=np.ones(n, dtype=np.uint8),
        weight=np.ones(n, dtype=np.float32),
    )
    stores = Stores(
        x=np.array(store_x, dtype=np.float32),
        y=np.zeros(k, dtype=np.float32),
        hex=np.zeros(k, dtype=np.uint32),
        radius_m=np.array(radius, dtype=np.float32),
        district=np.zeros(k, dtype=np.uint16),
    )
    return dataclasses.replace(base, residents=residents, stores=stores)


def test_supply_defaults_are_the_model_section_12_values() -> None:
    s = Supply()
    assert (s.initial_stock, s.reorder_point, s.order_up_to) == (0.036, 0.012, 0.036)
    assert (s.review_minutes, s.lead_minutes) == (60, 1440)
    assert (s.picks_per_hour, s.riders_per_store, s.rider_speed_kmh) == (120.0, 6, 20.0)
    assert (s.travel_noise_sigma, s.promise_minutes) == (0.15, 20)
    assert (s.spill_radius_factor, s.wait_minutes) == (1.67, 1440)
    p = Params()
    assert (p.p_spill, p.p_wait, p.satisfaction_drop) == (0.42, 0.38, 0.8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_stock": -0.1},
        {"order_up_to": float("inf")},
        {"review_minutes": 0},
        {"lead_minutes": -1},
        {"picks_per_hour": 0.0},
        {"riders_per_store": 0},
        {"rider_speed_kmh": 0.0},
        {"travel_noise_sigma": -0.1},
        {"wait_minutes": -5},
    ],
)
def test_supply_rejects_values_outside_its_domain(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Supply(**overrides)  # type: ignore[arg-type]


def test_supply_minutes_must_be_whole() -> None:
    with pytest.raises(TypeError):
        Supply(lead_minutes=90.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Scenario(supply={"lead_minutes": 60})  # type: ignore[arg-type]


@pytest.mark.parametrize("overrides", [{"p_spill": -0.1}, {"p_spill": 0.7, "p_wait": 0.4}])
def test_params_reject_out_of_stock_shares_that_are_not_shares(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        Params(**overrides)  # type: ignore[arg-type]


def test_supply_is_part_of_the_scenario_hash() -> None:
    assert json.loads(Scenario().canonical_json())["supply"] is None
    assert Scenario(supply=Supply()).hash() != Scenario().hash()
    assert Scenario(supply=Supply(lead_minutes=60)).hash() != Scenario(supply=Supply()).hash()
    assert Scenario(supply=Supply(riders_per_store=6)).hash() == Scenario(supply=Supply()).hash()
    assert Params(p_spill=0.3).hash() != Params().hash()


def test_load_scenario_reads_an_optional_supply_table(tmp_path: Path) -> None:
    plain = tmp_path / "plain.toml"
    plain.write_text("[scenario]\ndays = 2\n")
    assert load_scenario(plain)[0].supply is None

    stocked = tmp_path / "stocked.toml"
    stocked.write_text(
        "[scenario]\ndays = 2\n\n[scenario.supply]\ninitial_stock = 0.004\nlead_minutes = 2880\n"
        "\n[[scenario.campaigns]]\nname = 'launch'\nchannel = 'social'\nstart_minute = 0\n"
        "end_minute = 1440\n"
        "budget_per_day = 10.0\ncpm = 100.0\n\n[params]\np_wait = 0.5\n"
    )
    scenario, params = load_scenario(stocked)
    assert scenario.supply == Supply(initial_stock=0.004, lead_minutes=2880)
    assert [c.name for c in scenario.campaigns] == ["launch"]
    assert params.p_wait == 0.5

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("[scenario.supply]\nriders = 3\n")
    with pytest.raises(ValueError, match="riders"):
        load_scenario(unknown)


def test_own_store_is_the_nearest_store_whose_radius_holds_the_resident() -> None:
    city = line_city(
        resident_x=[0.0, 400.0, 600.0, 1000.0, 2500.0],
        store_x=[0.0, 1000.0],
        radius=[900.0, 300.0],
    )
    plan = build_supply_plan(city, Supply(spill_radius_factor=2.0))
    assert plan.own_store.tolist() == [0, 0, 0, 1, NO_STORE]
    assert plan.own_distance.tolist() == [0.0, 400.0, 600.0, 0.0, 0.0]
    assert plan.spill_store.tolist() == [NO_STORE, 1, 1, 0, NO_STORE]
    assert plan.spill_distance.tolist() == [0.0, 600.0, 400.0, 1000.0, 0.0]
    assert plan.catchment.tolist() == [3, 1]


def test_a_tie_between_stores_goes_to_the_lower_index_and_no_stores_serve_nobody() -> None:
    tied = build_supply_plan(line_city([500.0], [0.0, 1000.0], [900.0, 900.0]), Supply())
    assert (tied.own_store.tolist(), tied.spill_store.tolist()) == ([0], [1])
    empty = build_supply_plan(line_city([500.0], [], []), Supply())
    assert empty.own_store.tolist() == [NO_STORE] and empty.stores == 0


def test_toy_city_tables_match_a_resident_by_resident_search() -> None:
    city, supply = toy(), Supply()
    plan = build_supply_plan(city, supply)
    sx, sy = city.stores.x.astype(float), city.stores.y.astype(float)
    radius = city.stores.radius_m.astype(float)
    for r in range(0, city.residents.x.shape[0], 7):
        d = np.hypot(float(city.residents.x[r]) - sx, float(city.residents.y[r]) - sy)
        inside = [k for k in np.argsort(d, kind="stable") if d[k] <= radius[k]]
        own = inside[0] if inside else NO_STORE
        assert plan.own_store[r] == own
        if own == NO_STORE:
            assert plan.spill_store[r] == NO_STORE
            continue
        assert plan.own_distance[r] == d[own]
        near = [
            k
            for k in np.argsort(d, kind="stable")
            if k != own and d[k] <= supply.spill_radius_factor * radius[k]
        ]
        assert plan.spill_store[r] == (near[0] if near else NO_STORE)
    assert plan.catchment.sum() == np.count_nonzero(plan.own_store != NO_STORE)


def test_store_levels_are_the_rounded_rate_times_the_catchment() -> None:
    catchment = np.array([0, 1, 100, 357, 4000])
    assert stock_level(0.036, catchment).tolist() == [0, 0, 4, 13, 144]
    plan = build_supply_plan(toy(), Supply(initial_stock=0.05, reorder_point=0.01))
    assert plan.initial_stock.tolist() == [round(0.05 * c) for c in plan.catchment.tolist()]
    assert plan.reorder_point.tolist() == [round(0.01 * c) for c in plan.catchment.tolist()]
    assert plan.order_up_to.tolist() == [round(0.036 * c) for c in plan.catchment.tolist()]


def test_without_supply_every_order_is_a_served_attempt() -> None:
    launch = Campaign("launch", 0, 1440, budget_per_day=120.0, cpm=100.0, channel="social")
    scenario = Scenario(days=3, campaigns=(launch,))
    tape, stats = run_seed_with_stats(RunInputs(toy(), bytes(32), scenario, Params(), seed=5))
    orders = int(tape.aggregates.orders.sum())
    assert orders > 0
    assert (stats.attempts, stats.served) == (orders, orders)
    assert (stats.spilled, stats.deferred, stats.lost, stats.late) == (0, 0, 0, 0)
    assert stats.unserviceable_residents == 0
    for per_store in (stats.catchment, stats.initial_stock, stats.restocked, stats.stock_end):
        assert per_store.tolist() == [0, 0, 0]
    assert stats.served_by_store.tolist() == [0, 0, 0]
