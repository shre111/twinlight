import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from twl_engine.fit_prototype import (
    FIT_DAYS,
    PROTOTYPE_SHARES,
    REPORT_DAYS,
    Fit,
    fit_inputs,
    rewrite_params,
)
from twl_engine.fixtures.geometry import DISTRICT_TABLE
from twl_engine.goldens import (
    PROTOTYPE_SCENARIO_PATH,
    prototype_small_inputs,
    read_goldens,
)
from twl_engine.scenario import PROTOTYPE_TOD, Campaign, Params, Scenario, load_scenario

MINIMAL = """
[scenario]
days = 2
price = 59

[[scenario.campaigns]]
name = "burst"
start_minute = 0
end_minute = 1440
budget_per_day = 100
cpm = 50

[params]
q = 0.5
kappa = [1, 1, 1, 1, 1]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scenario.toml"
    path.write_text(text)
    return path


def test_loader_builds_the_scenario_and_params_it_reads(tmp_path: Path) -> None:
    scenario, params = load_scenario(write(tmp_path, MINIMAL))
    assert scenario == Scenario(
        days=2, price=59.0, campaigns=(Campaign("burst", 0, 1440, 100.0, 50.0),)
    )
    assert params == Params(q=0.5, kappa=(1.0,) * 5)
    assert scenario.hash() == Scenario(days=2, price=59.0, campaigns=scenario.campaigns).hash()


def test_loader_accepts_an_empty_file_as_the_defaults(tmp_path: Path) -> None:
    assert load_scenario(write(tmp_path, "")) == (Scenario(), Params())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("[params]", "[parameters]"),
        ("days = 2", "day = 2"),
        ("cpm = 50", "cpm = 50\nreach = 3"),
        ("q = 0.5", "q = 0.5\nqq = 1"),
    ],
)
def test_loader_rejects_unknown_keys(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        load_scenario(write(tmp_path, MINIMAL.replace(old, new)))


def test_fixture_is_three_campaigns_in_the_prototype_reach_ratio() -> None:
    scenario, _ = load_scenario(PROTOTYPE_SCENARIO_PATH)
    launch, drip, creator = scenario.campaigns
    assert [c.name for c in scenario.campaigns] == ["launch", "drip", "creator"]
    assert (scenario.days, scenario.tick_minutes, scenario.price, scenario.ref_price) == (
        30,
        15,
        49.0,
        49.0,
    )
    assert (launch.start_minute, launch.end_minute) == (540, 1440)
    assert (drip.start_minute, drip.end_minute) == (1440, 43_200)
    assert (creator.start_minute, creator.end_minute) == (186 * 60, 216 * 60)
    assert launch.impressions_per_day == pytest.approx(48_000.0)
    assert drip.impressions_per_day / launch.impressions_per_day == pytest.approx(0.0214)
    assert 2 * creator.impressions_per_day / launch.impressions_per_day == pytest.approx(0.75)
    for campaign in scenario.campaigns:
        assert campaign.cpm == 100.0 and campaign.frequency_cap == 3
        assert campaign.presence == PROTOTYPE_TOD and campaign.segment_weights == ()
    names = [name for name, _, _ in DISTRICT_TABLE]
    boosted = {names[i] for i, w in enumerate(creator.district_weights) if w == 10.0}
    assert boosted == {"Satellite", "Bodakdev", "Vastrapur"}
    assert sorted(set(creator.district_weights)) == [1.0, 10.0]
    assert launch.district_weights == drip.district_weights == ()


def test_fixture_params_differ_from_the_defaults_only_in_the_three_fitted_values() -> None:
    _, params = load_scenario(PROTOTYPE_SCENARIO_PATH)
    defaults = dataclasses.asdict(Params())
    changed = {k for k, v in dataclasses.asdict(params).items() if v != defaults[k]}
    assert changed <= {"paid_scale", "q", "alpha_0"}
    assert "[params]" in PROTOTYPE_SCENARIO_PATH.read_text().split("never edit by hand.\n")[1]


def test_fit_targets_are_days_6_and_13_and_day_30_is_only_reported() -> None:
    assert FIT_DAYS == (6, 13)
    assert set(FIT_DAYS) < set(REPORT_DAYS) and 30 in REPORT_DAYS
    assert PROTOTYPE_SHARES[6] == (23.0, 4.5) and PROTOTYPE_SHARES[13] == (41.7, 10.8)
    assert PROTOTYPE_SHARES[30] == (63.1, 23.0)


def test_fit_points_round_to_four_significant_digits() -> None:
    fit = Fit.at(np.array([math.log(1.23456789), math.log(0.0123456789), -6.123456]))
    assert fit == Fit(paid_scale=1.235, q=0.01235, alpha_0=-6.123)
    assert Fit.at(fit.coordinates()) == fit


def test_fit_inputs_override_only_the_fitted_params_and_the_length() -> None:
    fit = Fit(paid_scale=2.0, q=0.5, alpha_0=-1.0)
    inputs = fit_inputs(fit, 13, 4)
    scenario, params = load_scenario(PROTOTYPE_SCENARIO_PATH)
    assert inputs.seed == 4 and inputs.city.residents.x.shape[0] == 48_000
    assert inputs.scenario == dataclasses.replace(scenario, days=13)
    assert inputs.params == dataclasses.replace(params, paid_scale=2.0, q=0.5, alpha_0=-1.0)


def test_rewriting_params_keeps_the_scenario_and_round_trips(tmp_path: Path) -> None:
    path = write(tmp_path, PROTOTYPE_SCENARIO_PATH.read_text())
    before, _ = load_scenario(path)
    rewrite_params(path, Fit(paid_scale=1.5, q=0.0125, alpha_0=-5.75))
    after, params = load_scenario(path)
    assert after == before
    assert params == Params(paid_scale=1.5, q=0.0125, alpha_0=-5.75)
    rewrite_params(path, Fit(paid_scale=1.5, q=0.0125, alpha_0=-5.75))
    assert load_scenario(path) == (after, params)


def test_prototype_small_inputs_are_the_fixture_on_the_golden_city() -> None:
    inputs = prototype_small_inputs(2)
    assert inputs.seed == 2 and inputs.city.residents.x.shape[0] == 5_000
    scenario, params = load_scenario(PROTOTYPE_SCENARIO_PATH)
    assert inputs.scenario == scenario
    same_reach = dataclasses.replace(inputs.params, paid_scale=params.paid_scale)
    assert same_reach == params
    assert inputs.params.paid_scale == pytest.approx(params.paid_scale * 5_000 / 48_000)


def test_checked_in_goldens_cover_every_engine_run_with_all_probes() -> None:
    engine = read_goldens()["engine"]
    probes = ["hash", "aware_day30", "adopted_day30", "orders_total", "tape_bytes", "chatter_rows"]
    assert list(engine) == [
        "empty_seed1",
        "empty_seed2",
        "prototype_small_seed1",
        "prototype_small_seed2",
    ]
    assert all(list(table) == probes for table in engine.values())
    assert engine["empty_seed1"]["chatter_rows"] == 0
    assert int(engine["prototype_small_seed1"]["chatter_rows"]) > 0
