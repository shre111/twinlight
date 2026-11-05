import dataclasses
import json
import math

import numpy as np
import pytest

from twl_engine.clock import Clock
from twl_engine.scenario import PROTOTYPE_TOD, Campaign, Params, Scenario
from twl_engine.world import awareness_decay_per_tick, tick_shares

CLOCK = Clock(sim_minutes=3 * 1440, tick_minutes=15)


def campaign(**overrides: object) -> Campaign:
    fields: dict[str, object] = {
        "name": "launch",
        "start_minute": 540,
        "end_minute": 1440,
        "budget_per_day": 4800.0,
        "cpm": 100.0,
    }
    return Campaign(**(fields | overrides))  # type: ignore[arg-type]


def test_params_defaults_are_the_model_section_11_values() -> None:
    p = Params()
    assert (p.theta_aware, p.impression_gain, p.talk_gain, p.talk_floor, p.hearsay_factor) == (
        1.0,
        0.6,
        1.0,
        0.35,
        0.15,
    )
    assert (p.talk_decay_hours, p.awareness_half_life_days) == (72.0, 4.0)
    assert p.kappa == (1.0, 0.9, 0.62, 0.38, 0.2)
    assert (p.q, p.talk_kind) == (0.02, (1.0, 0.6, 1.0, 0.5))
    assert p.occasions_per_day == 3.0 and p.occasion_profile == PROTOTYPE_TOD
    assert p.alpha_0 == -3.0
    prototype_affinity = [math.log(v) for v in (1.35, 1.2, 0.9, 0.6, 0.38)]
    assert p.alpha_segment == pytest.approx(prototype_affinity, abs=5e-5)
    assert (p.affinity_sd, p.gamma, p.beta, p.delta) == (0.3, 2.5, 1.0, 1.0)
    assert (p.alpha_rep, p.rho_sd, p.paid_scale) == (-2.3, 0.3, 1.0)
    assert p.wom_mixing == "graph" and p.order_on_awareness is False


def test_prototype_profile_has_24_hours_and_peaks_in_the_evening() -> None:
    assert len(PROTOTYPE_TOD) == 24
    assert max(range(24), key=PROTOTYPE_TOD.__getitem__) == 20


def test_equal_values_of_different_number_types_hash_equally() -> None:
    as_ints = Scenario(price=49, campaigns=[campaign(budget_per_day=4800, cpm=100)])  # type: ignore[arg-type]
    as_floats = Scenario(price=49.0, campaigns=(campaign(),))
    assert as_ints == as_floats
    assert as_ints.hash() == as_floats.hash()
    assert hash(as_ints) == hash(as_floats)
    assert Params(kappa=[1, 1, 1, 1, 1]).hash() == Params(kappa=(1.0,) * 5).hash()  # type: ignore[arg-type]


def test_canonical_json_carries_campaigns_and_none_as_null() -> None:
    scenario = json.loads(Scenario(campaigns=(campaign(frequency_cap=None),)).canonical_json())
    assert scenario["campaigns"][0]["name"] == "launch"
    assert scenario["campaigns"][0]["frequency_cap"] is None
    params = json.loads(Params(talk_decay_hours=None).canonical_json())
    assert params["talk_decay_hours"] is None and params["wom_mixing"] == "graph"


def test_scenario_and_params_hashes_follow_their_fields() -> None:
    assert Scenario(price=59.0).hash() != Scenario().hash()
    assert Scenario(campaigns=(campaign(),)).hash() != Scenario().hash()
    assert Params(q=0.03).hash() != Params().hash()
    assert dataclasses.replace(Params(), q=0.02).hash() == Params().hash()


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_minute": 1440, "end_minute": 1440},
        {"start_minute": -15},
        {"cpm": 0.0},
        {"budget_per_day": -1.0},
        {"frequency_cap": 0},
        {"presence": (1.0, 2.0)},
        {"district_weights": (1.0, -1.0)},
    ],
)
def test_campaign_rejects_values_outside_its_domain(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        campaign(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"theta_aware": 0.0},
        {"q": -0.1},
        {"talk_decay_hours": 0.0},
        {"awareness_half_life_days": -1.0},
        {"talk_kind": (1.0, 1.0)},
        {"occasion_profile": (1.0,) * 23},
        {"wom_mixing": "ring"},
    ],
)
def test_params_reject_values_outside_their_domain(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Params(**overrides)  # type: ignore[arg-type]


def test_scenario_rejects_fractional_days_and_bad_prices() -> None:
    with pytest.raises(TypeError):
        Scenario(days=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Scenario(ref_price=0.0)


def test_a_partial_day_spends_the_whole_day_over_its_active_ticks() -> None:
    shares = tick_shares(campaign(), CLOCK)
    assert shares[: 540 // 15].sum() == 0.0
    assert shares[540 // 15 : 96].sum() == pytest.approx(1.0)
    assert shares[96:].sum() == 0.0
    assert np.ptp(shares[540 // 15 : 96]) == pytest.approx(0.0)


def test_presence_shapes_the_day_and_every_active_day_gets_a_full_share() -> None:
    shares = tick_shares(campaign(start_minute=0, end_minute=2880, presence=PROTOTYPE_TOD), CLOCK)
    assert shares[:96].sum() == pytest.approx(1.0)
    assert shares[96:192].sum() == pytest.approx(1.0)
    assert shares[192:].sum() == 0.0
    assert shares[20 * 4] / shares[3 * 4] == pytest.approx(PROTOTYPE_TOD[20] / PROTOTYPE_TOD[3])


def test_a_day_with_no_presence_in_its_active_hours_gets_nothing() -> None:
    night_only = tuple(1.0 if hour < 6 else 0.0 for hour in range(24))
    shares = tick_shares(campaign(presence=night_only), CLOCK)
    assert shares.sum() == 0.0


def test_awareness_halves_over_one_half_life() -> None:
    per_tick = awareness_decay_per_tick(Params(awareness_half_life_days=4.0), CLOCK)
    assert per_tick ** (4 * 96) == pytest.approx(0.5)
    assert awareness_decay_per_tick(Params(awareness_half_life_days=None), CLOCK) == 1.0
