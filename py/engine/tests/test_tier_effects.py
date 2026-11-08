import dataclasses

import numpy as np
import pytest

from twl_engine.clock import Clock
from twl_engine.fixtures import ToyCity, build_toy_city
from twl_engine.run import RunInputs, run_seed
from twl_engine.scenario import Campaign, Params, Scenario
from twl_engine.world import World, build_world

PRICE = 59.0


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=1_200)


def with_tiers(city: ToyCity) -> ToyCity:
    tier = (np.arange(city.residents.x.size) % 3).astype(np.uint8)
    return dataclasses.replace(city, residents=dataclasses.replace(city.residents, tier=tier))


def world(city: ToyCity, params: Params) -> World:
    scenario = Scenario(days=1, price=PRICE)
    return build_world(city, scenario, params, Clock(1440, 15), np.random.default_rng(3))


def test_tier_effects_default_to_neutral_and_need_three_values() -> None:
    assert Params().alpha_tier == (0.0, 0.0, 0.0)
    assert Params().beta_tier == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="3 values"):
        Params(alpha_tier=(0.0, 0.0))
    with pytest.raises(ValueError, match="beta_tier"):
        Params(beta_tier=(1.0, -0.5, 1.0))
    with pytest.raises(ValueError, match="alpha_tier must be finite"):
        Params(alpha_tier=(0.0, float("nan"), 0.0))
    assert Params(alpha_tier=(0, 0, 0)).hash() == Params().hash()  # type: ignore[arg-type]


def test_alpha_tier_shifts_only_the_first_order_logit_of_that_tier(city: ToyCity) -> None:
    tiered = with_tiers(city)
    base = world(tiered, Params()).choice
    shifted = world(tiered, Params(alpha_tier=(-0.5, 0.0, 0.75))).choice
    tier = tiered.residents.tier
    expected = np.choose(tier, [-0.5, 0.0, 0.75])
    np.testing.assert_allclose(shifted.first_logit - base.first_logit, expected, atol=1e-12)
    np.testing.assert_array_equal(shifted.repeat_logit, base.repeat_logit)


def test_beta_tier_scales_the_price_term_of_that_tier(city: ToyCity) -> None:
    tiered = with_tiers(city)
    params = Params(beta=1.5)
    base = world(tiered, params).choice
    scaled = world(tiered, dataclasses.replace(params, beta_tier=(2.0, 1.0, 0.5))).choice
    price_term = 1.5 * PRICE / Scenario().ref_price
    extra = -price_term * (np.choose(tiered.residents.tier, [2.0, 1.0, 0.5]) - 1.0)
    np.testing.assert_allclose(scaled.first_logit - base.first_logit, extra, atol=1e-12)
    np.testing.assert_allclose(scaled.repeat_logit - base.repeat_logit, extra, atol=1e-12)


def test_with_neutral_effects_tiers_change_nothing(city: ToyCity) -> None:
    launch = Campaign("launch", 0, 1440, 600.0, 100.0, channel="social")
    scenario = Scenario(days=3, price=PRICE, campaigns=(launch,))
    params = Params(alpha_0=-1.0, q=0.05)
    plain = run_seed(RunInputs(city, bytes(32), scenario, params, 7))
    tiered = run_seed(RunInputs(with_tiers(city), bytes(32), scenario, params, 7))
    for section in ("residents", "chatter", "aggregates"):
        for field in dataclasses.fields(getattr(plain, section)):
            np.testing.assert_array_equal(
                getattr(getattr(plain, section), field.name),
                getattr(getattr(tiered, section), field.name),
                err_msg=f"{section}.{field.name}",
            )
