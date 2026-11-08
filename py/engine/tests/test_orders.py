import numpy as np
import pytest

from twl_engine.clock import Clock
from twl_engine.fixtures import ToyCity, build_toy_city, build_uniform_city
from twl_engine.orders import NUDGE_WINDOW_MINUTES, log_nudges, sigmoid
from twl_engine.run import RunInputs, run_seed
from twl_engine.scenario import Campaign, Params, Scenario
from twl_engine.state import new_chatter_log, new_resident_state
from twl_engine.world import occasion_probability_per_tick
from twl_tape import CHATTER_HEARD, CHATTER_NUDGED, NEVER, SOURCE_PAID, SOURCE_WOM, Tape

RESIDENTS = 5_000
DAY = Clock(sim_minutes=1440, tick_minutes=15)
LIVELY = Params(q=0.06, alpha_0=-2.0)


def burst(
    residents: int, per_resident: float = 1.0, days: int = 8, price: float = 49.0
) -> Scenario:
    campaign = Campaign(
        name="launch",
        start_minute=0,
        end_minute=1440,
        budget_per_day=per_resident * residents * 100.0 / 1000.0,
        cpm=100.0,
        channel="social",
    )
    return Scenario(days=days, price=price, campaigns=(campaign,))


def run(city: ToyCity, scenario: Scenario, params: Params, seed: int = 1) -> Tape:
    return run_seed(RunInputs(city, bytes(32), scenario, params, seed))


@pytest.fixture(scope="module")
def toy() -> ToyCity:
    return build_toy_city(seed=49, residents=RESIDENTS)


@pytest.fixture(scope="module")
def lively(toy: ToyCity) -> Tape:
    return run(toy, burst(RESIDENTS), LIVELY)


def test_sigmoid_is_stable_at_the_extremes() -> None:
    values = sigmoid(np.array([-800.0, 0.0, 800.0, np.log(3.0)]))
    assert values == pytest.approx([0.0, 0.5, 1.0, 0.75])


def test_occasion_rate_integrates_to_occasions_per_day() -> None:
    by_segment = occasion_probability_per_tick(Params(), DAY)
    assert by_segment.shape == (96, 5) and (by_segment == by_segment[:, :1]).all()
    chance = by_segment[:, 0]
    assert -np.log1p(-chance).sum() == pytest.approx(3.0)
    assert chance.argmax() // 4 == 20
    flat = occasion_probability_per_tick(Params(occasion_profile=(1.0,) * 24), DAY)
    assert flat == pytest.approx(1.0 - np.exp(-3.0 / 96.0))
    assert not occasion_probability_per_tick(Params(occasions_per_day=0.0), DAY).any()


def test_orders_follow_awareness_and_counts_agree(lively: Tape) -> None:
    residents, aggregates = lively.residents, lively.aggregates
    ordered = residents.first_order_at != NEVER
    assert 200 < ordered.sum() < RESIDENTS
    assert (residents.aware_at[ordered] <= residents.first_order_at[ordered]).all()
    assert np.array_equal(ordered, residents.order_count > 0)
    assert int(aggregates.orders.sum()) == int(residents.order_count.sum())
    assert residents.order_count.max() > 1
    ends = (np.arange(aggregates.adopted.shape[0]) + 1) * 15
    for tick in (95, 300, 767):
        assert aggregates.adopted[tick] == (residents.first_order_at < ends[tick]).sum()
        assert aggregates.aware[tick] == (residents.aware_at < ends[tick]).sum()
    assert (np.diff(aggregates.adopted.astype(np.int64)) >= 0).all()


def test_nudged_rows_are_the_last_talk_before_a_first_order(lively: Tape) -> None:
    residents, chatter = lively.residents, lively.chatter
    nudged = chatter.kind == CHATTER_NUDGED
    heard = chatter.kind == CHATTER_HEARD
    assert nudged.sum() > 20 and heard.sum() > 20
    to, at = chatter.to[nudged], chatter.at[nudged].astype(np.int64)
    first_order = residents.first_order_at[to].astype(np.int64)
    assert (residents.first_order_at[to] != NEVER).all()
    assert (at <= first_order).all() and (first_order - at <= NUDGE_WINDOW_MINUTES).all()
    assert np.unique(to).shape == to.shape
    heard_rows = set(
        zip(
            chatter.from_[heard].tolist(),
            chatter.to[heard].tolist(),
            chatter.at[heard].tolist(),
            strict=True,
        )
    )
    nudged_rows = set(zip(chatter.from_[nudged].tolist(), to.tolist(), at.tolist(), strict=True))
    assert not heard_rows & nudged_rows
    assert lively.header.chatter_rows == chatter.at.shape[0]
    keys = list(zip(chatter.at.tolist(), chatter.to.tolist(), chatter.kind.tolist(), strict=True))
    assert keys == sorted(keys)


def test_no_occasions_means_no_orders(toy: ToyCity) -> None:
    tape = run(toy, burst(RESIDENTS, days=3), Params(occasions_per_day=0.0))
    assert not tape.residents.order_count.any() and not tape.aggregates.orders.any()
    assert (tape.residents.first_order_at == NEVER).all()
    assert (tape.chatter.kind == CHATTER_HEARD).all()


def test_a_hopeless_repeat_intercept_stops_repeats(toy: ToyCity) -> None:
    tape = run(toy, burst(RESIDENTS, days=4), Params(alpha_0=-2.0, alpha_rep=-60.0))
    assert tape.residents.order_count.max() == 1


def test_tripling_the_price_cuts_first_orders_by_the_logit_shift() -> None:
    """With rare orders the adopter ratio is close to exp(-beta * (147 - 49) / 49) = e^-2."""
    city = build_uniform_city(40_000)
    rare = Params(
        impression_gain=1.0,
        kappa=(1.0,),
        alpha_segment=(0.0,),
        alpha_0=-4.5,
        gamma=0.0,
        affinity_sd=0.0,
        rho_sd=0.0,
        q=0.0,
        awareness_half_life_days=None,
    )
    cheap = run(city, burst(40_000, per_resident=5.0, days=2), rare)
    dear = run(city, burst(40_000, per_resident=5.0, days=2, price=147.0), rare)
    ratio = dear.aggregates.adopted[-1] / cheap.aggregates.adopted[-1]
    assert cheap.aggregates.adopted[-1] > 1_500
    assert ratio == pytest.approx(np.exp(-2.0), rel=0.25)


def test_order_on_awareness_orders_in_the_minute_of_awareness() -> None:
    city = build_uniform_city(4_000)
    params = Params(
        impression_gain=1.0,
        talk_gain=1.0,
        hearsay_factor=0.0,
        talk_decay_hours=None,
        awareness_half_life_days=None,
        q=0.02,
        occasions_per_day=0.0,
        wom_mixing="uniform",
        order_on_awareness=True,
    )
    tape = run(city, burst(4_000, per_resident=0.05, days=6), params)
    residents = tape.residents
    aware = residents.aware_at != NEVER
    assert 400 < aware.sum() < 4_000
    assert np.array_equal(residents.first_order_at, residents.aware_at)
    assert np.array_equal(residents.order_count, aware.astype(np.uint16))
    assert np.array_equal(tape.aggregates.adopted, tape.aggregates.aware)
    assert int(tape.aggregates.orders.sum()) == aware.sum()
    assert set(np.unique(residents.first_source[aware]).tolist()) == {SOURCE_PAID, SOURCE_WOM}
    assert (tape.chatter.kind == CHATTER_HEARD).all()


def test_order_on_awareness_still_repeats_when_occasions_fire() -> None:
    city = build_uniform_city(2_000)
    params = Params(impression_gain=1.0, q=0.0, order_on_awareness=True, alpha_rep=2.0)
    tape = run(city, burst(2_000, per_resident=0.5, days=2), params)
    assert tape.residents.order_count.max() > 1
    assert int(tape.aggregates.orders.sum()) == int(tape.residents.order_count.sum())


def nudge_case(
    talk_at: int, talk_from: int, order_at: int, source: int, aware_at: int, heard_from: int
) -> list[tuple[int, int, int, int]]:
    residents = new_resident_state(3)
    log = new_chatter_log(3)
    residents.last_talk_at[1] = talk_at
    residents.last_talk_from[1] = talk_from
    residents.first_source[1] = source
    residents.aware_at[1] = aware_at
    residents.heard_from[1] = heard_from
    log_nudges(residents, log, np.array([1]), np.array([order_at]))
    rows = range(log.rows)
    return [(int(log.from_[i]), int(log.to[i]), int(log.at[i]), int(log.kind[i])) for i in rows]


def test_a_recent_talk_that_is_not_the_heard_talk_is_a_nudge() -> None:
    after_paid = nudge_case(5_000, 2, 6_000, SOURCE_PAID, 4_000, NEVER)
    assert after_paid == [(2, 1, 5_000, CHATTER_NUDGED)]
    later_talk = nudge_case(5_000, 2, 6_000, SOURCE_WOM, 4_000, 0)
    assert later_talk == [(2, 1, 5_000, CHATTER_NUDGED)]
    at_the_window_edge = nudge_case(1_680, 2, 6_000, SOURCE_PAID, 100, NEVER)
    assert at_the_window_edge == [(2, 1, 1_680, CHATTER_NUDGED)]


def test_no_nudge_for_the_heard_talk_an_old_talk_a_later_talk_or_no_talk() -> None:
    assert nudge_case(5_000, 2, 6_000, SOURCE_WOM, 5_000, 2) == []
    assert nudge_case(1_679, 2, 6_000, SOURCE_PAID, 100, NEVER) == []
    assert nudge_case(6_005, 2, 6_000, SOURCE_PAID, 6_000, NEVER) == []
    assert nudge_case(NEVER, NEVER, 6_000, SOURCE_PAID, 4_000, NEVER) == []
