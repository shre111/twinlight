import math

import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city, build_uniform_city
from twl_engine.run import RunInputs, run_seed
from twl_engine.scenario import Campaign, Params, Scenario
from twl_engine.wom import apply_talks, pick_tie, talker_rates
from twl_tape import CHATTER_HEARD, NEVER, SOURCE_PAID, SOURCE_WOM, Tape

RESIDENTS = 5_000
HEARSAY_ONLY = Params(q=0.05, hearsay_factor=1.0, occasions_per_day=0.0)


def u32(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.uint32)


def day_one_burst(residents: int, per_resident: float = 1.0, days: int = 6) -> Scenario:
    campaign = Campaign(
        name="launch",
        start_minute=0,
        end_minute=1440,
        budget_per_day=per_resident * residents * 100.0 / 1000.0,
        cpm=100.0,
    )
    return Scenario(days=days, campaigns=(campaign,))


def run(city: ToyCity, scenario: Scenario, params: Params, seed: int = 1) -> Tape:
    return run_seed(RunInputs(city, bytes(32), scenario, params, seed))


@pytest.fixture(scope="module")
def toy() -> ToyCity:
    return build_toy_city(seed=49, residents=RESIDENTS)


@pytest.fixture(scope="module")
def hearsay_tape(toy: ToyCity) -> Tape:
    return run(toy, day_one_burst(RESIDENTS), HEARSAY_ONLY)


def test_talker_rates_follow_role_decay_and_the_tick_start_rule() -> None:
    start = 6_000
    aware_at = u32([NEVER, 1_000, 1_000, 6_003, 1_000])
    first_order_at = u32([NEVER, NEVER, 2_000, NEVER, 2_000])
    last_order_at = u32([NEVER, NEVER, 5_000, NEVER, 5_000])
    pull = np.array([2.0, 2.0, 3.0, 2.0, 0.0])
    cum = np.zeros(5)
    decay = 1.0 / (72.0 * 60.0)
    floor = 0.35
    total = talker_rates(
        aware_at, first_order_at, last_order_at, pull, start, 0.15, floor, decay, cum
    )
    hearsay = 0.15 * 2.0 * math.exp(-(start - 1_000) * decay)
    adopter = 3.0 * (floor + (1.0 - floor) * math.exp(-(start - 5_000) * decay))
    assert np.diff(cum, prepend=0.0) == pytest.approx([0.0, hearsay, adopter, 0.0, 0.0])
    assert total == pytest.approx(hearsay + adopter)


def test_talker_rates_without_decay_are_the_pull() -> None:
    cum = np.zeros(2)
    total = talker_rates(
        u32([10, 10]), u32([NEVER, 20]), u32([NEVER, 20]), np.ones(2), 9_000, 0.5, 0.35, 0.0, cum
    )
    assert cum.tolist() == [0.5, 1.5] and total == 1.5


def test_pick_tie_splits_the_pull_by_weight_and_skips_zero_weight_ties() -> None:
    weight = np.array([9.0, 0.0, 1.0, 3.0, 0.0])
    assert [pick_tie(1, 5, weight, point) for point in (0.0, 0.99, 1.0, 3.99, 4.0)] == [
        2,
        2,
        3,
        3,
        3,
    ]


def apply(draws: list[list[float]], awareness: np.ndarray, uniform: bool = False) -> dict:
    n = 4
    out = {
        "awareness": awareness,
        "aware_at": u32([0, 0, NEVER, NEVER]),
        "first_source": np.array([1, 1, 0, 0], dtype=np.uint8),
        "heard_from": u32([NEVER] * n),
        "last_talk_from": u32([NEVER] * n),
        "last_talk_at": u32([NEVER] * n),
        "log_from": u32([0] * 8),
        "log_to": u32([0] * 8),
        "log_at": u32([0] * 8),
        "log_kind": np.zeros(8, dtype=np.uint8),
    }
    out["rows"] = apply_talks(
        np.array(draws),
        np.array([1.0, 2.0, 2.0, 2.0]),
        150,
        15,
        uniform,
        np.array([0, 1, 2, 2, 2]),
        np.array([2, 2]),
        np.ones(2),
        np.array([0.9, 0.9]),
        np.ones(n),
        0.9,
        1.0,
        out["awareness"],
        out["aware_at"],
        out["first_source"],
        out["heard_from"],
        out["last_talk_from"],
        out["last_talk_at"],
        out["log_from"],
        out["log_to"],
        out["log_at"],
        out["log_kind"],
        0,
    )
    return out


def test_talks_apply_in_minute_then_talker_order_and_the_crossing_one_is_heard() -> None:
    late_from_0, early_from_1 = [0.1, 0.5, 0.9], [0.6, 0.5, 0.2]
    out = apply([late_from_0, early_from_1], np.zeros(4))
    assert out["awareness"][2] == pytest.approx(1.8)
    assert (out["aware_at"][2], out["heard_from"][2], out["first_source"][2]) == (163, 0, 2)
    assert (out["last_talk_from"][2], out["last_talk_at"][2]) == (0, 163)
    assert out["rows"] == 1
    assert (out["log_from"][0], out["log_to"][0], out["log_at"][0], out["log_kind"][0]) == (
        0,
        2,
        163,
        CHATTER_HEARD,
    )


def test_one_default_talk_is_not_enough_but_tops_up_earlier_awareness() -> None:
    alone = apply([[0.1, 0.5, 0.0]], np.zeros(4))
    assert alone["aware_at"][2] == NEVER and alone["rows"] == 0
    assert (alone["last_talk_from"][2], alone["last_talk_at"][2]) == (0, 150)
    topped_up = apply([[0.1, 0.5, 0.0]], np.array([0.0, 0.0, 0.2, 0.0]))
    assert topped_up["aware_at"][2] == 150 and topped_up["heard_from"][2] == 0


def test_uniform_mixing_never_talks_to_oneself() -> None:
    draws = [[0.1, x, 0.0] for x in (0.0, 0.34, 0.5, 0.999)]
    out = apply(draws, np.zeros(4), uniform=True)
    assert out["awareness"].tolist() == pytest.approx([0.0, 0.9, 1.8, 0.9])
    assert out["aware_at"][2] == 150


def test_heard_rows_are_the_word_of_mouth_crossings(toy: ToyCity, hearsay_tape: Tape) -> None:
    residents, chatter = hearsay_tape.residents, hearsay_tape.chatter
    by_talk = residents.first_source == SOURCE_WOM
    assert by_talk.sum() > 100
    assert hearsay_tape.header.chatter_rows == by_talk.sum() == chatter.at.shape[0]
    assert (chatter.kind == CHATTER_HEARD).all()
    assert np.array_equal(np.sort(chatter.to), np.flatnonzero(by_talk))
    assert np.array_equal(residents.aware_at[chatter.to], chatter.at)
    assert np.array_equal(residents.heard_from[chatter.to], chatter.from_)
    assert (residents.heard_from[~by_talk] == NEVER).all()
    assert set(np.unique(residents.first_source).tolist()) == {0, SOURCE_PAID, SOURCE_WOM}


def test_talkers_were_aware_before_the_tick_and_talk_along_their_ties(
    toy: ToyCity, hearsay_tape: Tape
) -> None:
    chatter = hearsay_tape.chatter
    tick_start = chatter.at - chatter.at % 15
    assert (hearsay_tape.residents.aware_at[chatter.from_] < tick_start).all()
    offsets, targets = toy.ties.offsets, toy.ties.targets
    assert all(
        to in targets[offsets[talker] : offsets[talker + 1]]
        for talker, to in zip(chatter.from_.tolist(), chatter.to.tolist(), strict=True)
    )


def test_chatter_is_sorted_by_minute_then_listener(hearsay_tape: Tape) -> None:
    chatter = hearsay_tape.chatter
    keys = list(zip(chatter.at.tolist(), chatter.to.tolist(), chatter.kind.tolist(), strict=True))
    assert keys == sorted(keys)


def test_no_talk_rate_means_no_chatter(toy: ToyCity) -> None:
    tape = run(toy, day_one_burst(RESIDENTS), Params(q=0.0, occasions_per_day=0.0))
    assert tape.header.chatter_rows == 0
    assert (tape.residents.first_source != SOURCE_WOM).all()


def test_tie_kinds_with_zero_talk_weight_carry_nothing(toy: ToyCity) -> None:
    neighbours_only = Params(
        q=0.05, hearsay_factor=1.0, occasions_per_day=0.0, talk_kind=(1.0, 1.0, 0.0, 1.0)
    )
    chatter = run(toy, day_one_burst(RESIDENTS), neighbours_only).chatter
    offsets, targets, kind = toy.ties.offsets, toy.ties.targets, toy.ties.kind
    assert chatter.at.shape[0] > 50
    for talker, to in zip(chatter.from_.tolist(), chatter.to.tolist(), strict=True):
        ties = slice(offsets[talker], offsets[talker + 1])
        assert (kind[ties][targets[ties] == to] == 1).any()


def test_uniform_mixing_spreads_without_ties() -> None:
    city = build_uniform_city(4_000)
    params = Params(
        q=0.05, hearsay_factor=1.0, occasions_per_day=0.0, wom_mixing="uniform", talk_gain=1.0
    )
    tape = run(city, day_one_burst(4_000, per_resident=0.3, days=4), params)
    assert (tape.residents.first_source == SOURCE_WOM).sum() > 200
    assert (tape.chatter.from_ != tape.chatter.to).all()
    graph = run(city, day_one_burst(4_000, per_resident=0.3, days=4), HEARSAY_ONLY)
    assert graph.header.chatter_rows == 0


def test_more_talk_never_means_fewer_people_heard(toy: ToyCity) -> None:
    quiet = run(toy, day_one_burst(RESIDENTS), Params(q=0.005, occasions_per_day=0.0))
    loud = run(toy, day_one_burst(RESIDENTS), Params(q=0.2, occasions_per_day=0.0))
    assert loud.aggregates.aware[-1] > quiet.aggregates.aware[-1]
