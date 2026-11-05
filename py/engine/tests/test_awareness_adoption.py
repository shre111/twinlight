import dataclasses
import re
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from twl_engine.fixtures import ToyCity
from twl_engine.goldens import prototype_small_inputs
from twl_engine.run import RunInputs, run_seed, run_seed_bytes
from twl_engine.scenario import EMPTY, Campaign, Params, Scenario, load_scenario
from twl_tape.format import (
    CHATTER_HEARD,
    CHATTER_NUDGED,
    NEVER,
    SOURCE_NONE,
    SOURCE_PAID,
    SOURCE_WOM,
    Tape,
)

PROTOTYPE = Path(__file__).resolve().parents[3] / "fixtures" / "scenarios" / "prototype_launch.toml"
DAY = 1440
TICKS_PER_DAY = 96
TICK = 15
NUDGE_WINDOW = 72 * 60
SEEDS = (1, 2)
MEDIAN_SEEDS = (1, 2, 3, 4, 5)


@lru_cache(maxsize=1)
def city() -> ToyCity:
    return prototype_small_inputs(1).city


def base_inputs(seed: int) -> RunInputs:
    return prototype_small_inputs(seed)


@cache
def tape_of(scenario: Scenario, params: Params, seed: int) -> Tape:
    inputs = dataclasses.replace(base_inputs(seed), scenario=scenario, params=params, seed=seed)
    return run_seed(inputs)


def launch() -> tuple[Scenario, Params]:
    inputs = base_inputs(1)
    return inputs.scenario, inputs.params


def campaign(**changes: object) -> Campaign:
    fields: dict[str, object] = {
        "name": "test",
        "start_minute": 0,
        "end_minute": 30 * DAY,
        "budget_per_day": 2_000.0,
        "cpm": 1_000.0,
    }
    fields.update(changes)
    return Campaign(**fields)  # type: ignore[arg-type]


def with_campaigns(*campaigns: Campaign, days: int = 30) -> Scenario:
    return Scenario(days=days, campaigns=campaigns)


def hot() -> tuple[Scenario, Params]:
    return with_campaigns(campaign(budget_per_day=5_000.0)), Params(alpha_0=-1.0, q=0.05)


def residents_equal(a: Tape, b: Tape) -> bool:
    return all(
        np.array_equal(getattr(a.residents, f.name), getattr(b.residents, f.name))
        for f in dataclasses.fields(a.residents)
    )


def aware_count(tape: Tape) -> int:
    return int(np.count_nonzero(tape.residents.aware_at != NEVER))


def adopted_count(tape: Tape) -> int:
    return int(np.count_nonzero(tape.residents.first_order_at != NEVER))


def test_empty_scenario_makes_nothing_happen() -> None:
    tape = tape_of(EMPTY, Params(), 1)
    assert aware_count(tape) == 0
    assert adopted_count(tape) == 0
    assert int(tape.aggregates.orders.sum()) == 0
    assert len(tape.chatter.at) == 0


def test_campaign_with_default_params_makes_residents_aware_and_order() -> None:
    tape = tape_of(with_campaigns(campaign(budget_per_day=5_000.0)), Params(), 1)
    assert aware_count(tape) > 0
    assert adopted_count(tape) > 0
    assert int(tape.aggregates.orders.sum()) >= adopted_count(tape)


@pytest.fixture(scope="module", params=["prototype_small", "hot"])
def invariant_tapes(request: pytest.FixtureRequest) -> list[Tape]:
    scenario, params = launch() if request.param == "prototype_small" else hot()
    return [tape_of(scenario, params, seed) for seed in SEEDS]


def test_resident_columns_are_consistent(invariant_tapes: list[Tape]) -> None:
    n = len(city().residents.x)
    for tape in invariant_tapes:
        r = tape.residents
        aware = r.aware_at != NEVER
        ordered = r.first_order_at != NEVER
        both = aware & ordered
        assert np.all(r.aware_at[both] <= r.first_order_at[both])
        assert np.array_equal(ordered, r.order_count >= 1)
        assert np.all(r.first_source[aware] != SOURCE_NONE)
        assert np.all(np.isin(r.first_source[aware], (SOURCE_PAID, SOURCE_WOM)))
        assert np.all(r.first_source[~aware] == SOURCE_NONE)
        wom = r.first_source == SOURCE_WOM
        assert np.array_equal(r.heard_from != NEVER, wom)
        sources = r.heard_from[wom].astype(np.int64)
        assert np.all(sources < n)
        assert np.all(sources != np.flatnonzero(wom))
        assert np.all(~ordered | aware)


def test_aggregates_are_monotone_and_match_the_resident_counts(
    invariant_tapes: list[Tape],
) -> None:
    for tape in invariant_tapes:
        a = tape.aggregates
        assert np.all(np.diff(a.aware.astype(np.int64)) >= 0)
        assert np.all(np.diff(a.adopted.astype(np.int64)) >= 0)
        assert int(a.aware[-1]) == aware_count(tape)
        assert int(a.adopted[-1]) == adopted_count(tape)
        assert int(a.orders.astype(np.int64).sum()) == int(
            tape.residents.order_count.astype(np.int64).sum()
        )


def test_chatter_has_one_heard_row_per_wom_resident_and_at_most_one_nudge(
    invariant_tapes: list[Tape],
) -> None:
    for tape in invariant_tapes:
        c, r = tape.chatter, tape.residents
        assert np.all(np.diff(c.at.astype(np.int64)) >= 0)
        assert np.all(np.isin(c.kind, (CHATTER_HEARD, CHATTER_NUDGED)))
        heard = c.kind == CHATTER_HEARD
        wom = np.flatnonzero(r.first_source == SOURCE_WOM)
        assert np.array_equal(np.sort(c.to[heard]), wom)
        assert np.array_equal(c.from_[heard], r.heard_from[c.to[heard]])
        assert np.array_equal(c.at[heard], r.aware_at[c.to[heard]])
        nudged = c.kind == CHATTER_NUDGED
        targets = c.to[nudged]
        assert len(np.unique(targets)) == len(targets)
        first = r.first_order_at[targets].astype(np.int64)
        assert np.all(first != NEVER)
        at = c.at[nudged].astype(np.int64)
        assert np.all(first - NUDGE_WINDOW <= at)
        assert np.all(at <= first)
        heard_rows = {
            int(t): (int(f), int(a))
            for t, f, a in zip(c.to[heard], c.from_[heard], c.at[heard], strict=True)
        }
        for t, f, a in zip(targets, c.from_[nudged], c.at[nudged], strict=True):
            assert heard_rows.get(int(t)) != (int(f), int(a))


def test_hot_scenario_records_both_chatter_kinds() -> None:
    scenario, params = hot()
    chatter = tape_of(scenario, params, 1).chatter
    assert np.count_nonzero(chatter.kind == CHATTER_HEARD) > 0
    assert np.count_nonzero(chatter.kind == CHATTER_NUDGED) > 0


def test_zero_talk_rate_gives_no_wom_sources_and_no_chatter() -> None:
    scenario, _ = hot()
    tape = tape_of(scenario, Params(alpha_0=-1.0, q=0.0), 1)
    assert aware_count(tape) > 0
    assert not np.any(tape.residents.first_source == SOURCE_WOM)
    assert np.all(tape.residents.heard_from == NEVER)
    assert len(tape.chatter.at) == 0


def test_no_campaigns_means_nothing_happens_even_with_talk() -> None:
    tape = tape_of(EMPTY, Params(q=0.5, alpha_0=0.0), 1)
    assert aware_count(tape) == 0
    assert adopted_count(tape) == 0
    assert len(tape.chatter.at) == 0


def test_zero_paid_scale_equals_no_campaigns() -> None:
    scenario, params = hot()
    silent = tape_of(scenario, dataclasses.replace(params, paid_scale=0.0), 1)
    none = tape_of(EMPTY, dataclasses.replace(params, paid_scale=0.0), 1)
    assert residents_equal(silent, none)
    assert aware_count(silent) == 0
    assert len(silent.chatter.at) == 0


def test_nobody_is_aware_before_the_campaign_starts() -> None:
    start, end = 10 * DAY, 12 * DAY
    scenario = with_campaigns(campaign(start_minute=start, end_minute=end, budget_per_day=10_000.0))
    tape = tape_of(scenario, Params(q=0.0, alpha_0=-1.0), 1)
    aware_at = tape.residents.aware_at[tape.residents.aware_at != NEVER]
    assert len(aware_at) > 0
    assert int(aware_at.min()) >= start
    assert int(aware_at.max()) <= end + TICK
    assert np.all(tape.aggregates.aware[: start // TICK - 1] == 0)


def test_district_targeting_confines_paid_exposure_to_that_district() -> None:
    c = city()
    district_of = c.buildings.district[c.residents.building]
    counts = np.bincount(district_of.astype(np.int64), minlength=len(c.districts))
    target = int(np.argmax(counts))
    weights = tuple(1.0 if i == target else 0.0 for i in range(len(c.districts)))
    scenario = with_campaigns(campaign(district_weights=weights, budget_per_day=10_000.0))
    tape = tape_of(scenario, Params(alpha_0=-1.0), 1)
    paid = tape.residents.first_source == SOURCE_PAID
    assert np.count_nonzero(paid) > 0
    assert np.all(district_of[paid] == target)


def test_segment_targeting_confines_paid_exposure_to_that_segment() -> None:
    scenario = with_campaigns(
        campaign(segment_weights=(0.0, 0.0, 1.0, 0.0, 0.0), budget_per_day=10_000.0)
    )
    tape = tape_of(scenario, Params(alpha_0=-1.0), 1)
    paid = tape.residents.first_source == SOURCE_PAID
    assert np.count_nonzero(paid) > 0
    assert np.all(city().residents.segment[paid] == 2)


def test_frequency_cap_stops_sub_threshold_impressions_from_adding_up() -> None:
    one_day = dict(end_minute=DAY, budget_per_day=50_000.0)
    params = Params(
        impression_gain=0.9,
        q=0.0,
        awareness_half_life_days=None,
        talk_decay_hours=None,
    )
    capped = tape_of(with_campaigns(campaign(frequency_cap=1, **one_day)), params, 1)
    assert aware_count(capped) == 0
    thrice = tape_of(with_campaigns(campaign(frequency_cap=3, **one_day)), params, 1)
    assert aware_count(thrice) > 0


def test_a_single_sufficient_impression_makes_residents_aware() -> None:
    params = Params(
        impression_gain=1.0, q=0.0, awareness_half_life_days=None, talk_decay_hours=None
    )
    scenario = with_campaigns(campaign(frequency_cap=1, end_minute=DAY, budget_per_day=50_000.0))
    assert aware_count(tape_of(scenario, params, 1)) > 0


def test_same_inputs_give_identical_bytes_and_different_seeds_differ() -> None:
    scenario, params = hot()
    a = run_seed_bytes(dataclasses.replace(base_inputs(1), scenario=scenario, params=params))
    b = run_seed_bytes(dataclasses.replace(base_inputs(1), scenario=scenario, params=params))
    assert a == b
    other = tape_of(scenario, params, 2)
    assert not residents_equal(tape_of(scenario, params, 1), other)


def test_load_scenario_reads_the_prototype_fixture() -> None:
    scenario, params = load_scenario(PROTOTYPE)
    assert scenario.days == 30
    assert sorted(c.name for c in scenario.campaigns) == ["creator", "drip", "launch"]
    defaults = dataclasses.asdict(Params())
    got = dataclasses.asdict(params)
    changed = {k for k in defaults if defaults[k] != got[k]}
    assert changed == {"paid_scale", "q", "alpha_0"}


def test_load_scenario_matches_what_the_goldens_run() -> None:
    scenario, params = load_scenario(PROTOTYPE)
    inputs = prototype_small_inputs(1)
    assert inputs.scenario == scenario
    assert inputs.params == params


def test_load_scenario_rejects_an_unknown_key_anywhere(tmp_path: Path) -> None:
    text = PROTOTYPE.read_text()
    lines = text.splitlines()
    variants = {"top": "bogus_key = 1\n" + text}
    for i, line in enumerate(lines):
        if re.match(r"\s*\[.*\]\s*$", line):
            patched = [*lines[: i + 1], "bogus_key = 1", *lines[i + 1 :]]
            variants[f"after line {i + 1}: {line.strip()}"] = "\n".join(patched) + "\n"
    for label, content in variants.items():
        path = tmp_path / "scenario.toml"
        path.write_text(content)
        try:
            load_scenario(path)
        except Exception:
            continue
        pytest.fail(f"unknown key accepted: {label}")


def median_of(scenario: Scenario, params: Params, pick: str) -> float:
    values = []
    for seed in MEDIAN_SEEDS:
        tape = tape_of(scenario, params, seed)
        values.append(
            int((tape.aggregates.adopted if pick == "adopted" else tape.aggregates.aware)[-1])
        )
    return float(np.median(values))


# Runs with different inputs do not share random draws (D-027), hence the 0.5% margin.
MARGIN = 0.005 * 5_000
SLOW = settings(
    max_examples=6,
    derandomize=True,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


@SLOW
@given(
    price=st.sampled_from([39.0, 49.0, 59.0]),
    factor=st.floats(min_value=1.15, max_value=2.0),
)
def test_raising_the_price_never_raises_median_adopters(price: float, factor: float) -> None:
    scenario, params = launch()
    low = dataclasses.replace(scenario, price=price, ref_price=49.0)
    high = dataclasses.replace(scenario, price=price * factor, ref_price=49.0)
    assert median_of(high, params, "adopted") <= median_of(low, params, "adopted") + MARGIN


@SLOW
@given(
    scale=st.sampled_from([1.0, 2.0]),
    factor=st.floats(min_value=1.25, max_value=3.0),
)
def test_raising_every_budget_never_lowers_median_awareness(scale: float, factor: float) -> None:
    scenario, params = launch()

    def scaled(k: float) -> Scenario:
        campaigns = tuple(
            dataclasses.replace(c, budget_per_day=c.budget_per_day * k) for c in scenario.campaigns
        )
        return dataclasses.replace(scenario, campaigns=campaigns)

    low = median_of(scaled(scale), params, "aware")
    high = median_of(scaled(scale * factor), params, "aware")
    assert high >= low - MARGIN
