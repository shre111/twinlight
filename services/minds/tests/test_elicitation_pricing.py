import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from twl_minds.elicitation import pricing
from twl_minds.elicitation.pricing import (
    BATCH_FACTOR,
    PRICES,
    Rates,
    UnknownModel,
    Usage,
    cost_usd,
    upper_bound_usd,
)

MODEL = "claude-haiku-5-5"
BASE = Rates(0.10, 0.125, 0.01, 0.50)
LONG = Rates(0.50, 0.625, 0.05, 2.50)


def _usage(i: int = 0, cc: int = 0, cr: int = 0, o: int = 0) -> Usage:
    return Usage(
        input_tokens=i,
        cache_creation_input_tokens=cc,
        cache_read_input_tokens=cr,
        output_tokens=o,
    )


def test_price_table_holds_the_haiku_rates_and_threshold() -> None:
    price = PRICES[MODEL]
    assert price.base == BASE
    assert price.long == LONG
    assert price.long_threshold == 100_000
    assert BATCH_FACTOR == 0.5


def test_price_source_and_date_are_recorded_in_the_module() -> None:
    constants = [v for k, v in vars(pricing).items() if k.isupper() and isinstance(v, str)]
    assert any("platform.claude.com" in c for c in constants)
    assert any("2026-11-11" in c for c in constants)


def test_worked_cost_with_every_category() -> None:
    usage = _usage(1000, 2000, 10_000, 500)
    assert cost_usd(MODEL, usage, False) == pytest.approx(700 / 1e6)


def test_cache_read_only_prompt_is_priced_at_the_cache_read_rate() -> None:
    assert cost_usd(MODEL, _usage(cr=50_000), False) == pytest.approx(500 / 1e6)


def test_output_only_cost() -> None:
    assert cost_usd(MODEL, _usage(o=1000), False) == pytest.approx(500 / 1e6)


def test_zero_usage_costs_nothing() -> None:
    assert cost_usd(MODEL, _usage(), False) == 0.0


def test_batch_halves_every_category() -> None:
    usage = _usage(1000, 2000, 10_000, 500)
    assert cost_usd(MODEL, usage, True) == pytest.approx(350 / 1e6)


def test_prompt_of_exactly_100000_tokens_uses_base_rates() -> None:
    assert cost_usd(MODEL, _usage(i=100_000, o=1000), False) == pytest.approx(10_500 / 1e6)


def test_prompt_of_100001_tokens_uses_long_rates_for_every_category() -> None:
    assert cost_usd(MODEL, _usage(i=100_001, o=1000), False) == pytest.approx(52_500.5 / 1e6)


def test_prompt_length_sums_input_cache_write_and_cache_read_for_the_step() -> None:
    usage = _usage(i=1, cr=100_000)
    assert cost_usd(MODEL, usage, False) == pytest.approx((0.5 + 100_000 * 0.05) / 1e6)
    at_threshold = _usage(i=0, cr=100_000)
    assert cost_usd(MODEL, at_threshold, False) == pytest.approx(1000 / 1e6)


def test_long_step_stacks_with_the_batch_factor() -> None:
    assert cost_usd(MODEL, _usage(i=100_001, o=1000), True) == pytest.approx(26_250.25 / 1e6)


def test_upper_bound_prices_each_prompt_as_a_cache_write_with_full_max_tokens() -> None:
    expected = (1000 * 0.125 + 500 * 0.5) / 1e6
    assert upper_bound_usd(MODEL, [1000], 500, False) == pytest.approx(expected)


def test_upper_bound_sums_requests_at_the_rates_each_prompt_selects() -> None:
    expected = (1000 * 0.125 + 500 * 0.5 + 100_001 * 0.625 + 500 * 2.5) / 1e6
    assert upper_bound_usd(MODEL, [1000, 100_001], 500, False) == pytest.approx(expected)


def test_upper_bound_at_exactly_100000_uses_base_rates() -> None:
    expected = (100_000 * 0.125 + 500 * 0.5) / 1e6
    assert upper_bound_usd(MODEL, [100_000], 500, False) == pytest.approx(expected)


def test_upper_bound_batch_halves() -> None:
    full = upper_bound_usd(MODEL, [1000, 100_001], 500, False)
    assert upper_bound_usd(MODEL, [1000, 100_001], 500, True) == pytest.approx(full / 2)


def test_upper_bound_of_no_requests_is_zero() -> None:
    assert upper_bound_usd(MODEL, [], 500, False) == 0.0


@given(
    prompt=st.integers(min_value=0, max_value=300_000),
    max_tokens=st.integers(min_value=1, max_value=16_000),
    split=st.tuples(st.floats(0, 1), st.floats(0, 1)),
    out_share=st.floats(0, 1),
    batch=st.booleans(),
)
def test_upper_bound_is_at_least_the_cost_of_any_usage_within_the_bounds(
    prompt: int, max_tokens: int, split: tuple[float, float], out_share: float, batch: bool
) -> None:
    a, b = sorted(split)
    cc = int(prompt * a)
    cr = int(prompt * (b - a))
    usage = _usage(i=prompt - cc - cr, cc=cc, cr=cr, o=int(max_tokens * out_share))
    assert upper_bound_usd(MODEL, [prompt], max_tokens, batch) >= cost_usd(MODEL, usage, batch)


@given(
    prompt=st.integers(min_value=0, max_value=300_000),
    smaller=st.integers(min_value=0, max_value=300_000),
    batch=st.booleans(),
)
def test_upper_bound_covers_usage_with_a_shorter_prompt_than_the_bound(
    prompt: int, smaller: int, batch: bool
) -> None:
    used = min(prompt, smaller)
    usage = _usage(cc=used, o=100)
    assert upper_bound_usd(MODEL, [prompt], 100, batch) >= cost_usd(MODEL, usage, batch)


def test_unknown_model_is_refused_by_every_function() -> None:
    with pytest.raises(UnknownModel, match="claude-nope"):
        cost_usd("claude-nope", _usage(1), False)
    with pytest.raises(UnknownModel, match="claude-nope"):
        upper_bound_usd("claude-nope", [10], 10, False)
    assert issubclass(UnknownModel, KeyError)


@pytest.mark.parametrize("field", ["input_tokens", "cache_creation_input_tokens", "output_tokens"])
def test_usage_refuses_negative_counts(field: str) -> None:
    values = {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
        field: -1,
    }
    with pytest.raises(ValueError):
        Usage(**values)


def test_usage_and_rates_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        BASE.input = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _usage().output_tokens = 1  # type: ignore[misc]
