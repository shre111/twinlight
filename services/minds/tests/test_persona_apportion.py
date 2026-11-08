from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

sampling = pytest.importorskip("twl_minds.personas.sampling")
_apportion: Callable[..., Any] | None = getattr(sampling, "apportion", None)
if _apportion is None:
    pytest.skip("apportion is not public", allow_module_level=True)


def apportion(total: int, weights: list[float], caps: list[int]) -> list[int]:
    assert _apportion is not None
    return [int(x) for x in _apportion(total, weights, caps)]


def test_equal_weights_tie_goes_to_lowest_index() -> None:
    assert apportion(10, [1, 1, 1], [10, 10, 10]) == [4, 3, 3]


def test_two_leftover_units_go_to_two_lowest_tied_indices() -> None:
    assert apportion(2, [1, 1, 1], [5, 5, 5]) == [1, 1, 0]


def test_largest_fractional_part_wins_leftover() -> None:
    assert apportion(10, [1, 2], [10, 10]) == [3, 7]


def test_exact_shares_need_no_leftover() -> None:
    assert apportion(24, [100, 200], [100, 100]) == [8, 16]


def test_capped_entry_closes_and_rest_goes_to_open_entries() -> None:
    assert apportion(10, [1, 1], [2, 10]) == [2, 8]


def test_redistribution_repeats_over_open_entries_by_weight() -> None:
    assert apportion(12, [1, 1, 2], [1, 100, 100]) == [1, 4, 7]


def test_total_beyond_all_caps_fills_caps_and_stops() -> None:
    assert apportion(10, [1, 1], [2, 3]) == [2, 3]


def test_zero_total_gives_zeros() -> None:
    assert apportion(0, [1, 2], [5, 5]) == [0, 0]


def test_zero_cap_entry_gets_nothing() -> None:
    assert apportion(5, [1, 1], [0, 9]) == [0, 5]


@given(
    total=st.integers(0, 200),
    entries=st.lists(st.tuples(st.integers(1, 50), st.integers(0, 60)), min_size=1, max_size=8),
)
def test_result_respects_caps_and_sums_to_min_of_total_and_capacity(
    total: int, entries: list[tuple[int, int]]
) -> None:
    weights = [float(w) for w, _ in entries]
    caps = [c for _, c in entries]
    result = apportion(total, weights, caps)
    assert all(0 <= r <= c for r, c in zip(result, caps, strict=True))
    assert sum(result) == min(total, sum(caps))
