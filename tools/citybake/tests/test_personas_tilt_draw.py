import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from twl_citybake.personas.assign import draw_free_rows


def normalised(raw: list[int]) -> NDArray[np.float64]:
    w = np.array(raw, np.float64)
    return w / w.sum()


@settings(deadline=None)
@given(
    st.lists(st.integers(0, 50), min_size=2, max_size=40).filter(lambda w: sum(w) > 0),
    st.integers(1, 400),
    st.integers(0, 2**32 - 1),
    st.sampled_from(["zeros", "mixed"]),
)
def test_one_dimensional_weights_draw_exactly_what_rng_choice_draws(
    raw: list[int], count: int, seed: int, tiers: str
) -> None:
    weights = normalised(raw)
    slot_tier = (
        np.zeros(count, np.uint8) if tiers == "zeros" else (np.arange(count) % 3).astype(np.uint8)
    )
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    got = draw_free_rows(weights, slot_tier, a)
    expected = b.choice(len(weights), size=count, p=weights)
    np.testing.assert_array_equal(got, expected)
    assert a.bit_generator.state == b.bit_generator.state
    assert a.random() == b.random()


def test_three_equal_tier_rows_draw_exactly_what_rng_choice_draws() -> None:
    weights = normalised([5, 1, 0, 9, 2, 7])
    slot_tier = (np.arange(500) % 3).astype(np.uint8)
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    got = draw_free_rows(np.tile(weights, (3, 1)), slot_tier, a)
    np.testing.assert_array_equal(got, b.choice(6, size=500, p=weights))
    assert a.bit_generator.state == b.bit_generator.state


def test_draws_are_int64_row_indices_with_one_per_slot() -> None:
    weights = normalised([1, 2, 3])
    slot_tier = np.zeros(77, np.uint8)
    rows = draw_free_rows(weights, slot_tier, np.random.default_rng(1))
    assert rows.dtype == np.int64 and rows.shape == (77,)
    assert rows.min() >= 0 and rows.max() < 3


def test_each_tiers_slots_follow_that_tiers_weights() -> None:
    per_tier = np.array(
        [[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25], [0.05, 0.05, 0.2, 0.7]], np.float64
    )
    slot_tier = np.repeat(np.arange(3, dtype=np.uint8), 20_000)
    np.random.default_rng(2).shuffle(slot_tier)
    rows = draw_free_rows(per_tier, slot_tier, np.random.default_rng(5))
    for tier in range(3):
        freq = np.bincount(rows[slot_tier == tier], minlength=4) / 20_000
        np.testing.assert_allclose(freq, per_tier[tier], atol=0.015)


def test_a_zero_weight_row_is_never_drawn_in_its_tier() -> None:
    per_tier = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    slot_tier = (np.arange(3000) % 3).astype(np.uint8)
    rows = draw_free_rows(per_tier, slot_tier, np.random.default_rng(8))
    np.testing.assert_array_equal(rows, slot_tier.astype(np.int64))


def test_the_same_seed_gives_the_same_tiered_draw() -> None:
    per_tier = np.array([[0.6, 0.4], [0.5, 0.5], [0.1, 0.9]])
    slot_tier = (np.arange(1000) % 3).astype(np.uint8)
    a = draw_free_rows(per_tier, slot_tier, np.random.default_rng(4))
    b = draw_free_rows(per_tier, slot_tier, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)
