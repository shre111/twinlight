import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twl_citybake.population.ipf import integerise, ipf

Floats = npt.NDArray[np.float64]


def rel_error(matrix: Floats, rows: Floats, cols: Floats) -> float:
    row_err = np.abs(matrix.sum(1) - rows) / np.maximum(rows, 1.0)
    col_err = np.abs(matrix.sum(0) - cols) / np.maximum(cols, 1.0)
    return float(max(row_err.max(), col_err.max()))


@st.composite
def fittable(draw: st.DrawFn, zeros: bool = False) -> tuple[Floats, Floats, Floats]:
    n, k = draw(st.integers(2, 6)), draw(st.integers(2, 6))
    seed_values = draw(st.lists(st.floats(0.1, 10.0), min_size=n * k, max_size=n * k))
    truth_values = draw(st.lists(st.floats(0.1, 100.0), min_size=n * k, max_size=n * k))
    seed = np.asarray(seed_values).reshape(n, k)
    truth = np.asarray(truth_values).reshape(n, k)
    if zeros:
        r, c = draw(st.integers(0, n - 1)), draw(st.integers(0, k - 1))
        seed[r, c] = 0.0
        truth[r, c] = 0.0
    return seed, truth.sum(1), truth.sum(0)


@settings(deadline=None, max_examples=100)
@given(fittable())
def test_positive_seeds_converge_to_row_and_column_targets(
    case: tuple[Floats, Floats, Floats],
) -> None:
    seed, rows, cols = case
    result = ipf(seed, rows, cols, tol=1e-6, max_iter=5000)
    assert result.converged
    assert result.max_rel_error <= 1e-6
    assert result.matrix.sum(1) == pytest.approx(rows, rel=1e-5)
    assert result.matrix.sum(0) == pytest.approx(cols, rel=1e-5)
    assert (result.matrix > 0).all()


@settings(deadline=None, max_examples=100)
@given(fittable(zeros=True))
def test_structural_zeros_stay_zero(case: tuple[Floats, Floats, Floats]) -> None:
    seed, rows, cols = case
    result = ipf(seed, rows, cols, tol=1e-6, max_iter=5000)
    assert (result.matrix[seed == 0] == 0).all()
    assert (result.matrix >= 0).all()


@settings(deadline=None, max_examples=100)
@given(fittable())
def test_reported_error_matches_the_definition(case: tuple[Floats, Floats, Floats]) -> None:
    seed, rows, cols = case
    result = ipf(seed, rows, cols, tol=1e-3, max_iter=50)
    assert result.max_rel_error == pytest.approx(rel_error(result.matrix, rows, cols), abs=1e-12)


def test_zero_targets_zero_their_row_and_column() -> None:
    seed = np.ones((3, 3))
    result = ipf(seed, np.array([0.0, 10.0, 20.0]), np.array([30.0, 0.0, 0.0]))
    assert result.converged
    assert (result.matrix[0] == 0).all()
    assert (result.matrix[:, 1:] == 0).all()
    assert result.matrix[:, 0].tolist() == pytest.approx([0.0, 10.0, 20.0])


def test_a_seed_that_already_fits_is_returned_unchanged() -> None:
    seed = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = ipf(seed, seed.sum(1), seed.sum(0))
    assert result.converged
    assert result.iterations <= 3
    assert result.matrix == pytest.approx(seed)


def test_known_two_by_two_solution() -> None:
    result = ipf(np.ones((2, 2)), np.array([30.0, 70.0]), np.array([40.0, 60.0]))
    assert result.matrix == pytest.approx(np.array([[12.0, 18.0], [28.0, 42.0]]))


def test_not_converging_within_max_iter_is_reported() -> None:
    seed = np.array([[1.0, 1e-9], [1.0, 1.0]])
    result = ipf(seed, np.array([1.0, 1.0]), np.array([1.9, 0.1]), tol=1e-15, max_iter=2)
    assert not result.converged
    assert result.iterations == 2


def test_mismatched_target_totals_raise() -> None:
    with pytest.raises(ValueError):
        ipf(np.ones((2, 2)), np.array([10.0, 10.0]), np.array([10.0, 11.0]))


def test_positive_row_target_with_an_all_zero_seed_row_raises() -> None:
    seed = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        ipf(seed, np.array([5.0, 5.0]), np.array([5.0, 5.0]))


def test_positive_column_target_with_an_all_zero_seed_column_raises() -> None:
    seed = np.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        ipf(seed, np.array([5.0, 5.0]), np.array([5.0, 5.0]))


def test_all_zero_seed_line_with_zero_target_is_feasible() -> None:
    seed = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = ipf(seed, np.array([0.0, 10.0]), np.array([4.0, 6.0]))
    assert result.converged
    assert result.matrix[1].tolist() == pytest.approx([4.0, 6.0])


@st.composite
def integerise_cases(
    draw: st.DrawFn,
) -> tuple[Floats, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    n, k = draw(st.integers(1, 8)), draw(st.integers(1, 4))
    sizes = np.asarray(sorted(draw(st.sets(st.integers(1, 17), min_size=k, max_size=k))))
    expected = np.asarray(
        draw(st.lists(st.floats(0.0, 6.0), min_size=n * k, max_size=n * k))
    ).reshape(n, k)
    floor_persons = (np.floor(expected) * sizes).sum(1)
    slack = np.asarray(draw(st.lists(st.integers(0, 12), min_size=n, max_size=n)))
    capacity = (floor_persons + slack).astype(np.int64)
    col_totals = np.round(expected.sum(0)).astype(np.int64)
    return expected, sizes.astype(np.int64), capacity, col_totals


@settings(deadline=None, max_examples=200)
@given(integerise_cases())
def test_integerise_never_exceeds_capacity_or_column_totals(
    case: tuple[Floats, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> None:
    expected, sizes, capacity, col_totals = case
    x = integerise(expected, sizes, capacity, col_totals)
    assert x.dtype == np.int64
    assert x.shape == expected.shape
    assert (x >= np.floor(expected)).all()
    assert ((x * sizes).sum(1) <= capacity).all()
    assert (x.sum(0) <= col_totals).all()


@settings(deadline=None, max_examples=200)
@given(integerise_cases())
def test_integerise_reaches_column_totals_when_capacity_allows(
    case: tuple[Floats, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> None:
    expected, sizes, _, col_totals = case
    roomy = np.full(len(expected), 10**6, np.int64)
    x = integerise(expected, sizes, roomy, col_totals)
    assert (x.sum(0) == col_totals).all()


@settings(deadline=None, max_examples=100)
@given(integerise_cases())
def test_integerise_is_deterministic(
    case: tuple[Floats, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> None:
    first = integerise(*case)
    second = integerise(*case)
    assert np.array_equal(first, second)


def test_integerise_gives_the_extra_household_to_the_largest_remainder() -> None:
    expected = np.array([[0.2], [0.7], [0.6]])
    sizes = np.array([2])
    big = np.array([100, 100, 100])
    assert integerise(expected, sizes, big, np.array([1])).ravel().tolist() == [0, 1, 0]
    assert integerise(expected, sizes, big, np.array([2])).ravel().tolist() == [0, 1, 1]


def test_integerise_breaks_remainder_ties_by_lower_row_index() -> None:
    expected = np.array([[0.5], [0.5], [0.5]])
    got = integerise(expected, np.array([1]), np.array([9, 9, 9]), np.array([2]))
    assert got.ravel().tolist() == [1, 1, 0]


def test_integerise_skips_buildings_without_room_for_another_household() -> None:
    expected = np.array([[0.9], [0.8]])
    got = integerise(expected, np.array([3]), np.array([2, 5]), np.array([1]))
    assert got.ravel().tolist() == [0, 1]


def test_integerise_fills_larger_sizes_before_smaller_ones() -> None:
    expected = np.array([[0.6, 0.6]])
    got = integerise(expected, np.array([1, 4]), np.array([4]), np.array([1, 1]))
    assert got.tolist() == [[0, 1]]


def test_integerise_stops_short_when_nobody_has_room() -> None:
    expected = np.array([[0.5], [0.5]])
    got = integerise(expected, np.array([4]), np.array([3, 3]), np.array([2]))
    assert got.sum() == 0


def test_integerise_never_adds_a_household_to_a_structural_zero() -> None:
    expected = np.array([[0.0], [0.4]])
    got = integerise(expected, np.array([2]), np.array([50, 50]), np.array([3]))
    assert got.ravel().tolist() == [0, 3]


def test_integerise_leaves_column_short_when_only_structural_zeros_remain() -> None:
    got = integerise(np.zeros((3, 1)), np.array([1]), np.array([9, 9, 9]), np.array([2]))
    assert got.sum() == 0


def test_integerise_removes_overshoot_one_per_row_smallest_remainder_first() -> None:
    expected = np.array([[2.1], [1.9], [1.5]])
    got = integerise(expected, np.array([1]), np.array([99, 99, 99]), np.array([3]))
    assert got.ravel().tolist() == [1, 1, 1]
    got = integerise(expected, np.array([1]), np.array([99, 99, 99]), np.array([4]))
    assert got.ravel().tolist() == [2, 1, 1]


def test_integerise_overshoot_ties_remove_from_the_higher_row_first() -> None:
    expected = np.array([[1.5], [1.5], [1.5]])
    got = integerise(expected, np.array([1]), np.array([99, 99, 99]), np.array([2]))
    assert got.ravel().tolist() == [1, 1, 0]


def test_integerise_removes_at_most_one_per_row_per_pass() -> None:
    expected = np.array([[3.0], [1.0]])
    got = integerise(expected, np.array([1]), np.array([99, 99]), np.array([2]))
    assert got.ravel().tolist() == [2, 0]
    got = integerise(expected, np.array([1]), np.array([99, 99]), np.array([1]))
    assert got.ravel().tolist() == [1, 0]


@st.composite
def zeroed_cases(
    draw: st.DrawFn,
) -> tuple[Floats, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    expected, sizes, capacity, _ = draw(integerise_cases())
    mask = np.asarray(
        draw(st.lists(st.booleans(), min_size=expected.size, max_size=expected.size))
    ).reshape(expected.shape)
    expected = np.where(mask, 0.0, expected)
    totals = np.asarray(
        draw(st.lists(st.integers(0, 30), min_size=expected.shape[1], max_size=expected.shape[1])),
        np.int64,
    )
    return expected, sizes, capacity, totals


@settings(deadline=None, max_examples=300)
@given(zeroed_cases())
def test_integerise_has_no_positive_count_at_zero_expectation_and_respects_totals(
    case: tuple[Floats, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> None:
    expected, sizes, capacity, col_totals = case
    x = integerise(expected, sizes, capacity, col_totals)
    assert (x[expected == 0] == 0).all()
    assert (x.sum(0) <= col_totals).all()
    assert ((x * sizes).sum(1) <= capacity).all()
