import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from twl_citybake.footprints import correction
from twl_citybake.footprints.correction import (
    HeightCorrection,
    fit_correction,
    validate_by_district,
)

BANDS = ["1-2", "3-4", "5-7", "8-12", "13+"]
LEVEL_M = 3.2

HAND_RAW = np.array([0.2, 0.4, 1.5, 1.7, 1.9, 3.1])
HAND_TARGET = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 4.0])
HAND_CENTRES = [0.5, 1.5, 3.5]
HAND_FITTED = [2.0, 6.25, 6.25]

Panel = tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
]


def hand_fit() -> HeightCorrection:
    return fit_correction(HAND_RAW, HAND_TARGET)


def test_fit_bins_by_floor_and_takes_the_bin_median_at_the_bin_centre() -> None:
    fit = fit_correction([0.2, 0.4, 1.5, 3.1], [1.0, 3.0, 5.0, 8.0])
    np.testing.assert_allclose(fit.centres, [0.5, 1.5, 3.5])
    np.testing.assert_allclose(fit.fitted, [2.0, 5.0, 8.0])


def test_fit_pools_a_monotonicity_violation_by_count_weight() -> None:
    fit = hand_fit()
    np.testing.assert_allclose(fit.centres, HAND_CENTRES)
    np.testing.assert_allclose(fit.fitted, HAND_FITTED)


def test_a_heavy_bin_dominates_the_pooled_value() -> None:
    fit = fit_correction([1.5] * 9 + [3.5], [10.0] * 9 + [2.0])
    np.testing.assert_allclose(fit.fitted, [9.2, 9.2])


def test_fit_is_non_decreasing_even_for_a_decreasing_sample() -> None:
    fit = fit_correction([0.5, 1.5, 2.5, 3.5], [9.0, 7.0, 5.0, 3.0])
    assert (np.diff(fit.fitted) >= 0).all()
    np.testing.assert_allclose(fit.fitted, [6.0] * 4)


def test_fit_honours_the_bin_width() -> None:
    fit = fit_correction([0.5, 1.5, 2.5, 3.5], [1.0, 2.0, 3.0, 4.0], bin_m=2.0)
    np.testing.assert_allclose(fit.centres, [1.0, 3.0])
    np.testing.assert_allclose(fit.fitted, [1.5, 3.5])


def test_fit_drops_nan_raw_values() -> None:
    raw = np.append(HAND_RAW, [np.nan, np.nan])
    target = np.append(HAND_TARGET, [50.0, 60.0])
    fit = fit_correction(raw, target)
    np.testing.assert_allclose(fit.centres, HAND_CENTRES)
    np.testing.assert_allclose(fit.fitted, HAND_FITTED)


@pytest.mark.parametrize(
    ("raw", "target"),
    [
        ([1.2, 1.4, 1.9], [3.0, 4.0, 5.0]),
        ([np.nan, np.nan, 2.5], [1.0, 2.0, 3.0]),
        ([], []),
    ],
)
def test_fit_raises_with_fewer_than_two_non_empty_bins(
    raw: list[float], target: list[float]
) -> None:
    with pytest.raises(ValueError):
        fit_correction(raw, target)


def test_apply_equals_fitted_at_the_centres() -> None:
    fit = hand_fit()
    np.testing.assert_allclose(fit.apply(fit.centres), fit.fitted)


def test_apply_interpolates_linearly_between_centres() -> None:
    fit = hand_fit()
    np.testing.assert_allclose(fit.apply([1.0, 1.5, 2.0]), [4.125, 6.25, 6.25])
    two = fit_correction([0.5, 2.5], [1.0, 5.0])
    np.testing.assert_allclose(two.apply([1.0, 1.5, 2.0]), [2.0, 3.0, 4.0])


def test_apply_keeps_the_top_bin_ratio_above_the_top_centre() -> None:
    fit = hand_fit()
    ratio = HAND_FITTED[-1] / HAND_CENTRES[-1]
    np.testing.assert_allclose(fit.apply([7.0, 20.0]), np.array([7.0, 20.0]) * ratio)


def test_apply_keeps_the_bottom_bin_ratio_below_the_bottom_centre() -> None:
    fit = hand_fit()
    ratio = HAND_FITTED[0] / HAND_CENTRES[0]
    np.testing.assert_allclose(fit.apply([0.0, 0.25]), np.array([0.0, 0.25]) * ratio)


def test_apply_is_continuous_at_both_ends() -> None:
    fit = hand_fit()
    eps = 1e-9
    for centre in (fit.centres[0], fit.centres[-1]):
        below, at, above = fit.apply([centre - eps, centre, centre + eps])
        assert below == pytest.approx(at, abs=1e-6)
        assert above == pytest.approx(at, abs=1e-6)


def test_apply_passes_nan_through_and_accepts_scalars_and_lists() -> None:
    fit = hand_fit()
    out = fit.apply([np.nan, 1.5, np.nan])
    assert np.isnan(out[[0, 2]]).all()
    assert out[1] == pytest.approx(6.25)
    assert float(fit.apply(1.5)) == pytest.approx(6.25)


def test_knots_are_lists_rounded_to_three_decimals() -> None:
    fit = fit_correction([0.1234, 1.5555, 2.9], [1.00049, 2.00051, 3.123456])
    knots = fit.knots()
    assert set(knots) == {"centres", "fitted"}
    assert knots["centres"] == [0.5, 1.5, 2.5]
    assert knots["fitted"] == [1.0, 2.001, 3.123]


@settings(deadline=None, max_examples=100)
@given(
    sample=st.lists(
        st.tuples(
            st.floats(0.0, 80.0, allow_nan=False),
            st.floats(1.0, 100.0, allow_nan=False),
        ),
        min_size=2,
        max_size=60,
    ),
    queries=st.lists(st.floats(0.0, 200.0, allow_nan=False), min_size=2, max_size=40),
)
def test_apply_is_monotone_non_decreasing(
    sample: list[tuple[float, float]], queries: list[float]
) -> None:
    raw = np.array([r for r, _ in sample])
    target = np.array([t for _, t in sample])
    assume(len(np.unique(np.floor(raw))) >= 2)
    fit = fit_correction(raw, target)
    out = fit.apply(np.sort(queries))
    assert (np.diff(fit.fitted) >= -1e-12).all()
    assert (np.diff(out) >= -1e-9).all()


def panel(n_districts: int = 10, top: int = 20, bias: float = 0.7) -> Panel:
    levels = np.tile(np.arange(1, top + 1), n_districts)
    district = np.repeat(np.arange(n_districts), top)
    target = levels * LEVEL_M
    return target * bias, target, district, levels


def test_validation_holds_out_whole_districts_by_id_modulo_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, target, district, levels = panel(n_districts=7, top=14)
    raw = raw + np.arange(len(raw)) * 1e-4
    trained: list[set[int]] = []
    real = correction.fit_correction

    def spy(r: npt.ArrayLike, t: npt.ArrayLike, bin_m: float = 1.0) -> HeightCorrection:
        trained.append(set(np.flatnonzero(np.isin(raw, np.asarray(r))).tolist()))
        return real(r, t, bin_m)

    monkeypatch.setattr(correction, "fit_correction", spy)
    result = validate_by_district(raw, target, district, levels, folds=3)

    assert result["folds"] == 3
    assert len(trained) == 3
    everything = set(range(len(raw)))
    held_out = {frozenset(everything - rows) for rows in trained}
    expected = {frozenset(np.flatnonzero(district % 3 == f).tolist()) for f in range(3)}
    assert held_out == expected


def test_validation_assigns_storey_bands_at_the_edges() -> None:
    edge_levels = [1, 2, 3, 4, 5, 7, 8, 12, 13, 20]
    n_districts = 10
    levels = np.tile(edge_levels, n_districts)
    district = np.repeat(np.arange(n_districts), len(edge_levels))
    target = levels * LEVEL_M
    result = validate_by_district(target * 0.9, target, district, levels)
    counts = {band: result["bands"][band]["n"] for band in BANDS}
    assert counts == {band: 2 * n_districts for band in BANDS}
    assert result["overall"]["n"] == len(levels)


def test_validation_corrects_a_known_multiplicative_bias_in_tall_bands() -> None:
    raw, target, district, levels = panel(bias=0.7)
    result = validate_by_district(raw, target, district, levels)

    assert result["folds"] == 5
    assert list(result["bands"]) == BANDS
    for band in ("5-7", "8-12", "13+"):
        stats = result["bands"][band]
        assert stats["before"] == pytest.approx(0.3, abs=0.01), band
        assert stats["after"] < 0.1, band
        assert stats["after"] < stats["before"] / 3, band
    overall = result["overall"]
    assert overall["n"] == len(raw)
    assert overall["after"] < overall["before"]


def test_validation_reports_rounded_fractions_and_is_deterministic() -> None:
    raw, target, district, levels = panel(bias=0.83)
    first = validate_by_district(raw, target, district, levels)
    second = validate_by_district(raw, target, district, levels)
    assert first == second
    for stats in [*first["bands"].values(), first["overall"]]:
        for key in ("before", "after"):
            assert 0.0 <= stats[key] <= 1.0
            assert stats[key] == round(stats[key], 3)
