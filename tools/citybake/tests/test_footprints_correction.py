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
    return fit_correction(HAND_RAW, HAND_TARGET, min_top_count=1)


def rows_in_bins(counts: list[int], offset: float = 0.25) -> npt.NDArray[np.float64]:
    return np.concatenate([np.full(c, i + offset) for i, c in enumerate(counts)])


def test_fit_bins_by_floor_and_takes_the_bin_mean_at_the_bin_centre() -> None:
    fit = fit_correction([0.2, 0.4, 1.5, 3.1], [1.0, 3.0, 5.0, 8.0], min_top_count=1)
    np.testing.assert_allclose(fit.centres, [0.5, 1.5, 3.5])
    np.testing.assert_allclose(fit.fitted, [2.0, 5.0, 8.0])


def test_the_bin_target_is_the_mean_not_the_median_of_discrete_storeys() -> None:
    raw = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
    target = [3.2, 3.2, 9.6, 12.8, 12.8, 12.8]
    fit = fit_correction(raw, target, min_top_count=1)
    np.testing.assert_allclose(fit.fitted, [(3.2 + 3.2 + 9.6) / 3, 12.8])


def test_fit_pools_a_monotonicity_violation_by_count_weight() -> None:
    fit = hand_fit()
    np.testing.assert_allclose(fit.centres, HAND_CENTRES)
    np.testing.assert_allclose(fit.fitted, HAND_FITTED)


def test_a_heavy_bin_dominates_the_pooled_value() -> None:
    fit = fit_correction([1.5] * 9 + [3.5], [10.0] * 9 + [2.0], min_top_count=1)
    np.testing.assert_allclose(fit.fitted, [9.2, 9.2])


def test_fit_is_non_decreasing_even_for_a_decreasing_sample() -> None:
    fit = fit_correction([0.5, 1.5, 2.5, 3.5], [9.0, 7.0, 5.0, 3.0], min_top_count=1)
    assert (np.diff(fit.fitted) >= 0).all()
    np.testing.assert_allclose(fit.fitted, [6.0] * 4)


def test_fit_honours_the_bin_width() -> None:
    fit = fit_correction([0.5, 1.5, 2.5, 3.5], [1.0, 2.0, 3.0, 4.0], bin_m=2.0, min_top_count=1)
    np.testing.assert_allclose(fit.centres, [1.0, 3.0])
    np.testing.assert_allclose(fit.fitted, [1.5, 3.5])


def test_fit_drops_nan_raw_values() -> None:
    raw = np.append(HAND_RAW, [np.nan, np.nan])
    target = np.append(HAND_TARGET, [50.0, 60.0])
    fit = fit_correction(raw, target, min_top_count=1)
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


def test_bins_that_already_hold_the_minimum_stay_alone_at_their_midpoint() -> None:
    raw = rows_in_bins([5, 5, 5, 5])
    target = np.array([3.0] * 5 + [6.0] * 5 + [9.0] * 5 + [12.0] * 5)
    fit = fit_correction(raw, target, min_top_count=5)
    np.testing.assert_allclose(fit.centres, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(fit.fitted, [3.0, 6.0, 9.0, 12.0])


def test_bins_pool_from_the_top_until_a_group_holds_the_minimum() -> None:
    counts = [2, 2, 3, 4, 5, 6]
    raw = rows_in_bins(counts)
    split = sum(counts[:4])
    raw[:split] += np.linspace(0.0, 0.5, split)
    raw[split:] += np.linspace(0.0, 0.5, len(raw) - split)
    target = np.concatenate([np.full(c, 3.0 * (i + 1)) for i, c in enumerate(counts)])
    fit = fit_correction(raw, target, min_top_count=10)
    assert len(fit.centres) == 2
    np.testing.assert_allclose(fit.centres, [raw[:split].mean(), raw[split:].mean()])
    np.testing.assert_allclose(fit.fitted, [target[:split].mean(), target[split:].mean()])


def test_an_under_filled_lowest_group_joins_the_group_above() -> None:
    raw = rows_in_bins([2, 10, 10])
    target = np.array([3.0] * 2 + [6.0] * 10 + [9.0] * 10)
    fit = fit_correction(raw, target, min_top_count=10)
    assert len(fit.centres) == 2
    np.testing.assert_allclose(fit.centres, [raw[:12].mean(), 2.5])
    np.testing.assert_allclose(fit.fitted, [target[:12].mean(), 9.0])


def test_a_sample_too_small_to_make_two_groups_stays_unpooled() -> None:
    fit = fit_correction([0.2, 1.7], [1.0, 3.0])
    np.testing.assert_allclose(fit.centres, [0.5, 1.5])
    np.testing.assert_allclose(fit.fitted, [1.0, 3.0])
    sparse = fit_correction(rows_in_bins([5, 5, 5, 5]), np.arange(20.0))
    np.testing.assert_allclose(sparse.centres, [0.5, 1.5, 2.5, 3.5])


def test_the_pooled_centre_is_the_mean_raw_height_of_its_rows() -> None:
    raw = np.array([0.1, 1.1, 1.9, 2.1, 2.3, 2.9, 3.2, 3.8])
    target = np.array([3.0, 3.2, 3.4, 6.0, 6.2, 6.4, 9.0, 9.4])
    fit = fit_correction(raw, target, min_top_count=3)
    assert len(fit.centres) == 2
    np.testing.assert_allclose(fit.centres, [raw[:3].mean(), raw[3:].mean()])


@settings(deadline=None, max_examples=60)
@given(
    counts=st.lists(st.integers(1, 12), min_size=3, max_size=12),
    minimum=st.integers(2, 25),
)
def test_pooled_centres_increase_and_never_outnumber_the_bins(
    counts: list[int], minimum: int
) -> None:
    raw = rows_in_bins(counts)
    target = np.arange(len(raw), dtype=np.float64) + 1.0
    fit = fit_correction(raw, target, min_top_count=minimum)
    assert 2 <= len(fit.centres) <= len(counts)
    assert (np.diff(fit.centres) > 0).all()
    assert (np.diff(fit.fitted) >= -1e-12).all()


def test_apply_equals_fitted_at_the_centres() -> None:
    fit = hand_fit()
    np.testing.assert_allclose(fit.apply(fit.centres), fit.fitted)


def test_apply_interpolates_linearly_between_centres() -> None:
    fit = hand_fit()
    np.testing.assert_allclose(fit.apply([1.0, 1.5, 2.0]), [4.125, 6.25, 6.25])
    two = fit_correction([0.5, 2.5], [1.0, 5.0], min_top_count=1)
    np.testing.assert_allclose(two.apply([1.0, 1.5, 2.0]), [2.0, 3.0, 4.0])


def test_apply_keeps_the_top_bin_ratio_above_the_top_centre_when_it_is_at_least_one() -> None:
    fit = hand_fit()
    ratio = HAND_FITTED[-1] / HAND_CENTRES[-1]
    assert ratio > 1.0
    np.testing.assert_allclose(fit.apply([7.0, 20.0]), np.array([7.0, 20.0]) * ratio)


def test_apply_never_shrinks_a_reading_above_the_top_centre() -> None:
    fit = fit_correction([0.5, 2.5], [0.4, 1.0], min_top_count=1)
    np.testing.assert_allclose(fit.apply([2.6, 10.0, 80.0]), [2.6, 10.0, 80.0])


def test_apply_keeps_the_bottom_bin_ratio_below_the_bottom_centre() -> None:
    fit = hand_fit()
    ratio = HAND_FITTED[0] / HAND_CENTRES[0]
    np.testing.assert_allclose(fit.apply([0.0, 0.25]), np.array([0.0, 0.25]) * ratio)


def test_apply_is_continuous_at_both_ends_when_the_top_ratio_is_at_least_one() -> None:
    fit = hand_fit()
    eps = 1e-9
    for centre in (fit.centres[0], fit.centres[-1]):
        below, at, above = fit.apply([centre - eps, centre, centre + eps])
        assert below == pytest.approx(at, abs=1e-6)
        assert above == pytest.approx(at, abs=1e-6)


def test_apply_jumps_upward_at_the_top_centre_when_the_floor_binds() -> None:
    fit = fit_correction([0.5, 2.5], [0.4, 1.0], min_top_count=1)
    eps = 1e-9
    below, at, above = fit.apply([2.5 - eps, 2.5, 2.5 + eps])
    assert at == pytest.approx(1.0)
    assert below == pytest.approx(at, abs=1e-6)
    assert above - at == pytest.approx(1.5, abs=1e-6)
    first = fit.centres[0]
    low, mid = fit.apply([first - eps, first])
    assert low == pytest.approx(mid, abs=1e-6)


def test_apply_passes_nan_through_and_accepts_scalars_and_lists() -> None:
    fit = hand_fit()
    out = fit.apply([np.nan, 1.5, np.nan])
    assert np.isnan(out[[0, 2]]).all()
    assert out[1] == pytest.approx(6.25)
    assert float(fit.apply(1.5)) == pytest.approx(6.25)


def test_knots_are_lists_rounded_to_three_decimals() -> None:
    fit = fit_correction([0.1234, 1.5555, 2.9], [1.00049, 2.00051, 3.123456], min_top_count=1)
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
    minimum=st.sampled_from([1, 5, 30]),
)
def test_apply_is_monotone_non_decreasing_and_never_shrinks_tall_readings(
    sample: list[tuple[float, float]], queries: list[float], minimum: int
) -> None:
    raw = np.array([r for r, _ in sample])
    target = np.array([t for _, t in sample])
    assume(len(np.unique(np.floor(raw))) >= 2)
    fit = fit_correction(raw, target, min_top_count=minimum)
    ordered = np.sort(queries)
    out = fit.apply(ordered)
    assert (np.diff(fit.fitted) >= -1e-12).all()
    assert (np.diff(out) >= -1e-9).all()
    tall = ordered > fit.centres[-1]
    assert (out[tall] >= ordered[tall] - 1e-9).all()


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

    def spy(
        r: npt.ArrayLike, t: npt.ArrayLike, bin_m: float = 1.0, min_top_count: int = 30
    ) -> HeightCorrection:
        trained.append(set(np.flatnonzero(np.isin(raw, np.asarray(r))).tolist()))
        return real(r, t, bin_m, min_top_count)

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


RASTER_BINS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-24", "24-30", "30+"]
STAT_KEYS = {"n", "before", "after", "signed_before", "signed_after", "aggregate_after"}


def test_validation_groups_raster_bins_by_raw_height_with_edges_going_up() -> None:
    edges = [0.0, 2.99, 3.0, 5.99, 6.0, 8.99, 9.0, 11.99, 12.0]
    edges += [14.99, 15.0, 17.99, 18.0, 23.99, 24.0, 29.99, 30.0, 55.0]
    n_districts = 10
    raw = np.tile(edges, n_districts)
    district = np.repeat(np.arange(n_districts), len(edges))
    result = validate_by_district(raw, raw + 1.0, district, np.ones(len(raw)))
    assert list(result["raster_bins"]) == RASTER_BINS
    assert {k: v["n"] for k, v in result["raster_bins"].items()} == {
        name: 2 * n_districts for name in RASTER_BINS
    }


def test_validation_gives_every_group_the_same_fields_and_empty_groups_are_none() -> None:
    raw, target, district, levels = panel(top=2)
    result = validate_by_district(raw, target, district, levels)
    groups = [*result["bands"].values(), *result["raster_bins"].values(), result["overall"]]
    for stats in groups:
        assert set(stats) == STAT_KEYS
    empty = result["bands"]["13+"]
    assert empty["n"] == 0
    assert {k: v for k, v in empty.items() if k != "n"} == dict.fromkeys(STAT_KEYS - {"n"})
    assert result["raster_bins"]["30+"]["n"] == 0
    assert result["overall"]["n"] == len(raw)


def test_validation_signed_error_is_the_median_and_aggregate_the_summed_ratio() -> None:
    raw = np.array([1.1, 1.2, 1.3, 1.4])
    target = np.array([1.0, 2.0, 2.0, 4.0])
    result = validate_by_district(raw, target, np.arange(4), np.ones(4))
    overall = result["overall"]
    assert overall["n"] == 4
    assert overall["before"] == pytest.approx(0.375, abs=1e-3)
    assert overall["signed_before"] == pytest.approx(-0.375, abs=1e-3)
    assert overall["signed_after"] == overall["signed_before"]
    assert overall["aggregate_after"] == pytest.approx(raw.sum() / target.sum() - 1, abs=1e-3)
    assert result["raster_bins"]["0-3"] == overall


def skewed_panel() -> Panel:
    rng = np.random.default_rng(7)
    n = 8000
    levels = rng.choice([1, 5, 14], size=n, p=[0.6, 0.25, 0.15])
    target = levels * LEVEL_M
    raw = rng.uniform(3.0, 42.0, size=n)
    district = rng.integers(0, 25, size=n)
    return raw, target, district, levels


def test_a_mean_fit_keeps_raster_bin_aggregates_near_zero_while_medians_and_bands_skew() -> None:
    raw, target, district, levels = skewed_panel()
    result = validate_by_district(raw, target, district, levels)

    bins = [v for v in result["raster_bins"].values() if v["n"] >= 200]
    assert len(bins) >= 5
    for stats in bins:
        assert abs(stats["aggregate_after"]) < 0.15
    assert max(abs(s["signed_after"]) for s in bins) > 0.5
    assert abs(result["overall"]["aggregate_after"]) < 0.05

    assert result["bands"]["1-2"]["signed_after"] > 0.5
    assert result["bands"]["5-7"]["signed_after"] < result["bands"]["1-2"]["signed_after"]
    assert result["bands"]["13+"]["signed_after"] < -0.3
