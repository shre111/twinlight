import math
from typing import Any

import numpy as np
import pytest
from correction_fixture import (
    DEFAULT_EDGES,
    Bldg,
    cell_area,
    make_grid,
    run_correction,
)

W = 0.25
MAIN_VALUES = [[400.0, 600.0, 100.0], [100.0, 50.0, 30.0]]
MAIN_REFERENCE = [[True, True, True], [False, False, False]]
MAIN_BUILDINGS = [
    Bldg(0, 0, 200.0, 1.0, 100.0),
    Bldg(0, 1, 200.0, 1.0, 100.0),
    Bldg(0, 2, 200.0, 4.0, 200.0),
    Bldg(1, 0, 200.0, 1.0, 50.0),
    Bldg(1, 2, 200.0, 1.0, 0.0),
]
MAIN_FIT = {(0, 0): 500.0, (0, 1): 500.0, (0, 2): 100.0, (1, 0): 250.0}
MAIN_TOTAL = 1280.0
CLASS_LOW = "h0-f1-c0"
CLASS_TALL = "h2-f1-c0"


def main_run(values: Any = None, **overrides: Any) -> Any:
    return run_correction(
        MAIN_VALUES if values is None else values,
        MAIN_BUILDINGS,
        MAIN_REFERENCE,
        **overrides,
    )


def blend(grid_value: float, fit: float, w: float) -> float:
    return grid_value ** (1 - w) * fit**w


def test_corrected_cells_are_raster_to_one_minus_w_times_fit_to_w_rescaled_to_the_grid_total() -> (
    None
):
    raw = {
        (0, 0): 400.0**0.75 * 500.0**0.25,
        (0, 1): 600.0**0.75 * 500.0**0.25,
        (0, 2): 100.0**0.75 * 100.0**0.25,
        (1, 0): 100.0**0.75 * 250.0**0.25,
    }
    scale = MAIN_TOTAL / sum(raw.values())
    out = main_run(weight=W).grid.values
    for cell, value in raw.items():
        assert out[cell] == pytest.approx(value * scale, rel=1e-12)
    assert out[1, 1] == 0.0
    assert out[1, 2] == 0.0


def test_total_equals_the_whole_grid_including_people_in_cells_without_housing() -> None:
    out = main_run(weight=W).grid.values
    assert out.sum() == pytest.approx(MAIN_TOTAL, rel=1e-12)
    housed_total = 400.0 + 600.0 + 100.0 + 100.0
    assert out.sum() != pytest.approx(housed_total, rel=1e-6)


def test_swapping_w_and_one_minus_w_would_give_other_values() -> None:
    out = main_run(weight=W).grid.values
    swapped = {
        cell: blend(MAIN_VALUES[cell[0]][cell[1]], fit, 1 - W) for cell, fit in MAIN_FIT.items()
    }
    scale = MAIN_TOTAL / sum(swapped.values())
    assert out[0, 0] != pytest.approx(swapped[(0, 0)] * scale, rel=1e-3)
    assert out[0, 1] != pytest.approx(swapped[(0, 1)] * scale, rel=1e-3)


def test_weight_one_gives_the_fit_rescaled_to_the_grid_total() -> None:
    out = main_run(weight=1.0, weight_range=(0.0, 1.0)).grid.values
    fit_total = sum(MAIN_FIT.values())
    for cell, fit in MAIN_FIT.items():
        assert out[cell] == pytest.approx(fit * MAIN_TOTAL / fit_total, rel=1e-12)
    assert out[1, 1] == 0.0 and out[1, 2] == 0.0


def test_fit_applies_the_class_rate_to_housing_floor_not_to_all_floor() -> None:
    out = main_run(weight=1.0).grid.values
    on_all_floor = {(0, 0): 1000.0, (0, 1): 1000.0, (0, 2): 400.0, (1, 0): 1000.0}
    scale = MAIN_TOTAL / sum(on_all_floor.values())
    assert out[0, 2] / out[0, 0] == pytest.approx(100.0 / 500.0, rel=1e-12)
    assert out[0, 2] != pytest.approx(on_all_floor[(0, 2)] * scale, rel=1e-3)


def test_a_cell_the_raster_gives_zero_stays_zero_below_weight_one() -> None:
    values = [[400.0, 600.0, 100.0], [0.0, 50.0, 30.0]]
    out = main_run(values, weight=W).grid.values
    assert out[1, 0] == 0.0
    assert out[0, 0] > 0.0
    assert out.sum() == pytest.approx(1180.0, rel=1e-12)


def test_people_moved_is_half_the_sum_of_absolute_cell_changes() -> None:
    result = main_run(weight=W)
    expected = 0.5 * float(np.abs(result.grid.values - np.asarray(MAIN_VALUES)).sum())
    assert result.report["people_moved"] == pytest.approx(expected, rel=1e-12)
    assert result.report["people_moved"] >= 80.0 / 2


def test_corrected_grid_keeps_the_input_geometry_and_the_input_is_not_modified() -> None:
    grid = make_grid(MAIN_VALUES)
    result = main_run(weight=W)
    assert result.grid.values.shape == grid.values.shape
    assert (result.grid.west, result.grid.north) == (grid.west, grid.north)
    assert (result.grid.dlon, result.grid.dlat) == (grid.dlon, grid.dlat)
    values = np.asarray(MAIN_VALUES)
    run_correction(values, MAIN_BUILDINGS, MAIN_REFERENCE, weight=W)
    assert values.tolist() == MAIN_VALUES


def test_the_report_carries_the_config_and_the_reference_cell_count() -> None:
    report = main_run(weight=W, weight_range=(0.1, 0.9)).report
    assert report["weight"] == W
    assert list(report["weight_range"]) == [0.1, 0.9]
    assert report["weight_evidence"] == "chosen after seeing the ward data"
    assert list(report["reference_polygons"]) == ["relation/1"]
    assert report["reference_cells"] == 3


def test_a_second_run_gives_identical_cells() -> None:
    first = main_run(weight=W).grid.values
    second = main_run(weight=W).grid.values
    assert np.array_equal(first, second)


def test_buildings_outside_the_grid_window_are_ignored() -> None:
    base = main_run(weight=W).grid.values
    outside = [
        (72.0 - 0.5, 23.0 - 0.0005, Bldg(0, 0, 200.0, 1.0, 9999.0)),
        (72.0 + 0.0005, 23.0 + 0.5, Bldg(0, 0, 200.0, 1.0, 9999.0)),
        (72.0 + 5.0, 23.0 - 0.0005, Bldg(0, 0, 200.0, 1.0, 9999.0)),
        (72.0 + 0.0005, 23.0 - 5.0, Bldg(0, 0, 200.0, 1.0, 9999.0)),
    ]
    result = run_correction(
        MAIN_VALUES, MAIN_BUILDINGS, MAIN_REFERENCE, extra_lonlat=outside, weight=W
    )
    assert np.array_equal(result.grid.values, base)


def shares(out: Any) -> np.ndarray:
    return out.grid.values / out.grid.values.sum()


def test_changing_the_raster_outside_the_reference_leaves_every_rate_unchanged() -> None:
    base = shares(main_run(weight=1.0))
    values = [[400.0, 600.0, 100.0], [5000.0, 50.0, 30.0]]
    assert shares(main_run(values, weight=1.0)) == pytest.approx(base, rel=1e-12)


def test_changing_the_raster_inside_the_reference_changes_the_rates() -> None:
    base = shares(main_run(weight=1.0))
    values = [[800.0, 600.0, 100.0], [100.0, 50.0, 30.0]]
    changed = shares(main_run(values, weight=1.0))
    assert not np.allclose(changed, base, rtol=1e-6)
    fit_low = (800.0 + 600.0) / 200.0
    expected = np.array([fit_low * 100.0, fit_low * 100.0, 100.0, fit_low * 50.0])
    got = changed[[0, 0, 0, 1], [0, 1, 2, 0]]
    assert got == pytest.approx(expected / expected.sum(), rel=1e-12)


def single_cell_label(buildings: list[Bldg], **overrides: Any) -> str:
    result = run_correction([[100.0]], buildings, [[True]], weight=W, **overrides)
    (label,) = result.report["classes"]["used"]
    return label


@pytest.mark.parametrize(
    ("floors", "label"),
    [
        (1.9, "h0"),
        (2.0, "h1"),
        (2.9, "h1"),
        (3.0, "h2"),
        (5.0, "h3"),
        (7.9, "h3"),
        (8.0, "h4"),
        (30.0, "h4"),
    ],
)
def test_height_edges_split_classes_with_the_edge_in_the_upper_bin(
    floors: float, label: str
) -> None:
    assert single_cell_label([Bldg(0, 0, 200.0, floors, 100.0)]) == f"{label}-f1-c0"


@pytest.mark.parametrize(
    ("footprint", "label"),
    [(50.0, "f0"), (100.0, "f1"), (249.0, "f1"), (250.0, "f2"), (600.0, "f3"), (1500.0, "f4")],
)
def test_footprint_edges_split_classes_with_the_edge_in_the_upper_bin(
    footprint: float, label: str
) -> None:
    assert single_cell_label([Bldg(0, 0, footprint, 1.0, 100.0)]).split("-")[1] == label


@pytest.mark.parametrize(
    ("edges_in_cell_coverages", "label"),
    [
        ((0.5, 1.5, 3.0), "c1"),
        ((0.2, 0.5, 0.9), "c3"),
        ((2.0, 3.0, 4.0), "c0"),
        ((0.1, 0.5, 2.0), "c2"),
    ],
)
def test_coverage_edges_apply_to_footprint_over_cell_area(
    edges_in_cell_coverages: tuple[float, float, float], label: str
) -> None:
    coverage = 200.0 / cell_area(make_grid([[100.0]]))
    edges = tuple(e * coverage for e in edges_in_cell_coverages)
    assert (
        single_cell_label([Bldg(0, 0, 200.0, 1.0, 100.0)], coverage_edges=edges).split("-")[2]
        == label
    )


def test_swapping_height_and_footprint_would_give_a_different_class() -> None:
    assert single_cell_label([Bldg(0, 0, 600.0, 3.0, 100.0)]) == "h2-f3-c0"
    assert single_cell_label([Bldg(0, 0, 150.0, 6.0, 100.0)]) == "h3-f1-c0"


def test_height_and_footprint_are_floor_weighted_means_over_the_cell() -> None:
    cell = [Bldg(0, 0, 100.0, 1.0, 50.0), Bldg(0, 0, 300.0, 10.0, 50.0)]
    assert single_cell_label(cell) == "h4-f2-c0"


def test_coverage_counts_footprints_not_all_floor() -> None:
    cell = [Bldg(0, 0, 100.0, 1.0, 50.0), Bldg(0, 0, 300.0, 10.0, 50.0)]
    assert single_cell_label(cell).endswith("-c0")
    coverage = 400.0 / cell_area(make_grid([[100.0]]))
    assert coverage < DEFAULT_EDGES["coverage_edges"][0]


def test_classes_are_reported_as_used_pooled_and_capped_label_lists() -> None:
    report = main_run(weight=W).report["classes"]
    assert sorted(report["used"]) == [CLASS_LOW, CLASS_TALL]
    assert list(report["pooled"]) == []
    assert list(report["capped"]) == []


POOL_VALUES = [[2000.0, 30.0, 7.0, 7.0, 3.0]]
POOL_BUILDINGS = [
    Bldg(0, 0, 200.0, 1.0, 1000.0),
    Bldg(0, 1, 200.0, 4.0, 10.0),
    Bldg(0, 2, 200.0, 4.0, 100.0),
    Bldg(0, 3, 200.0, 1.0, 500.0),
    Bldg(0, 4, 200.0, 10.0, 100.0),
]
POOL_REFERENCE = [[True, True, False, False, False]]


def pool_run(**overrides: Any) -> Any:
    return run_correction(POOL_VALUES, POOL_BUILDINGS, POOL_REFERENCE, weight=1.0, **overrides)


def test_a_class_under_min_class_share_and_an_absent_class_use_the_reference_rate() -> None:
    result = pool_run(min_class_share=0.05)
    reference_rate = 2030.0 / 1010.0
    fits = [2000.0, reference_rate * 10, reference_rate * 100, 2.0 * 500.0, reference_rate * 100]
    expected = np.array(fits) * 2047.0 / sum(fits)
    assert result.grid.values[0] == pytest.approx(expected, rel=1e-12)
    assert CLASS_TALL in result.report["classes"]["pooled"]
    assert CLASS_LOW in result.report["classes"]["used"]


def test_a_class_above_min_class_share_keeps_its_own_rate() -> None:
    result = pool_run(min_class_share=0.005)
    fits = [2000.0, 30.0, 300.0, 1000.0, (2030.0 / 1010.0) * 100]
    expected = np.array(fits) * 2047.0 / sum(fits)
    assert result.grid.values[0] == pytest.approx(expected, rel=1e-12)
    assert CLASS_TALL in result.report["classes"]["used"]
    assert CLASS_TALL not in result.report["classes"]["pooled"]


def test_a_rate_above_one_over_fit_min_is_capped_and_reported() -> None:
    values = [[500.0, 900.0, 100.0]]
    buildings = [
        Bldg(0, 0, 200.0, 1.0, 100.0),
        Bldg(0, 1, 200.0, 4.0, 900.0),
        Bldg(0, 2, 200.0, 1.0, 100.0),
    ]
    result = run_correction(
        values, buildings, [[True, True, False]], weight=1.0, fit_min_m2_per_person=0.5
    )
    fits = np.array([200.0, 900.0, 200.0])
    assert result.grid.values[0] == pytest.approx(fits * 1500.0 / fits.sum(), rel=1e-12)
    assert list(result.report["classes"]["capped"]) == [CLASS_LOW]


def test_a_class_rate_is_the_ratio_of_sums_not_the_mean_of_cell_ratios() -> None:
    values = [[100.0, 500.0, 10.0, 100.0]]
    buildings = [
        Bldg(0, 0, 200.0, 1.0, 100.0),
        Bldg(0, 1, 200.0, 1.0, 300.0),
        Bldg(0, 2, 200.0, 1.0, 100.0),
        Bldg(0, 3, 200.0, 4.0, 100.0),
    ]
    result = run_correction(values, buildings, [[True, True, False, True]], weight=1.0)
    fits = np.array([150.0, 450.0, 150.0, 100.0])
    assert result.grid.values[0] == pytest.approx(fits * 710.0 / fits.sum(), rel=1e-12)


def test_a_class_absent_from_the_reference_uses_the_reference_rate_even_with_no_share_floor() -> (
    None
):
    result = pool_run(min_class_share=0.0)
    reference_rate = 2030.0 / 1010.0
    fits = [2000.0, 30.0, 300.0, 1000.0, reference_rate * 100]
    expected = np.array(fits) * 2047.0 / sum(fits)
    assert result.grid.values[0] == pytest.approx(expected, rel=1e-12)
    assert list(result.report["classes"]["pooled"]) == ["h4-f1-c0"]


def test_a_rate_exactly_at_the_cap_is_not_reported_as_capped() -> None:
    values = [[200.0, 900.0]]
    buildings = [Bldg(0, 0, 200.0, 1.0, 100.0), Bldg(0, 1, 200.0, 4.0, 900.0)]
    result = run_correction(
        values, buildings, [[True, True]], weight=1.0, fit_min_m2_per_person=0.5
    )
    assert list(result.report["classes"]["capped"]) == []


@pytest.mark.parametrize(
    ("values", "reference"),
    [
        ([[100.0, 100.0]], [[False, False]]),
        ([[0.0, 0.0]], [[True, True]]),
    ],
)
def test_a_reference_with_no_people_or_no_selected_cells_raises(
    values: list[list[float]], reference: list[list[bool]]
) -> None:
    buildings = [Bldg(0, 0, 200.0, 1.0, 100.0), Bldg(0, 1, 200.0, 1.0, 100.0)]
    with pytest.raises(ValueError):
        run_correction(values, buildings, reference, weight=W)


def test_a_reference_cell_with_people_but_no_housing_enters_no_rate() -> None:
    buildings = [
        Bldg(0, 0, 200.0, 1.0, 100.0),
        Bldg(0, 1, 200.0, 1.0, 100.0),
        Bldg(0, 3, 200.0, 4.0, 100.0),
    ]
    result = run_correction(
        [[100.0, 300.0, 1000.0, 50.0]],
        buildings,
        [[True, True, True, False]],
        weight=1.0,
    )
    third = 1450.0 / 3
    assert result.grid.values[0] == pytest.approx([third, third, 0.0, third], rel=1e-12)


def test_a_pooled_rate_above_the_cap_is_capped_too() -> None:
    buildings = [Bldg(0, 0, 200.0, 1.0, 100.0), Bldg(0, 1, 200.0, 4.0, 100.0)]
    result = run_correction(
        [[900.0, 100.0]], buildings, [[True, False]], weight=1.0, fit_min_m2_per_person=0.5
    )
    assert result.grid.values[0] == pytest.approx([500.0, 500.0], rel=1e-12)
    assert list(result.report["classes"]["pooled"]) == [CLASS_TALL]
    assert list(result.report["classes"]["capped"]) == [CLASS_LOW, CLASS_TALL]


def test_a_class_with_housing_but_no_people_in_the_reference_gives_its_cells_nothing() -> None:
    buildings = [
        Bldg(0, 0, 200.0, 1.0, 100.0),
        Bldg(0, 1, 200.0, 4.0, 100.0),
        Bldg(0, 2, 200.0, 4.0, 100.0),
    ]
    result = run_correction([[100.0, 0.0, 50.0]], buildings, [[True, True, False]], weight=0.5)
    assert result.grid.values[0] == pytest.approx([150.0, 0.0, 0.0], rel=1e-12)


def test_a_reference_whose_cells_have_no_housing_raises() -> None:
    buildings = [Bldg(0, 0, 200.0, 1.0, 0.0), Bldg(0, 1, 200.0, 1.0, 100.0)]
    with pytest.raises(ValueError):
        run_correction([[100.0, 100.0]], buildings, [[True, False]], weight=W)


FOLD_COLS = 66


def fold_grid(
    x_housing: dict[int, float],
    x_people: dict[int, float],
    extra: dict[int, tuple[float, float, float]] | None = None,
) -> Any:
    values = np.zeros((1, FOLD_COLS))
    buildings = []
    for col, housing in x_housing.items():
        values[0, col] = x_people[col]
        buildings.append(Bldg(0, col, 200.0, 1.0, housing))
    for col, (people, housing, floors) in (extra or {}).items():
        values[0, col] = people
        buildings.append(Bldg(0, col, 200.0, floors, housing))
    return run_correction(values, buildings, np.ones((1, FOLD_COLS), bool), weight=W)


X_HOUSING = {0: 200.0, 1: 200.0, 22: 100.0, 23: 300.0, 44: 200.0, 45: 200.0}
X_PEOPLE = {0: 200.0, 1: 200.0, 22: 100.0, 23: 500.0, 44: 700.0, 45: 700.0}
CV_X = math.sqrt(3.5 / 3) / 2.0


def test_fold_variation_is_std_over_mean_of_the_per_fold_class_rates() -> None:
    report = fold_grid(X_HOUSING, X_PEOPLE).report["fold_variation"]
    assert report["floor_weighted"] == pytest.approx(CV_X, rel=1e-12)
    assert report["max"] == pytest.approx(CV_X, rel=1e-12)
    assert report["floor_share"] == pytest.approx(1.0, rel=1e-12)


def test_fold_variation_floor_weights_the_mean_and_max_takes_the_largest_class() -> None:
    z = {3: (100.0, 100.0, 4.0), 25: (100.0, 100.0, 4.0), 47: (700.0, 100.0, 4.0)}
    report = fold_grid(X_HOUSING, X_PEOPLE, z).report["fold_variation"]
    cv_z = math.sqrt(8.0) / 3.0
    assert report["max"] == pytest.approx(cv_z, rel=1e-12)
    assert report["floor_weighted"] == pytest.approx(
        (1200.0 * CV_X + 300.0 * cv_z) / 1500.0, rel=1e-12
    )
    assert report["floor_share"] == pytest.approx(1.0, rel=1e-12)


def test_fold_blocks_are_22_cells_wide() -> None:
    cols: tuple[int, ...] = (20, 21, 42, 43, 64, 65)
    housing: dict[int, float] = dict(zip(cols, X_HOUSING.values(), strict=True))
    people: dict[int, float] = dict(zip(cols, X_PEOPLE.values(), strict=True))
    report = fold_grid(housing, people).report["fold_variation"]
    assert report["max"] == pytest.approx(CV_X, rel=1e-12)


def test_the_fold_index_wraps_after_five_blocks() -> None:
    values = np.zeros((1, 132))
    buildings = []
    for col, people in ((0, 100.0), (22, 200.0), (44, 300.0), (110, 500.0)):
        values[0, col] = people
        buildings.append(Bldg(0, col, 200.0, 1.0, 100.0))
    result = run_correction(values, buildings, np.ones((1, 132), bool), weight=W)
    expected = math.sqrt(2.0 / 9.0) / (8.0 / 3.0)
    assert result.report["fold_variation"]["max"] == pytest.approx(expected, rel=1e-12)


def test_a_class_present_in_fewer_than_three_folds_is_left_out_of_fold_variation() -> None:
    y = {2: (100.0, 100.0, 4.0), 24: (900.0, 100.0, 4.0)}
    report = fold_grid(X_HOUSING, X_PEOPLE, y).report["fold_variation"]
    assert report["max"] == pytest.approx(CV_X, rel=1e-12)
    assert report["floor_weighted"] == pytest.approx(CV_X, rel=1e-12)
    assert report["floor_share"] == pytest.approx(1200.0 / 1400.0, rel=1e-12)


def test_fold_variation_report_has_the_documented_keys() -> None:
    report = fold_grid(X_HOUSING, X_PEOPLE).report["fold_variation"]
    assert {"floor_weighted", "max", "classes", "floor_share"} <= set(report)
    keys = set(fold_grid(X_HOUSING, X_PEOPLE).report)
    assert {
        "weight",
        "weight_range",
        "weight_evidence",
        "reference_polygons",
        "reference_cells",
        "classes",
        "fold_variation",
        "people_moved",
    } <= keys
