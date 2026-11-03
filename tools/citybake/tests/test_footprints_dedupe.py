import numpy as np
import numpy.typing as npt
import shapely
from footprints_fixture import bowtie, square, zero_area
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from twl_citybake.footprints.dedupe import dedupe

THRESHOLD = 0.3


def geoms(*items: shapely.Geometry) -> npt.NDArray[np.object_]:
    return np.asarray(items, dtype=object)


def rect(x0: float, y0: float, x1: float, y1: float) -> shapely.Geometry:
    return shapely.box(x0, y0, x1, y1)


def keep(higher: npt.NDArray[np.object_], lower: npt.NDArray[np.object_]) -> list[bool]:
    return dedupe(higher, lower, THRESHOLD).tolist()


def test_a_lower_footprint_with_29_percent_overlap_is_kept() -> None:
    assert keep(geoms(rect(0, 0, 20, 20)), geoms(rect(14.2, 0, 34.2, 20))) == [True]


def test_a_lower_footprint_with_31_percent_overlap_is_dropped() -> None:
    assert keep(geoms(rect(0, 0, 20, 20)), geoms(rect(13.8, 0, 33.8, 20))) == [False]


def test_overlap_exactly_at_the_threshold_is_kept() -> None:
    mask = dedupe(geoms(rect(0, 0, 10, 20)), geoms(rect(0, 0, 20, 20)), 0.5)
    assert mask.tolist() == [True]


def test_a_fully_covered_footprint_is_dropped() -> None:
    assert keep(geoms(rect(0, 0, 20, 20)), geoms(rect(0, 0, 20, 20))) == [False]


def test_a_footprint_inside_a_larger_higher_one_is_dropped() -> None:
    assert keep(geoms(rect(0, 0, 100, 100)), geoms(rect(40, 40, 50, 50))) == [False]


def test_disjoint_and_edge_touching_footprints_are_kept() -> None:
    lower = geoms(rect(100, 100, 120, 120), rect(20, 0, 40, 20))
    assert keep(geoms(rect(0, 0, 20, 20)), lower) == [True, True]


def test_overlaps_with_several_higher_footprints_add_up() -> None:
    lower = geoms(rect(0, 0, 20, 20))
    one = geoms(rect(0, 0, 4, 20))
    two = geoms(rect(0, 0, 4, 20), rect(16, 0, 20, 20))
    assert keep(one, lower) == [True]
    assert keep(two, lower) == [False]


def test_the_mask_is_boolean_aligned_with_lower_and_keeps_input_order() -> None:
    higher = geoms(rect(0, 0, 20, 20))
    lower = geoms(rect(0, 0, 20, 20), rect(50, 50, 60, 60), rect(0, 0, 19, 20), rect(70, 0, 80, 5))
    mask = dedupe(higher, lower, THRESHOLD)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [False, True, False, True]


def test_no_higher_footprints_keeps_every_valid_lower_one() -> None:
    lower = geoms(rect(0, 0, 20, 20), rect(0, 0, 20, 20))
    assert keep(geoms(), lower) == [True, True]


def test_no_lower_footprints_gives_an_empty_mask() -> None:
    mask = dedupe(geoms(rect(0, 0, 20, 20)), geoms(), THRESHOLD)
    assert mask.dtype == np.bool_
    assert mask.shape == (0,)


def test_invalid_and_zero_area_footprints_are_dropped_even_without_overlap() -> None:
    lower = geoms(bowtie(0, 0), zero_area(500, 500), shapely.Polygon(), square(900, 900))
    assert keep(geoms(square(-900, -900)), lower) == [False, False, False, True]


def test_invalid_higher_footprints_do_not_break_the_comparison() -> None:
    mask = dedupe(geoms(bowtie(0, 0), square(100, 100)), geoms(square(100, 100)), THRESHOLD)
    assert mask.tolist() == [False]


def test_the_inputs_are_not_modified() -> None:
    higher, lower = geoms(rect(0, 0, 20, 20)), geoms(rect(0, 0, 20, 20), rect(30, 0, 50, 20))
    before = (shapely.to_wkb(higher).copy(), shapely.to_wkb(lower).copy())
    dedupe(higher, lower, THRESHOLD)
    assert (shapely.to_wkb(higher) == before[0]).all()
    assert (shapely.to_wkb(lower) == before[1]).all()


def test_later_tiers_compare_against_the_kept_footprints_of_earlier_tiers_only() -> None:
    osm = geoms(rect(0, 0, 20, 20))
    google = geoms(rect(13.8, 0, 33.8, 20))
    inside_dropped_google = geoms(rect(25, 2, 33, 18))
    google_kept = dedupe(osm, google, THRESHOLD)
    assert google_kept.tolist() == [False]
    higher = np.concatenate([osm, google[google_kept]])
    assert dedupe(higher, inside_dropped_google, THRESHOLD).tolist() == [True]
    assert dedupe(np.concatenate([osm, google]), inside_dropped_google, THRESHOLD).tolist() == [
        False
    ]


COORD = st.integers(0, 20)


@st.composite
def rects(draw: st.DrawFn) -> tuple[int, int, int, int]:
    x0, y0 = draw(COORD), draw(COORD)
    return x0, y0, draw(st.integers(x0 + 1, 24)), draw(st.integers(y0 + 1, 24))


def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(max(w, 0) * max(h, 0))


@settings(max_examples=200, derandomize=True, deadline=None)
@given(higher=rects(), lower=rects(), threshold=st.sampled_from([0.1, 0.3, 0.5, 0.9]))
def test_a_single_higher_rectangle_drops_exactly_when_overlap_exceeds_the_threshold(
    higher: tuple[int, int, int, int], lower: tuple[int, int, int, int], threshold: float
) -> None:
    area = (lower[2] - lower[0]) * (lower[3] - lower[1])
    overlap = overlap_area(higher, lower)
    assume(abs(overlap - threshold * area) > 1e-6)
    mask = dedupe(geoms(rect(*higher)), geoms(rect(*lower)), threshold)
    assert mask.tolist() == [overlap <= threshold * area]


@settings(max_examples=100, derandomize=True, deadline=None)
@given(
    higher=st.lists(rects(), min_size=1, max_size=5),
    lower=st.lists(rects(), min_size=1, max_size=6),
)
def test_a_higher_threshold_never_drops_more_and_higher_order_is_irrelevant(
    higher: list[tuple[int, int, int, int]], lower: list[tuple[int, int, int, int]]
) -> None:
    h, lo = geoms(*(rect(*r) for r in higher)), geoms(*(rect(*r) for r in lower))
    strict, loose = dedupe(h, lo, 0.2), dedupe(h, lo, 0.6)
    assert (loose | ~strict).all()
    np.testing.assert_array_equal(dedupe(h[::-1], lo, 0.2), strict)


@settings(max_examples=50, derandomize=True, deadline=None)
@given(lower=st.lists(rects(), min_size=1, max_size=6))
def test_every_valid_footprint_is_dropped_against_a_higher_copy_of_itself(
    lower: list[tuple[int, int, int, int]],
) -> None:
    arr = geoms(*(rect(*r) for r in lower))
    assert not dedupe(arr, arr, THRESHOLD).any()
