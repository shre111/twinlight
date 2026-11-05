import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twl_citybake.population.allocate import Allocation, allocate
from twl_citybake.population.raster import PopGrid


def make_grid(values: list[list[float]]) -> PopGrid:
    return PopGrid(np.asarray(values, np.float64), west=0.0, north=10.0, dlon=1.0, dlat=1.0)


def run(
    grid: PopGrid,
    cells: list[tuple[int, int]],
    capacity: list[int],
    district: list[int],
    n_districts: int = 1,
) -> Allocation:
    lon, lat = grid.centres()
    rows = np.asarray([r for r, _ in cells], np.intp)
    cols = np.asarray([c for _, c in cells], np.intp)
    return allocate(
        grid,
        lon[rows, cols] + 0.1,
        lat[rows, cols] - 0.1,
        np.asarray(capacity, np.int64),
        np.asarray(district, np.uint16),
        n_districts,
    )


def test_cell_population_is_shared_in_proportion_to_capacity() -> None:
    out = run(make_grid([[20.0]]), [(0, 0), (0, 0)], [30, 10], [0, 0])
    assert out.target.tolist() == pytest.approx([15.0, 5.0])
    assert out.placed.tolist() == pytest.approx([15.0, 5.0])


def test_zero_capacity_buildings_get_nothing_beside_a_capable_neighbour() -> None:
    out = run(make_grid([[20.0]]), [(0, 0), (0, 0)], [0, 40], [0, 0])
    assert out.target.tolist() == pytest.approx([0.0, 20.0])
    assert out.placed[0] == 0.0


def test_each_cell_feeds_only_its_own_buildings_before_spill() -> None:
    out = run(make_grid([[10.0, 50.0]]), [(0, 0), (0, 1)], [100, 100], [0, 0])
    assert out.target.tolist() == pytest.approx([10.0, 50.0])


def test_cells_with_people_but_no_capacity_are_counted_and_unplaced() -> None:
    grid = make_grid([[55.0, 25.0], [0.0, 15.0]])
    out = run(grid, [(0, 0), (1, 0)], [50, 50], [0, 0])
    assert out.cells_with_people == 3
    assert out.cells_without_buildings == 2
    assert out.people_without_buildings == pytest.approx(40.0)
    assert out.target.sum() == pytest.approx(55.0)


def test_cell_whose_only_buildings_have_zero_capacity_counts_as_without_buildings() -> None:
    out = run(make_grid([[30.0]]), [(0, 0)], [0], [0])
    assert out.cells_without_buildings == 1
    assert out.people_without_buildings == pytest.approx(30.0)
    assert out.placed.tolist() == [0.0]


def test_overflow_spills_to_spare_capacity_in_the_same_district_only() -> None:
    grid = make_grid([[100.0, 0.0, 0.0]])
    out = run(
        grid,
        [(0, 0), (0, 1), (0, 2)],
        [40, 100, 1000],
        [0, 0, 1],
        n_districts=2,
    )
    assert out.placed.tolist() == pytest.approx([40.0, 60.0, 0.0])
    assert out.shortfall.tolist() == pytest.approx([0.0, 0.0])


def test_spill_is_proportional_to_spare_capacity() -> None:
    grid = make_grid([[100.0, 0.0, 0.0]])
    out = run(grid, [(0, 0), (0, 1), (0, 2)], [40, 30, 90], [0, 0, 0])
    assert out.placed[0] == pytest.approx(40.0)
    assert out.placed[1] / out.placed[2] == pytest.approx(30 / 90)
    assert out.placed.sum() == pytest.approx(100.0)


def test_full_district_reports_shortfall_and_fills_every_building() -> None:
    grid = make_grid([[80.0, 20.0]])
    out = run(grid, [(0, 0), (0, 1)], [30, 20], [0, 1], n_districts=2)
    assert out.placed.tolist() == [30.0, 20.0]
    assert out.shortfall.tolist() == pytest.approx([50.0, 0.0])
    assert out.marginal.tolist() == pytest.approx([80.0, 20.0])


def test_marginal_is_the_district_sum_of_target() -> None:
    grid = make_grid([[60.0, 40.0]])
    out = run(grid, [(0, 0), (0, 0), (0, 1)], [10, 30, 50], [0, 1, 1], n_districts=2)
    assert out.marginal.tolist() == pytest.approx([out.target[0], out.target[1] + out.target[2]])


def test_placed_never_exceeds_capacity_after_spill() -> None:
    grid = make_grid([[1000.0, 3.0]])
    capacity = [33, 17, 5, 999]
    out = run(grid, [(0, 0), (0, 0), (0, 0), (0, 1)], capacity, [0, 0, 0, 0])
    assert (out.placed <= np.asarray(capacity)).all()


@st.composite
def scenarios(
    draw: st.DrawFn,
) -> tuple[list[list[float]], list[tuple[int, int]], list[int], list[int], int]:
    rows, cols = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    values = [[float(draw(st.integers(0, 500))) for _ in range(cols)] for _ in range(rows)]
    n_districts = draw(st.integers(1, 3))
    n = draw(st.integers(1, 12))
    cells = [(draw(st.integers(0, rows - 1)), draw(st.integers(0, cols - 1))) for _ in range(n)]
    capacity = [draw(st.integers(0, 300)) for _ in range(n)]
    district = [draw(st.integers(0, n_districts - 1)) for _ in range(n)]
    return values, cells, capacity, district, n_districts


@settings(deadline=None, max_examples=200)
@given(scenarios())
def test_allocation_conserves_mass_and_respects_capacity(
    scenario: tuple[list[list[float]], list[tuple[int, int]], list[int], list[int], int],
) -> None:
    values, cells, capacity, district, n_districts = scenario
    grid = make_grid(values)
    out = run(grid, cells, capacity, district, n_districts)
    cap: npt.NDArray[np.int64] = np.asarray(capacity, np.int64)
    assert (out.placed <= cap).all()
    assert (out.placed >= 0).all()
    assert (out.shortfall >= 0).all()
    assert out.target.sum() + out.people_without_buildings == pytest.approx(
        grid.values.sum(), abs=1e-6
    )
    assert out.placed.sum() + out.shortfall.sum() == pytest.approx(out.target.sum(), abs=1e-6)
    by_district = np.bincount(np.asarray(district), weights=out.target, minlength=n_districts)
    assert out.marginal.tolist() == pytest.approx(by_district.tolist(), abs=1e-6)


@pytest.mark.parametrize(
    ("lon", "capacity", "district"),
    [
        ([0.5], [10, 10], [0, 0]),
        ([0.5, 0.5], [10, -1], [0, 0]),
        ([0.5, 0.5], [10, np.nan], [0, 0]),
        ([0.5, 0.5], [10, 10], [0, -1]),
    ],
)
def test_misaligned_or_negative_inputs_are_rejected(
    lon: list[float], capacity: list[float], district: list[int]
) -> None:
    grid = PopGrid(np.ones((1, 1)), west=0.0, north=1.0, dlon=1.0, dlat=1.0)
    with pytest.raises(ValueError):
        allocate(grid, lon, [0.5, 0.5], capacity, district, 1)
