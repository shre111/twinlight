from pathlib import Path

import numpy as np
import pytest
from population_fixture import NODATA, write_geotiff

from twl_citybake.config import BBox
from twl_citybake.population.worldpop import PopGrid, read_worldpop

CELL = 0.25
WEST = 72.0
NORTH = 24.0


@pytest.fixture
def source() -> np.ndarray:
    values = np.arange(1, 6 * 8 + 1, dtype=np.float32).reshape(6, 8)
    values[2, 3] = NODATA
    values[3, 4] = np.nan
    values[1, 5] = -7.0
    return values


@pytest.fixture
def tif(tmp_path: Path, source: np.ndarray) -> Path:
    return write_geotiff(tmp_path / "pop.tif", source, WEST, NORTH, cell=CELL)


def test_window_covers_exactly_the_cells_intersecting_the_bbox(tif: Path) -> None:
    bbox = BBox(WEST + 2.3 * CELL, NORTH - 3.6 * CELL, WEST + 5.4 * CELL, NORTH - 1.2 * CELL)
    grid = read_worldpop(tif, bbox)
    assert grid.values.shape == (3, 4)
    assert grid.west == pytest.approx(WEST + 2 * CELL)
    assert grid.north == pytest.approx(NORTH - 1 * CELL)
    assert grid.dlon == pytest.approx(CELL)
    assert grid.dlat == pytest.approx(CELL)


def test_window_values_come_from_the_matching_raster_cells(tif: Path, source: np.ndarray) -> None:
    bbox = BBox(WEST + 2.3 * CELL, NORTH - 3.6 * CELL, WEST + 5.4 * CELL, NORTH - 1.2 * CELL)
    grid = read_worldpop(tif, bbox)
    expected = np.where(np.isfinite(source) & (source > 0), source, 0.0)[1:4, 2:6]
    assert np.array_equal(grid.values, expected)


def test_nodata_nan_and_negative_cells_read_as_zero(tif: Path) -> None:
    grid = read_worldpop(tif, BBox(WEST, NORTH - 6 * CELL, WEST + 8 * CELL, NORTH))
    assert grid.values.shape == (6, 8)
    assert grid.values[2, 3] == 0.0
    assert grid.values[3, 4] == 0.0
    assert grid.values[1, 5] == 0.0
    assert grid.values.min() >= 0.0
    assert np.isfinite(grid.values).all()


def test_bbox_edge_exactly_on_a_cell_boundary_adds_no_extra_cell(tif: Path) -> None:
    grid = read_worldpop(tif, BBox(WEST + CELL, NORTH - 3 * CELL, WEST + 3 * CELL, NORTH - CELL))
    assert grid.values.shape == (2, 2)


def test_a_bbox_the_raster_only_partly_covers_is_rejected(tif: Path) -> None:
    with pytest.raises(ValueError, match="does not cover"):
        read_worldpop(tif, BBox(WEST - 1.0, NORTH - 7.0, WEST + 9.0, NORTH + 1.0))


def test_population_values_are_float64(tif: Path) -> None:
    grid = read_worldpop(tif, BBox(WEST, NORTH - 6 * CELL, WEST + 8 * CELL, NORTH))
    assert grid.values.dtype == np.float64


@pytest.fixture
def grid() -> PopGrid:
    return PopGrid(np.ones((3, 4)), west=10.0, north=20.0, dlon=0.5, dlat=0.25)


def test_cell_of_maps_interior_points_to_row_and_col(grid: PopGrid) -> None:
    row, col = grid.cell_of(np.array([10.2, 11.7, 10.99]), np.array([19.9, 19.3, 19.26]))
    assert row.tolist() == [0, 2, 2]
    assert col.tolist() == [0, 3, 1]


def test_cell_of_puts_west_and_north_edges_in_the_cell(grid: PopGrid) -> None:
    row, col = grid.cell_of(np.array([10.0, 10.5]), np.array([20.0, 19.75]))
    assert row.tolist() == [0, 1]
    assert col.tolist() == [0, 1]


def test_cell_of_returns_minus_one_outside_the_grid(grid: PopGrid) -> None:
    lon = np.array([9.999, 12.001, 11.0, 11.0])
    lat = np.array([19.5, 19.5, 20.001, 19.249])
    row, col = grid.cell_of(lon, lat)
    assert row.tolist() == [-1] * 4
    assert col.tolist() == [-1] * 4


def test_cell_of_returns_int64_arrays(grid: PopGrid) -> None:
    row, col = grid.cell_of(np.array([10.1]), np.array([19.9]))
    assert row.dtype == np.int64
    assert col.dtype == np.int64


def test_centres_have_grid_shape_and_cell_midpoints(grid: PopGrid) -> None:
    lon, lat = grid.centres()
    assert lon.shape == lat.shape == (3, 4)
    assert lon[0].tolist() == pytest.approx([10.25, 10.75, 11.25, 11.75])
    assert lat[:, 0].tolist() == pytest.approx([19.875, 19.625, 19.375])
    assert (lon == lon[0]).all()
    assert (lat[:, 0] == lat.T).all()


def test_cell_of_inverts_centres(grid: PopGrid) -> None:
    lon, lat = grid.centres()
    row, col = grid.cell_of(lon.ravel(), lat.ravel())
    assert row.tolist() == np.repeat(np.arange(3), 4).tolist()
    assert col.tolist() == np.tile(np.arange(4), 3).tolist()
