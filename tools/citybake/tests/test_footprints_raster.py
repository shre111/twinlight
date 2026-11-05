from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import shapely
from footprints_fixture import CELL_M, UTM, HeightGrid, empty_height_grid, square
from rasterio.crs import CRS

from twl_citybake.config import BBox
from twl_citybake.footprints.raster import HeightRaster, fetch_height_mosaic, mosaic_key
from twl_citybake.projection import LocalProjection

MIN_HEIGHT = 1.0
BBOX = BBox(72.57 - 0.00975, 23.02 - 0.00905, 72.57 + 0.00975, 23.02 + 0.00905)
PROJ = LocalProjection.for_bbox(BBOX)

A, B, C, D, E = (-600.0, 0.0), (-300.0, 0.0), (0.0, 0.0), (300.0, 0.0), (600.0, 0.0)


def medians(
    grid: HeightGrid, *footprints: shapely.Geometry, min_height: float = MIN_HEIGHT
) -> npt.NDArray[np.float32]:
    raster = HeightRaster(values=grid.values, transform=grid.transform, crs=CRS.from_string(UTM))
    return raster.footprint_medians(np.asarray(footprints, dtype=object), PROJ, min_height)


def test_a_footprint_takes_the_median_of_its_pixels() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [8.0] * 9)
    result = medians(grid, square(*A))
    assert result.dtype == np.float32
    assert result.tolist() == [8.0]


def test_the_median_is_not_the_mean() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [6.0] * 6 + [30.0] * 3)
    assert medians(grid, square(*A)).tolist() == [6.0]


def test_pixels_below_the_minimum_height_are_ignored() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [0.4] * 5 + [9.5] * 4)
    assert medians(grid, square(*A)).tolist() == [9.5]


def test_a_pixel_exactly_at_the_minimum_height_counts() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [1.0] * 9)
    assert medians(grid, square(*A)).tolist() == [1.0]


def test_the_minimum_height_argument_is_honoured() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [2.0] * 5 + [9.0] * 4)
    assert medians(grid, square(*A), min_height=1.0).tolist() == [2.0]
    assert medians(grid, square(*A), min_height=3.0).tolist() == [9.0]


def test_a_footprint_whose_pixels_are_all_below_the_minimum_is_nan() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [0.4] * 9)
    assert np.isnan(medians(grid, square(*A))).all()


def test_footprints_on_nodata_or_outside_the_raster_are_nan() -> None:
    grid = empty_height_grid(PROJ)
    result = medians(grid, square(*A), square(9000.0, 9000.0))
    assert np.isnan(result).all()


def test_each_footprint_gets_its_own_median_in_input_order() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [5.0] * 9)
    grid.paint(*C, [12.0] * 9)
    grid.paint(*E, [0.5] * 9)
    result = medians(grid, square(*E), square(*C), square(*B), square(*A))
    assert result[1] == 12.0 and result[3] == 5.0
    assert np.isnan(result[0]) and np.isnan(result[2])


def test_a_footprint_smaller_than_a_pixel_takes_the_pixel_under_its_centroid() -> None:
    grid = empty_height_grid(PROJ)
    row, col = grid.paint_cell(*D, 12.0)
    cx, cy = grid.cell_centre_local(row, col)
    off_centre = 1.5
    centre = (cx + off_centre, cy + off_centre)
    tiny = square(*centre, side=1.5)
    assert shapely.area(tiny) < CELL_M**2
    assert medians(grid, tiny).tolist() == [12.0]


def test_a_sub_pixel_footprint_over_a_ground_pixel_is_nan() -> None:
    grid = empty_height_grid(PROJ)
    row, col = grid.paint_cell(*D, 0.4)
    cx, cy = grid.cell_centre_local(row, col)
    assert np.isnan(medians(grid, square(cx + 1.5, cy + 1.5, side=1.5))).all()


def test_a_sub_pixel_footprint_over_nodata_is_nan() -> None:
    grid = empty_height_grid(PROJ)
    row, col = grid.cell_of(*D)
    cx, cy = grid.cell_centre_local(row, col)
    assert np.isnan(medians(grid, square(cx + 1.5, cy + 1.5, side=1.5))).all()


def test_footprints_are_matched_in_the_raster_crs_not_in_local_metres() -> None:
    grid = empty_height_grid(PROJ)
    grid.paint(*A, [7.0] * 9)
    shifted = square(A[0] + 60.0, A[1], side=12.0)
    assert np.isnan(medians(grid, shifted)).all()
    assert medians(grid, square(*A)).tolist() == [7.0]


MANIFEST = "https://example.invalid/manifest.json"
MOSAIC_BBOX = BBox(72.0, 23.0, 72.1, 23.1)


def test_a_mosaic_cache_built_from_the_same_inputs_is_kept(tmp_path: Path) -> None:
    grid = empty_height_grid(LocalProjection.for_bbox(MOSAIC_BBOX), 100.0)
    dest = grid.write(tmp_path / "heights.tif", mosaic_key(MANIFEST, MOSAIC_BBOX, CELL_M))
    before = dest.read_bytes()

    assert fetch_height_mosaic(MANIFEST, MOSAIC_BBOX, CELL_M, dest) == dest
    assert dest.read_bytes() == before


@pytest.mark.parametrize(
    "tags",
    [
        mosaic_key("https://example.invalid/other.json", MOSAIC_BBOX, CELL_M),
        mosaic_key(MANIFEST, BBox(72.0, 23.0, 72.2, 23.1), CELL_M),
        mosaic_key(MANIFEST, MOSAIC_BBOX, CELL_M * 2),
        None,
    ],
)
def test_a_mosaic_cache_built_from_anything_else_is_refused(
    tmp_path: Path, tags: dict[str, str] | None
) -> None:
    grid = empty_height_grid(LocalProjection.for_bbox(MOSAIC_BBOX), 100.0)
    dest = grid.write(tmp_path / "heights.tif", tags)

    with pytest.raises(ValueError, match="delete it to fetch again"):
        fetch_height_mosaic(MANIFEST, MOSAIC_BBOX, CELL_M, dest)
