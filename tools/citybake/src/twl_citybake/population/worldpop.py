import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.windows import Window

from twl_citybake.config import BBox

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
# Fractions of a cell (about 1e-7 m at 100 m) absorbed when snapping degrees to the grid, so
# points and bbox edges that sit on a cell boundary do not flip cells through rounding.
SNAP = 1e-9


@dataclass(frozen=True)
class PopGrid:
    """Persons per raster cell over a north-up window; row 0 is the northernmost."""

    values: FloatArray
    west: float
    north: float
    dlon: float
    dlat: float

    def cell_of(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> tuple[IntArray, IntArray]:
        """(row, col) of each point, -1 for both outside the window. West and north edges
        belong to the cell."""
        col = np.floor((np.asarray(lon, np.float64) - self.west) / self.dlon + SNAP)
        row = np.floor((self.north - np.asarray(lat, np.float64)) / self.dlat + SNAP)
        rows, cols = self.values.shape
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        return (
            np.where(inside, row, -1).astype(np.int64),
            np.where(inside, col, -1).astype(np.int64),
        )

    def centres(self) -> tuple[FloatArray, FloatArray]:
        rows, cols = self.values.shape
        lon = self.west + (np.arange(cols) + 0.5) * self.dlon
        lat = self.north - (np.arange(rows) + 0.5) * self.dlat
        grid_lon, grid_lat = np.meshgrid(lon, lat)
        return grid_lon, grid_lat


def read_worldpop(tif: Path, bbox: BBox) -> PopGrid:
    """Every raster cell that intersects the bbox; nodata, NaN and negative cells read as 0."""
    with rasterio.open(tif) as src:
        t = src.transform
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"{tif} is not in EPSG:4326")
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError(f"{tif} is not a north-up raster")
        dlon, dlat = t.a, -t.e
        col0 = _clamp(math.floor((bbox.west - t.c) / dlon + SNAP), src.width)
        col1 = _clamp(math.ceil((bbox.east - t.c) / dlon - SNAP), src.width)
        row0 = _clamp(math.floor((t.f - bbox.north) / dlat + SNAP), src.height)
        row1 = _clamp(math.ceil((t.f - bbox.south) / dlat - SNAP), src.height)
        if col1 <= col0 or row1 <= row0:
            raise ValueError(f"{tif} does not overlap {bbox}")
        band = src.read(1, window=Window.from_slices((row0, row1), (col0, col1)), masked=True)
    values = np.ma.filled(band.astype(np.float64), 0.0)
    values[~np.isfinite(values) | (values < 0)] = 0.0
    return PopGrid(values, west=t.c + col0 * dlon, north=t.f - row0 * dlat, dlon=dlon, dlat=dlat)


def _clamp(index: int, size: int) -> int:
    return min(max(index, 0), size)
