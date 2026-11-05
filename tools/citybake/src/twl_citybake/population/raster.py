import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import osmium
import rasterio
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area
from rasterio.windows import Window

from twl_citybake.config import BBox

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
# Fractions of a cell (about 1e-7 m at 3 arc seconds) absorbed when snapping degrees to the grid, so
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


def read_raster(tif: Path, bbox: BBox) -> PopGrid:
    """Every raster cell that intersects the bbox; nodata, NaN and negative cells read as 0."""
    with rasterio.open(tif) as src:
        t = src.transform
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"{tif} is not in EPSG:4326")
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError(f"{tif} is not a north-up raster")
        dlon, dlat = t.a, -t.e
        col0 = math.floor((bbox.west - t.c) / dlon + SNAP)
        col1 = math.ceil((bbox.east - t.c) / dlon - SNAP)
        row0 = math.floor((t.f - bbox.north) / dlat + SNAP)
        row1 = math.ceil((t.f - bbox.south) / dlat - SNAP)
        if col0 < 0 or row0 < 0 or col1 > src.width or row1 > src.height:
            raise ValueError(f"{tif} does not cover {bbox}")
        band = src.read(1, window=Window.from_slices((row0, row1), (col0, col1)), masked=True)
    values = np.ma.filled(band.astype(np.float64), 0.0)
    values[~np.isfinite(values) | (values < 0)] = 0.0
    return PopGrid(values, west=t.c + col0 * dlon, north=t.f - row0 * dlat, dlon=dlon, dlat=dlat)


def read_boundary(pbf: Path, polygon: str) -> shapely.Geometry:
    """An OSM boundary (`relation/<id>` or `way/<id>`) from the clipped extract, in lon/lat. The
    clip keeps boundary relations complete, so this is the whole area, not its part in the bbox."""
    found = _scan_boundaries(pbf, [polygon])
    if polygon not in found:
        raise ValueError(f"population_total polygon {polygon} is not in {pbf}")
    return found[polygon]


def read_boundaries(pbf: Path, polygons: Sequence[str]) -> dict[str, shapely.Geometry]:
    """Like `read_boundary` for several polygons, in one pass over the extract."""
    found = _scan_boundaries(pbf, polygons)
    if missing := [polygon for polygon in polygons if polygon not in found]:
        raise ValueError(f"polygons {', '.join(missing)} are not in {pbf}")
    return {polygon: found[polygon] for polygon in polygons}


def _scan_boundaries(pbf: Path, polygons: Sequence[str]) -> dict[str, shapely.Geometry]:
    wanted = {
        (kind == "way", int(osm_id)): polygon
        for polygon in polygons
        for kind, _, osm_id in [polygon.partition("/")]
    }
    found: dict[str, shapely.Geometry] = {}
    if not wanted:
        return found
    factory = WKBFactory()
    for obj in osmium.FileProcessor(str(pbf)).with_areas():
        if isinstance(obj, Area) and (obj.from_way(), obj.orig_id()) in wanted:
            polygon = wanted[(obj.from_way(), obj.orig_id())]
            found.setdefault(polygon, shapely.from_wkb(factory.create_multipolygon(obj)))
            if len(found) == len(wanted):
                break
    return found


def rescale_to_total(
    grid: PopGrid, people: float, boundary: shapely.Geometry
) -> tuple[PopGrid, float]:
    """Scale the cells whose centres lie in `boundary` so they sum to `people`, leaving the rest
    alone; returns the grid and the raster's own total inside the boundary. A boundary reaching
    past the window is refused: the total would be spread over only part of its area."""
    rows, cols = grid.values.shape
    west, south, east, north = shapely.bounds(boundary)
    if (
        west < grid.west
        or east > grid.west + cols * grid.dlon
        or north > grid.north
        or south < grid.north - rows * grid.dlat
    ):
        raise ValueError("population_total polygon extends beyond the raster window")
    lon, lat = grid.centres()
    inside = shapely.contains_xy(boundary, lon, lat)
    before = float(grid.values[inside].sum())
    if before <= 0:
        raise ValueError("the raster has no people inside the population_total polygon")
    values = grid.values.copy()
    values[inside] *= people / before
    return replace(grid, values=values), before
