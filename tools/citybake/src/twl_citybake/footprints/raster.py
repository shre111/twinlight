import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window

from twl_citybake.config import BBox
from twl_citybake.osm.geometry import GeometryArray, bbox_outline
from twl_citybake.projection import LocalProjection

FloatArray = npt.NDArray[np.float32]
HEIGHT_BAND = "building_height"


@dataclass(frozen=True)
class HeightRaster:
    """Building height in metres on a north-up grid; NaN where there is no data."""

    values: FloatArray
    transform: Affine
    crs: CRS

    @classmethod
    def read(cls, path: Path) -> "HeightRaster":
        with rasterio.open(path) as src:
            values = np.ma.filled(src.read(1, masked=True).astype(np.float32), np.nan)
            values[~np.isfinite(values)] = np.nan
            return cls(values, src.transform, src.crs)

    def footprint_medians(
        self, geoms_local: GeometryArray, proj: LocalProjection, min_height: float
    ) -> FloatArray:
        """Median of the pixels at or above `min_height` whose centres lie in each footprint.
        A footprint over no pixel centre takes the pixel under its centroid; NaN when there is no
        such pixel. Where footprints overlap, the earlier one claims the shared pixels."""
        result = np.full(len(geoms_local), np.nan, np.float32)
        to_crs = Transformer.from_crs(proj.proj4, self.crs.to_wkt(), always_xy=True)
        geoms = shapely.transform(
            geoms_local, lambda xy: np.column_stack(to_crs.transform(xy[:, 0], xy[:, 1]))
        )
        owner = self._burn(geoms).ravel()
        covered = np.flatnonzero(owner)
        owner, height = owner[covered] - 1, self.values.ravel()[covered]
        tall = height >= min_height
        groups, medians = _medians(owner[tall], height[tall])
        result[groups] = medians
        alone = np.flatnonzero(np.bincount(owner, minlength=len(geoms)) == 0)
        result[alone] = self._centroid_pixels(geoms[alone], min_height)
        return result

    def _burn(self, geoms: GeometryArray) -> npt.NDArray[np.int32]:
        """Footprint index + 1 at each pixel whose centre it covers, 0 elsewhere."""
        usable = np.flatnonzero(~shapely.is_missing(geoms) & ~shapely.is_empty(geoms))
        if not len(usable):
            return np.zeros(self.values.shape, np.int32)
        # rasterize paints in order and the last shape wins, so paint in reverse priority
        shapes = ((geoms[i], int(i) + 1) for i in usable[::-1])
        return rasterize(
            shapes,
            out_shape=self.values.shape,
            transform=self.transform,
            fill=0,
            all_touched=False,
            dtype="int32",
        )

    def _centroid_pixels(self, geoms: GeometryArray, min_height: float) -> FloatArray:
        out = np.full(len(geoms), np.nan, np.float32)
        if not len(geoms):
            return out
        centroids = shapely.centroid(geoms)
        x, y = shapely.get_x(centroids), shapely.get_y(centroids)
        inverse = ~self.transform
        col = np.floor(inverse.a * x + inverse.b * y + inverse.c)
        row = np.floor(inverse.d * x + inverse.e * y + inverse.f)
        rows, cols = self.values.shape
        inside = np.flatnonzero((row >= 0) & (row < rows) & (col >= 0) & (col < cols))
        value = self.values[row[inside].astype(np.intp), col[inside].astype(np.intp)]
        out[inside] = np.where(value >= min_height, value, np.nan)
        return out


def _medians(keys: np.ndarray, values: np.ndarray) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Per-group medians, equal to np.median: the mean of the two middle values once sorted."""
    order = np.lexsort((values, keys))
    keys, values = keys[order], values[order].astype(np.float64)
    groups, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    medians = (values[starts + (counts - 1) // 2] + values[starts + counts // 2]) / 2
    return groups.astype(np.intp), medians.astype(np.float32)


@dataclass(frozen=True)
class Tile:
    uri: str
    x0: float
    y0: float
    scale: float
    width: int
    height: int

    def box(self) -> shapely.Polygon:
        x1, y1 = self.x0 + self.scale * self.width, self.y0 - self.scale * self.height
        return shapely.box(self.x0, y1, x1, self.y0)


def fetch_height_mosaic(manifest_url: str, bbox: BBox, cell_m: float, dest: Path) -> Path:
    """The manifest's building height band over the bbox at `cell_m`, read from the tiles' own
    overviews and cached once as a GeoTIFF in the manifest CRS. Where tiles overlap, the first by
    URI with data wins."""
    if dest.exists():
        return dest
    manifest = _get_json(manifest_url)
    band, missing = _height_band(manifest)
    crs = _manifest_crs(manifest)
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    outline = shapely.Polygon(np.column_stack(to_crs.transform(*bbox_outline(bbox))))
    tiles = sorted(
        (t for t in _tiles(manifest) if t.box().intersects(outline)), key=lambda t: t.uri
    )
    if not tiles:
        raise ValueError(f"no tile of {manifest_url} covers {bbox}")
    transform, shape = _grid(outline.bounds, tiles[0], cell_m)
    values = np.full(shape, np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for tile in tiles:
            _paste(values, transform, tile, band, missing, cell_m)
    _write_mosaic(dest, values, transform, crs, manifest_url, [t.uri for t in tiles])
    return dest


def mosaic_tiles(path: Path) -> list[str]:
    """The tile URIs a mosaic written by fetch_height_mosaic was built from."""
    with rasterio.open(path) as src:
        return json.loads(src.tags().get("tiles", "[]"))


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _height_band(manifest: dict[str, Any]) -> tuple[int, list[float]]:
    band = next(b for b in manifest["bands"] if b["id"] == HEIGHT_BAND)
    missing = band.get("missingData", {}).get("values", [])
    return int(band.get("tilesetBandIndex", 0)) + 1, [float(v) for v in missing]


def _manifest_crs(manifest: dict[str, Any]) -> str:
    crs = {tileset["crs"] for tileset in manifest["tilesets"]}
    if len(crs) != 1:
        raise ValueError(f"manifest tilesets use several CRS {sorted(crs)}")
    return crs.pop()


def _tiles(manifest: dict[str, Any]) -> list[Tile]:
    prefix = manifest["uriPrefix"].replace("gs://", "https://storage.googleapis.com/", 1)
    tiles = []
    for tileset in manifest["tilesets"]:
        for source in tileset["sources"]:
            a, size = source["affineTransform"], source["dimensions"]
            if a.get("shearX", 0) or a.get("shearY", 0) or a["scaleX"] != -a["scaleY"]:
                raise ValueError(f"tile {source['uris'][0]} is not north-up with square pixels")
            tiles.append(
                Tile(
                    uri=prefix + source["uris"][0],
                    x0=float(a["translateX"]),
                    y0=float(a["translateY"]),
                    scale=float(a["scaleX"]),
                    width=int(size["width"]),
                    height=int(size["height"]),
                )
            )
    return tiles


def _grid(
    bounds: tuple[float, float, float, float], anchor: Tile, cell_m: float
) -> tuple[Affine, tuple[int, int]]:
    """Cells of `cell_m` covering `bounds`, on the lattice through the anchor tile's corner."""
    min_x, min_y, max_x, max_y = bounds
    west = anchor.x0 + math.floor((min_x - anchor.x0) / cell_m) * cell_m
    north = anchor.y0 + math.ceil((max_y - anchor.y0) / cell_m) * cell_m
    cols = math.ceil((max_x - west) / cell_m)
    rows = math.ceil((north - min_y) / cell_m)
    return Affine(cell_m, 0.0, west, 0.0, -cell_m, north), (rows, cols)


def _paste(
    values: FloatArray,
    transform: Affine,
    tile: Tile,
    band: int,
    missing: list[float],
    cell_m: float,
) -> None:
    factor = round(cell_m / tile.scale)
    col0 = (tile.x0 - transform.c) / cell_m
    row0 = (transform.f - tile.y0) / cell_m
    exact = math.isclose(factor * tile.scale, cell_m) and col0.is_integer() and row0.is_integer()
    if not exact or tile.width % factor or tile.height % factor:
        raise ValueError(f"{tile.uri} does not align with the {cell_m} m grid")
    c0, r0 = int(col0), int(row0)
    rows, cols = values.shape
    top, bottom = max(r0, 0), min(r0 + tile.height // factor, rows)
    left, right = max(c0, 0), min(c0 + tile.width // factor, cols)
    if top >= bottom or left >= right:
        return
    window = Window.from_slices(
        ((top - r0) * factor, (bottom - r0) * factor), ((left - c0) * factor, (right - c0) * factor)
    )
    with rasterio.open(f"/vsicurl/{tile.uri}") as src:
        cells = src.read(band, window=window, out_shape=(bottom - top, right - left))
    cells = cells.astype(np.float32)
    cells[np.isin(cells, missing) | ~np.isfinite(cells)] = np.nan
    target = values[top:bottom, left:right]
    fill = np.isnan(target) & ~np.isnan(cells)
    target[fill] = cells[fill]


def _write_mosaic(
    dest: Path, values: FloatArray, transform: Affine, crs: str, manifest: str, tiles: list[str]
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    profile = {
        "driver": "GTiff",
        "width": values.shape[1],
        "height": values.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float("nan"),
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    try:
        with rasterio.open(partial, "w", **profile) as dst:
            dst.write(values, 1)
            dst.update_tags(manifest=manifest, tiles=json.dumps(tiles))
        partial.rename(dest)
    finally:
        partial.unlink(missing_ok=True)
