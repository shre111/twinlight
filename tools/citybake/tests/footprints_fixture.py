import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import shapely
from pyproj import Transformer
from rasterio.transform import Affine, from_origin
from shapely import Geometry

from twl_citybake.projection import LocalProjection

UTM = "EPSG:32643"
CELL_M = 4.0
BLOCK = 3

GOOGLE, MICROSOFT = "google", "microsoft"

SQUARE = 20.0

G29_AT = (74.2, -800.0)
G31_AT = (-286.2, -700.0)
GFULL_AT = (-700.0, 500.0)
GNONE_AT = (200.0, 800.0)
GBOWTIE_AT = (-500.0, -500.0)
M_NONE_AT = (-400.0, 800.0)
M_STRIP_AT = (-283.0, -700.0)
M_DUP_OSM_AT = (-600.0, 500.0)
M_ZERO_AT = (100.0, -600.0)


@dataclass(frozen=True)
class CacheRow:
    source: str
    confidence: float | None
    geom: Geometry


def square(cx: float, cy: float, side: float = SQUARE) -> Geometry:
    return shapely.box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def bowtie(cx: float, cy: float) -> Geometry:
    h = SQUARE / 2
    return shapely.Polygon([(cx - h, cy - h), (cx + h, cy + h), (cx + h, cy - h), (cx - h, cy + h)])


def zero_area(cx: float, cy: float) -> Geometry:
    return shapely.Polygon([(cx, cy)] * 4)


def tiny_cache_rows() -> list[CacheRow]:
    """Google in cache order: none, 31% overlap with OSM 1011, bowtie, 29% overlap with OSM 1010,
    full copy of OSM 1003. Microsoft: copy of the kept Google none, none, zero area, a strip
    inside the dropped 31% Google clear of OSM, copy of OSM 1004."""
    strip = shapely.box(-288.0, -705.0, -278.0, -695.0)
    return [
        CacheRow(GOOGLE, 0.91, square(*GNONE_AT)),
        CacheRow(GOOGLE, 0.88, square(*G31_AT)),
        CacheRow(GOOGLE, 0.5, bowtie(*GBOWTIE_AT)),
        CacheRow(GOOGLE, 0.85, square(*G29_AT)),
        CacheRow(GOOGLE, 0.95, square(*GFULL_AT)),
        CacheRow(MICROSOFT, None, square(*GNONE_AT)),
        CacheRow(MICROSOFT, None, square(*M_NONE_AT)),
        CacheRow(MICROSOFT, None, zero_area(*M_ZERO_AT)),
        CacheRow(MICROSOFT, None, strip),
        CacheRow(MICROSOFT, None, square(*M_DUP_OSM_AT)),
    ]


def to_lonlat_wkb(geom: Geometry, proj: LocalProjection) -> bytes:
    def forward(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.column_stack(proj.inverse(xy[:, 0], xy[:, 1]))

    return bytes(shapely.to_wkb(shapely.transform(geom, forward)))


def write_vida_cache(path: Path, rows: Sequence[CacheRow], proj: LocalProjection) -> Path:
    import duckdb

    records = [
        (r.source, r.confidence, float(abs(shapely.area(r.geom))), to_lonlat_wkb(r.geom, proj))
        for r in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("CREATE TABLE cache (source VARCHAR, confidence FLOAT, area_m2 DOUBLE, wkb BLOB)")
    if records:
        con.executemany("INSERT INTO cache VALUES (?, ?, ?, ?)", records)
    con.execute(f"COPY (SELECT * FROM cache ORDER BY rowid) TO '{path}' (FORMAT PARQUET)")
    con.close()
    return path


@dataclass
class HeightGrid:
    values: npt.NDArray[np.float32]
    transform: Affine
    proj: LocalProjection

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        easting, northing = utm_of(self.proj, x, y)
        col = int(np.floor((easting - self.transform.c) / CELL_M))
        row = int(np.floor((self.transform.f - northing) / CELL_M))
        return row, col

    def paint(self, x: float, y: float, pattern: npt.ArrayLike) -> None:
        block = np.asarray(pattern, np.float32).reshape(BLOCK, BLOCK)
        row, col = self.cell_of(x, y)
        self.values[row - 1 : row + 2, col - 1 : col + 2] = block

    def paint_cell(self, x: float, y: float, value: float) -> tuple[int, int]:
        row, col = self.cell_of(x, y)
        self.values[row, col] = value
        return row, col

    def cell_centre_local(self, row: int, col: int) -> tuple[float, float]:
        easting = self.transform.c + (col + 0.5) * CELL_M
        northing = self.transform.f - (row + 0.5) * CELL_M
        lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(
            easting, northing
        )
        x, y = self.proj.forward(lon, lat)
        return float(x), float(y)

    def write(self, path: Path) -> Path:
        import rasterio

        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=self.values.shape[0],
            width=self.values.shape[1],
            count=1,
            dtype="float32",
            crs=UTM,
            transform=self.transform,
            nodata=float("nan"),
            compress="deflate",
        ) as dst:
            dst.write(self.values, 1)
        return path


def utm_of(proj: LocalProjection, x: float, y: float) -> tuple[float, float]:
    lon, lat = proj.inverse(x, y)
    easting, northing = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(lon, lat)
    return float(easting), float(northing)


def empty_height_grid(proj: LocalProjection, half_extent_m: float = 1250.0) -> HeightGrid:
    corners = [(sx * half_extent_m, sy * half_extent_m) for sx in (-1, 1) for sy in (-1, 1)]
    utm = np.asarray([utm_of(proj, x, y) for x, y in corners])
    west = np.floor(utm[:, 0].min() / CELL_M) * CELL_M
    north = np.ceil(utm[:, 1].max() / CELL_M) * CELL_M
    cols = int(np.ceil((utm[:, 0].max() - west) / CELL_M))
    rows = int(np.ceil((north - utm[:, 1].min()) / CELL_M))
    return HeightGrid(
        values=np.full((rows, cols), np.nan, np.float32),
        transform=from_origin(west, north, CELL_M, CELL_M),
        proj=proj,
    )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
