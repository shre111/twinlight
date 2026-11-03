from pathlib import Path

import numpy as np
import numpy.typing as npt

from twl_citybake.config import BBox
from twl_citybake.projection import LocalProjection
from twl_citybake.store import write_json, write_table

ARC_3 = 3 / 3600
NODATA = -99999.0

CENSUS_SIZES = (1, 2, 3, 4, 5, 6, 8, 12, 17)
CENSUS_COUNTS = (37094, 97013, 176374, 310595, 229831, 143502, 158773, 18359, 4216)
RESIDENTIAL_SHARE = {
    "unknown": 0.9,
    "residential": 1.0,
    "commercial": 0.2,
    "industrial": 0.05,
    "civic": 0.1,
    "religious": 0.0,
    "other": 0.0,
}


def write_geotiff(
    path: Path,
    values: npt.ArrayLike,
    west: float,
    north: float,
    cell: float = ARC_3,
    nodata: float | None = NODATA,
) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    data = np.asarray(values, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, cell, cell),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def synthetic_households_input(
    seed: int = 11, n_buildings: int = 400, n_districts: int = 4
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    capacity = rng.integers(30, 400, n_buildings).astype(np.int64)
    district = np.sort(rng.integers(0, n_districts, n_buildings)).astype(np.uint16)
    district[:n_districts] = np.arange(n_districts)
    district = np.sort(district)
    placed = capacity * rng.uniform(0.3, 0.95, n_buildings)
    marginal = np.bincount(district, weights=placed, minlength=n_districts)
    return {"capacity": capacity, "district": district, "placed": placed, "marginal": marginal}


def synthetic_buildings(n: int, seed: int = 3, n_districts: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "x": rng.uniform(-500, 500, n),
        "y": rng.uniform(-500, 500, n),
        "w": rng.uniform(8, 40, n).astype(np.float32),
        "d": rng.uniform(6, 30, n).astype(np.float32),
        "rot": rng.uniform(0, np.pi, n).astype(np.float32),
        "height_m": rng.uniform(3, 40, n).astype(np.float32),
        "district": np.sort(rng.integers(0, n_districts, n)).astype(np.uint16),
        "bank": rng.integers(0, 2, n).astype(np.uint8),
    }


def write_world(
    bbox: BBox, raw_tif: Path, layers_dir: Path, block: int = 36, seed: int = 5
) -> dict[str, float]:
    """A raster covering the bbox, populated in a centred block of cells, and OSM layers with
    buildings in the block except its outer ring (those cells hold people but no buildings)."""
    rng = np.random.default_rng(seed)
    cols = int(np.ceil((bbox.east - bbox.west) / ARC_3)) + 2
    rows = int(np.ceil((bbox.north - bbox.south) / ARC_3)) + 2
    west = bbox.west - ARC_3
    north = bbox.north + ARC_3
    values = np.full((rows, cols), NODATA, np.float32)
    r0, c0 = rows // 2 - block // 2, cols // 2 - block // 2
    people = rng.integers(40, 300, (block, block)).astype(np.float32)
    values[r0 : r0 + block, c0 : c0 + block] = people
    write_geotiff(raw_tif, values, west, north)

    projection = LocalProjection.for_bbox(bbox)
    xs: list[float] = []
    ys: list[float] = []
    for r in range(r0 + 1, r0 + block - 1):
        for c in range(c0 + 1, c0 + block - 1):
            x, y = projection.forward(west + (c + 0.5) * ARC_3, north - (r + 0.5) * ARC_3)
            for dx, dy in ((-20.0, -20.0), (0.0, 20.0), (20.0, 0.0)):
                xs.append(float(x) + dx)
                ys.append(float(y) + dy)
    n = len(xs)
    x_arr = np.asarray(xs)
    write_table(
        layers_dir / "buildings",
        {
            "x": x_arr,
            "y": np.asarray(ys),
            "w": np.full(n, 15.0, np.float32),
            "d": np.full(n, 15.0, np.float32),
            "rot": np.zeros(n, np.float32),
            "area_m2": np.full(n, 225.0, np.float32),
            "height_m": np.full(n, 12.8, np.float32),
            "kind": np.ones(n, np.uint8),
            "district": (x_arr > 0).astype(np.uint16),
            "bank": np.zeros(n, np.uint8),
        },
    )
    write_json(
        layers_dir / "districts.json", [{"id": 0, "name": "West"}, {"id": 1, "name": "East"}]
    )
    return {"people_in_block": float(people.sum()), "buildings": float(n)}
