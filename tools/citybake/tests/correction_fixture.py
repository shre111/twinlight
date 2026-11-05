import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twl_citybake.config import PopulationCorrection
from twl_citybake.population.correction import Correction, correct_raster
from twl_citybake.population.raster import PopGrid

CELL = 0.001
WEST = 72.0
NORTH = 23.0
DEFAULT_EDGES = {
    "height_edges": (2.0, 3.0, 5.0, 8.0),
    "footprint_edges": (100.0, 250.0, 600.0, 1500.0),
    "coverage_edges": (0.10, 0.25, 0.45),
}


@dataclass(frozen=True)
class Bldg:
    row: int
    col: int
    footprint: float
    floors: float
    housing: float


def make_grid(values: npt.ArrayLike) -> PopGrid:
    return PopGrid(np.asarray(values, np.float64), west=WEST, north=NORTH, dlon=CELL, dlat=CELL)


def cell_area(grid: PopGrid) -> float:
    rows = grid.values.shape[0]
    mid_lat = grid.north - rows * grid.dlat / 2
    return grid.dlon * 111320 * float(np.cos(np.radians(mid_lat))) * (grid.dlat * 110574)


def make_config(**overrides: Any) -> PopulationCorrection:
    kwargs: dict[str, Any] = {
        "weight": 0.25,
        "weight_range": (0.0, 1.0),
        "weight_evidence": "chosen after seeing the ward data",
        "reference_polygons": ("relation/1",),
        "fit_min_m2_per_person": 0.1,
    }
    kwargs.update(overrides)
    return PopulationCorrection(**kwargs)


def run_correction(
    values: npt.ArrayLike,
    buildings: Sequence[Bldg],
    in_reference: npt.ArrayLike,
    extra_lonlat: Sequence[tuple[float, float, Bldg]] = (),
    **overrides: Any,
) -> Correction:
    grid = make_grid(values)
    lon_c, lat_c = grid.centres()
    lon = [float(lon_c[b.row, b.col]) for b in buildings]
    lat = [float(lat_c[b.row, b.col]) for b in buildings]
    parts = list(buildings)
    for x, y, b in extra_lonlat:
        lon.append(x)
        lat.append(y)
        parts.append(b)
    return correct_raster(
        grid,
        np.asarray(lon),
        np.asarray(lat),
        np.asarray([b.footprint for b in parts], np.float64),
        np.asarray([b.floors for b in parts], np.float64),
        np.asarray([b.housing for b in parts], np.float64),
        np.asarray(in_reference, np.bool_),
        make_config(**overrides),
    )


def table_digest(tables: dict[str, dict[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tables):
        for column in sorted(tables[name]):
            array = tables[name][column]
            if array.dtype.kind in "iu":
                digest.update(f"{name}.{column}:{array.dtype}:{array.shape}".encode())
                digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_stage_tables(out: Path) -> dict[str, dict[str, np.ndarray]]:
    from twl_citybake.store import read_table

    return {
        name: {k: np.asarray(v) for k, v in read_table(out / name).items()}
        for name in ("building_population", "households", "residents")
    }
