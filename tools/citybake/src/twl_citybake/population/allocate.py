from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twl_citybake.population.raster import PopGrid

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Allocation:
    """`target` and `placed` per building, `marginal` and `shortfall` per district."""

    target: FloatArray
    placed: FloatArray
    marginal: FloatArray
    shortfall: FloatArray
    people_without_buildings: float
    cells_with_people: int
    cells_without_buildings: int


def allocate(
    grid: PopGrid,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    capacity: npt.ArrayLike,
    district: npt.ArrayLike,
    n_districts: int,
) -> Allocation:
    """Share each cell's people among the buildings whose centroid lies in it, by capacity;
    then spill what a building cannot hold onto spare capacity across its district."""
    cap = np.asarray(capacity, np.float64)
    district = np.asarray(district, np.intp)
    if not (len(np.asarray(lon)) == len(np.asarray(lat)) == len(cap) == len(district)):
        raise ValueError("lon, lat, capacity and district must have one value per building")
    if (cap < 0).any() or not np.isfinite(cap).all():
        raise ValueError("capacity must be finite and non-negative")
    if len(district) and (district.min() < 0 or district.max() >= n_districts):
        raise ValueError(f"district ids must be in [0, {n_districts})")
    row, col = grid.cell_of(lon, lat)
    b = np.flatnonzero(row >= 0)
    cell = row[b] * grid.values.shape[1] + col[b]
    people = grid.values.ravel()
    cell_capacity = np.bincount(cell, weights=cap[b], minlength=people.size)
    housed = cell_capacity[cell] > 0
    b, cell = b[housed], cell[housed]
    target = np.zeros(len(cap))
    target[b] = people[cell] * cap[b] / cell_capacity[cell]
    placed, shortfall = _spill(target, cap, district, n_districts)
    unhoused = (people > 0) & (cell_capacity == 0)
    return Allocation(
        target=target,
        placed=placed,
        marginal=np.bincount(district, weights=target, minlength=n_districts),
        shortfall=shortfall,
        people_without_buildings=float(people[unhoused].sum()),
        cells_with_people=int((people > 0).sum()),
        cells_without_buildings=int(unhoused.sum()),
    )


def _spill(
    target: FloatArray, cap: FloatArray, district: npt.NDArray[np.intp], n_districts: int
) -> tuple[FloatArray, FloatArray]:
    placed = np.minimum(target, cap)
    spare_b = cap - placed
    excess = np.bincount(district, weights=target - placed, minlength=n_districts)
    spare = np.bincount(district, weights=spare_b, minlength=n_districts)
    full = (spare <= excess)[district]
    fill = np.divide(excess, spare, out=np.zeros(n_districts), where=spare > 0)[district]
    placed = np.where(full, cap, np.minimum(placed + spare_b * fill, cap))
    return placed, np.maximum(excess - spare, 0.0)
