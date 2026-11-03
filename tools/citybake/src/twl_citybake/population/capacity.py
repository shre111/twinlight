from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from twl_citybake.osm.tags import LEVEL_HEIGHT_M, BuildingKind


def storeys(height_m: npt.ArrayLike) -> npt.NDArray[np.int64]:
    levels = np.round(np.asarray(height_m, np.float64) / LEVEL_HEIGHT_M)
    return np.maximum(1, levels).astype(np.int64)


def residential_capacity(
    area_m2: npt.ArrayLike,
    height_m: npt.ArrayLike,
    kind: npt.ArrayLike,
    share: Mapping[BuildingKind, float],
    min_floor_m2_per_person: float,
) -> npt.NDArray[np.int64]:
    """Most people a building can house: its housing floor area at the densest plausible
    crowding."""
    if not min_floor_m2_per_person > 0:
        raise ValueError(f"min_floor_m2_per_person must be positive, got {min_floor_m2_per_person}")
    if missing := [k.name for k in BuildingKind if k not in share]:
        raise ValueError(f"housing share missing for {missing}")
    area = np.asarray(area_m2, np.float64)
    height = np.asarray(height_m, np.float64)
    kinds = np.asarray(kind, np.intp)
    if not (np.isfinite(area).all() and np.isfinite(height).all()):
        raise ValueError("footprint areas and heights must be finite")
    if ((kinds < 0) | (kinds >= len(BuildingKind))).any():
        raise ValueError("building kind out of range")
    by_kind = np.asarray([share[k] for k in BuildingKind], np.float64)
    floor_m2 = area * storeys(height)
    persons = np.floor(floor_m2 * by_kind[kinds] / min_floor_m2_per_person)
    return np.maximum(persons, 0).astype(np.int64)
