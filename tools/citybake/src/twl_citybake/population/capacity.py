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
    by_kind = np.asarray([share[k] for k in BuildingKind], np.float64)
    floor_m2 = np.asarray(area_m2, np.float64) * storeys(height_m)
    persons = np.floor(floor_m2 * by_kind[np.asarray(kind, np.intp)] / min_floor_m2_per_person)
    return np.maximum(persons, 0).astype(np.int64)
