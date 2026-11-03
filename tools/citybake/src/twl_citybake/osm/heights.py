import numpy as np
import numpy.typing as npt

from twl_citybake.osm.tags import LEVEL_HEIGHT_M, BuildingKind, HeightSource

MIN_HEIGHT_SAMPLES = 20
FALLBACK_LEVELS = 2.0


def default_levels(
    district: np.ndarray, kind: np.ndarray, levels: np.ndarray
) -> npt.NDArray[np.float64]:
    """Storeys for buildings without tagged heights, from the levels of LEVELS-sourced buildings
    (`levels` is NaN elsewhere): the median of the building's district and kind when that group
    has MIN_HEIGHT_SAMPLES, else its kind city-wide, else the whole city, else 2.
    Medians are rounded to half a level."""
    sampled = ~np.isnan(levels)
    city = float(_rounded(np.median(levels[sampled]))) if sampled.any() else FALLBACK_LEVELS
    result = np.full(len(levels), city, np.float64)
    kind_key = kind.astype(np.int64)
    district_kind_key = district.astype(np.int64) * len(BuildingKind) + kind_key
    for key in (kind_key, district_kind_key):
        groups, medians = _group_medians(key[sampled], levels[sampled])
        at = np.searchsorted(groups, key)
        found = at < len(groups)
        found[found] = groups[at[found]] == key[found]
        result[found] = _rounded(medians[at[found]])
    return result


def final_heights(
    tagged_height_m: np.ndarray,
    height_source: np.ndarray,
    district: np.ndarray,
    kind: np.ndarray,
    levels: np.ndarray,
) -> npt.NDArray[np.float32]:
    defaults = LEVEL_HEIGHT_M * default_levels(district, kind, levels)
    untagged = height_source == HeightSource.DEFAULT
    return np.where(untagged, defaults, tagged_height_m).astype(np.float32)


def _group_medians(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Medians of the groups with at least MIN_HEIGHT_SAMPLES values, equal to np.median per
    group: the mean of the two middle values once sorted."""
    order = np.lexsort((values, keys))
    keys, values = keys[order], values[order]
    groups, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    medians = (values[starts + (counts - 1) // 2] + values[starts + counts // 2]) / 2
    enough = counts >= MIN_HEIGHT_SAMPLES
    return groups[enough], medians[enough]


def _rounded(levels: npt.ArrayLike) -> np.ndarray:
    return np.round(np.asarray(levels, np.float64) * 2) / 2
