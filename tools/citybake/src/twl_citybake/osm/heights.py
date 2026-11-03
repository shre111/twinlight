import numpy as np
import numpy.typing as npt

from twl_citybake.osm.tags import LEVEL_HEIGHT_M, BuildingKind, HeightSource

MIN_HEIGHT_SAMPLES = 20
FALLBACK_LEVELS = 2.0
FOOTPRINT_BANDS_M2 = (250.0, 700.0)


def default_levels(
    district: np.ndarray, kind: np.ndarray, area_m2: np.ndarray, levels: np.ndarray
) -> npt.NDArray[np.float64]:
    """Storeys for buildings without tagged heights, from the levels of LEVELS-sourced buildings
    (`levels` is NaN elsewhere). Starting from the city median (else 2), each finer group that
    has MIN_HEIGHT_SAMPLES overrides the coarser one: kind, district and kind, kind and footprint
    band, then district, kind and band. Medians are rounded to half a level.
    Footprint bands matter because tagged storeys rise with footprint (residential in Ahmedabad:
    about 2 below 250 m2, 5 to 6 up to 700 m2, 8 or more above)."""
    sampled = ~np.isnan(levels)
    city = float(_rounded(np.median(levels[sampled]))) if sampled.any() else FALLBACK_LEVELS
    result = np.full(len(levels), city, np.float64)
    bands = len(FOOTPRINT_BANDS_M2) + 1
    band = np.searchsorted(FOOTPRINT_BANDS_M2, area_m2, side="right").astype(np.int64)
    kind_key = kind.astype(np.int64)
    district_kind = district.astype(np.int64) * len(BuildingKind) + kind_key
    for key in (kind_key, district_kind, kind_key * bands + band, district_kind * bands + band):
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
    area_m2: np.ndarray,
    levels: np.ndarray,
) -> npt.NDArray[np.float32]:
    defaults = LEVEL_HEIGHT_M * default_levels(district, kind, area_m2, levels)
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
