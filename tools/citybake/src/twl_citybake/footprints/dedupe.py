from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import shapely

from twl_citybake.osm.geometry import GeometryArray

BoolArray = npt.NDArray[np.bool_]


def valid_footprints(geoms: GeometryArray) -> BoolArray:
    """Valid geometries with positive area; missing, empty, self-crossing and degenerate ones
    fail."""
    return np.asarray(shapely.is_valid(geoms) & (shapely.area(geoms) > 0), np.bool_)


def dedupe(higher: GeometryArray, lower: GeometryArray, threshold: float) -> BoolArray:
    """Keep mask for `lower`: valid footprints whose area covered by the valid `higher`
    footprints is at most `threshold` of their own."""
    keep = valid_footprints(lower)
    higher = higher[valid_footprints(higher)]
    candidates = np.flatnonzero(keep)
    if not len(higher) or not len(candidates):
        return keep
    pairs = shapely.STRtree(higher).query(lower[candidates], predicate="intersects")
    owner = candidates[pairs[0]]
    shared = shapely.area(shapely.intersection(lower[owner], higher[pairs[1]]))
    overlap = np.bincount(owner, weights=shared, minlength=len(lower))
    limit = threshold * shapely.area(lower)
    # summed pairs double-count where higher footprints overlap each other; settle those exactly
    several = np.bincount(owner, minlength=len(lower)) > 1
    for i in np.flatnonzero(several & (overlap > limit)):
        union = shapely.union_all(higher[pairs[1][owner == i]])
        overlap[i] = shapely.area(shapely.intersection(lower[i], union))
    keep[candidates] = overlap[candidates] <= limit[candidates]
    return keep


def dedupe_tiers(
    top: GeometryArray, tiers: Sequence[GeometryArray], threshold: float
) -> list[BoolArray]:
    """Keep masks for each tier in priority order, each checked against `top` and the kept
    footprints of the tiers before it."""
    masks: list[BoolArray] = []
    higher = top
    for tier in tiers:
        keep = dedupe(higher, tier, threshold)
        masks.append(keep)
        higher = np.concatenate([higher, tier[keep]])
    return masks
