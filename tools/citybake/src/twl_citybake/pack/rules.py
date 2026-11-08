"""The D-031 pack-time rules as pure functions over baked tables (ARCHITECTURE.md section 5)."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

from twl_citybake.osm.banks import EAST
from twl_citybake.osm.geometry import GeometryArray, polygon_parts, ring_polygons
from twl_citybake.osm.tags import RoadClass
from twl_pack import RiverColumns
from twl_pack.format import FLOOR_HEIGHT_M

STREET, ARTERIAL, BRIDGE = 0, 1, 2
FLOOR_TOLERANCE_M = 0.01
_ON_FLOOR_EPSILON = 1e-4
MAX_FLOOR = 255


class PackStageError(ValueError):
    """A known condition that stops the pack stage."""


@dataclass(frozen=True, eq=False)
class Rehomed:
    """Per resident, in resident order. `building` indexes all baked buildings."""

    building: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    floor: NDArray[np.uint8]
    repointed: int
    moved: int
    floors_lowered: int
    repoint_distance_m: NDArray[np.float64]


def footprint_polygons(offsets: np.ndarray, xy: np.ndarray) -> GeometryArray:
    return ring_polygons(offsets, xy)


def isolated_small(
    polygons: ArrayLike, area_m2: ArrayLike, min_area_m2: float, isolated_m: float
) -> NDArray[np.bool_]:
    """Under `min_area_m2` with no other polygon within `isolated_m`, a neighbour at exactly that
    distance counting as within."""
    geoms = np.asarray(polygons, dtype=object)
    small = np.flatnonzero(np.asarray(area_m2, np.float64) < min_area_m2)
    isolated = np.zeros(len(geoms), np.bool_)
    if not small.size:
        return isolated
    query, found = shapely.STRtree(geoms).query(
        geoms[small], predicate="dwithin", distance=isolated_m
    )
    isolated[small] = True
    isolated[small[query[small[query] != found]]] = False
    return isolated


def water_geometry(water: Mapping[str, np.ndarray]) -> shapely.Geometry:
    """The union of the water polygons. The stored rings are `f32`, whose rounding can make a valid
    ring self-touch, so each polygon is repaired and only polygonal parts stay."""
    xy = water["xy"].astype(np.float64)
    rings = [xy[a:b] for a, b in pairwise(water["ring_offsets"].astype(np.intp))]
    polygons = [
        shapely.Polygon(rings[first], rings[first + 1 : end])
        for first, end in pairwise(water["poly_offsets"].astype(np.intp))
    ]
    merged = shapely.union_all(shapely.make_valid(np.asarray(polygons, dtype=object)))
    return shapely.multipolygons(polygon_parts(merged))


def in_water(geometry: shapely.Geometry, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
    shapely.prepare(geometry)
    return shapely.contains_xy(geometry, np.asarray(x, np.float64), np.asarray(y, np.float64))


def floors_from_z(z: ArrayLike) -> NDArray[np.uint8]:
    heights = np.asarray(z, np.float64)
    floor = np.rint(heights / FLOOR_HEIGHT_M - 0.5)
    off = ~(np.abs((floor + 0.5) * FLOOR_HEIGHT_M - heights) <= FLOOR_TOLERANCE_M)
    off |= (floor < 0) | (floor > MAX_FLOOR)
    if off.any():
        raise PackStageError(
            f"{np.count_nonzero(off)} heights are not on a mid-floor, the first is "
            f"z {heights[off][0]}"
        )
    return floor.astype(np.uint8)


def top_floor(height_m: ArrayLike) -> NDArray[np.int64]:
    heights = np.asarray(height_m, np.float64)
    # A height on a mid-floor, such as 4.8 m or an f32 14.4 m, divides to a hair under a whole
    # number; the margin, in floors, covers f32 rounding of baked heights.
    floor = np.floor(heights / FLOOR_HEIGHT_M - 0.5 + _ON_FLOOR_EPSILON)
    return np.maximum(floor, 0).astype(np.int64)


def rehome_residents(
    home: ArrayLike,
    floor: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    standing_in_water: ArrayLike,
    kept: ArrayLike,
    building_x: ArrayLike,
    building_y: ArrayLike,
    height_m: ArrayLike,
) -> Rehomed:
    """Each resident's building after the rule, always a kept one. Only a resident standing in
    water moves, to the centre of that building, on the highest floor it fits if theirs does not."""
    keep = np.asarray(kept, np.bool_)
    if not keep.any():
        raise PackStageError("no building is kept, so residents have nowhere to live")
    was_home = np.asarray(home, np.int64)
    centre_x = np.asarray(building_x, np.float64)
    centre_y = np.asarray(building_y, np.float64)
    wet = np.asarray(standing_in_water, np.bool_)

    building = _nearest_kept(was_home, keep, centre_x, centre_y)
    repointed = building != was_home
    new_x = np.array(x, np.float64)
    new_y = np.array(y, np.float64)
    new_x[wet], new_y[wet] = centre_x[building[wet]], centre_y[building[wet]]

    new_floor = np.array(floor, np.uint8)
    fits = top_floor(np.asarray(height_m)[building[wet]])
    lowered = new_floor[wet] > fits
    new_floor[wet] = np.minimum(new_floor[wet], fits)
    return Rehomed(
        building=building,
        x=new_x,
        y=new_y,
        floor=new_floor,
        repointed=int(np.count_nonzero(repointed)),
        moved=int(np.count_nonzero(wet)),
        floors_lowered=int(np.count_nonzero(lowered)),
        repoint_distance_m=np.hypot(
            centre_x[building[repointed]] - centre_x[was_home[repointed]],
            centre_y[building[repointed]] - centre_y[was_home[repointed]],
        ),
    )


def _nearest_kept(
    home: NDArray[np.int64],
    keep: NDArray[np.bool_],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.int64]:
    building = home.copy()
    dropped = ~keep[home]
    if not dropped.any():
        return building
    homes, resident_home = np.unique(home[dropped], return_inverse=True)
    rows = np.flatnonzero(keep)
    tree = KDTree(np.column_stack([x[rows], y[rows]]))
    nearest = np.asarray(tree.query(np.column_stack([x[homes], y[homes]]))[1], np.intp)
    building[dropped] = rows[nearest][resident_home]
    return building


def pack_road_class(
    road_class: ArrayLike, bridge: ArrayLike, arterial: Collection[str | int]
) -> NDArray[np.uint8]:
    """`arterial` holds lowercase `RoadClass` names or their values."""
    codes = [_road_code(name) for name in arterial]
    is_arterial = np.isin(np.asarray(road_class), np.asarray(codes, np.int64))
    classes = np.where(np.asarray(bridge).astype(np.bool_), BRIDGE, is_arterial * ARTERIAL)
    return classes.astype(np.uint8)


def _road_code(name: str | int) -> int:
    if not isinstance(name, str):
        return int(name)
    try:
        return int(RoadClass[name.upper()])
    except KeyError:
        raise ValueError(f"unknown road class {name!r}") from None


def river_columns(water: Mapping[str, np.ndarray]) -> RiverColumns:
    xy = water["xy"].reshape(-1, 2)
    offsets = water["ring_offsets"].astype(np.int64)
    starts, ends = offsets[:-1], offsets[1:]
    closed = (ends - starts > 1) & (xy[starts] == xy[ends - 1]).all(axis=1)
    keep = np.ones(len(xy), np.bool_)
    keep[ends[closed] - 1] = False
    points = ends - starts - closed
    return RiverColumns(
        offsets=np.concatenate(([0], np.cumsum(points))).astype(np.uint32),
        x=xy[keep, 0],
        y=xy[keep, 1],
        kind=water["poly_kind"],
        rings=np.diff(water["poly_offsets"].astype(np.int64)),
    )


def pack_districts(districts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": district["id"],
            "name": district["name"],
            "name_local": district["name_local"],
            "x": district["seed"][0],
            "y": district["seed"][1],
            "east": bool(district["bank"] == EAST),
            "polygons": [_open_ring(ring) for ring in district["polygon"]],
        }
        for district in districts
    ]


def _open_ring(ring: Sequence[Sequence[float]]) -> list[Sequence[float]]:
    return list(ring[:-1]) if len(ring) > 1 and ring[0] == ring[-1] else list(ring)


def attributions(*layer_metas: Mapping[str, Any]) -> list[str]:
    flat: list[str] = []
    for meta in layer_metas:
        credit = meta.get("attribution")
        if isinstance(credit, str):
            flat.append(credit)
        elif credit:
            flat.extend(credit)
    return flat
