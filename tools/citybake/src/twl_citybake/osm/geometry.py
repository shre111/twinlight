from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import shapely
from shapely import Polygon

from twl_citybake.config import BBox
from twl_citybake.projection import LocalProjection

CLIP_POINTS_PER_SIDE = 64
MULTI_TYPE_IDS = (4, 5, 6, 7)

GeometryArray = npt.NDArray[np.object_]


def clip_polygon(bbox: BBox, proj: LocalProjection) -> Polygon:
    """The bbox in local metres, densified so its curved projected edges stay accurate."""
    t = np.linspace(0.0, 1.0, CLIP_POINTS_PER_SIDE, endpoint=False)
    w, s, e, n = bbox.west, bbox.south, bbox.east, bbox.north
    lon = np.concatenate([w + t * (e - w), np.full_like(t, e), e - t * (e - w), np.full_like(t, w)])
    lat = np.concatenate([np.full_like(t, s), s + t * (n - s), np.full_like(t, n), n - t * (n - s)])
    x, y = proj.forward(lon, lat)
    clip = shapely.Polygon(np.column_stack([x, y]))
    shapely.prepare(clip)
    return clip


def north_west(bbox: BBox, proj: LocalProjection) -> tuple[float, float]:
    x, y = proj.forward(bbox.west, bbox.north)
    return float(x), float(y)


def project(geoms: GeometryArray, proj: LocalProjection) -> GeometryArray:
    def forward(lonlat: np.ndarray) -> np.ndarray:
        return np.column_stack(proj.forward(lonlat[:, 0], lonlat[:, 1]))

    return np.asarray(shapely.transform(geoms, forward), dtype=object)


def from_wkb(wkb: Sequence[str], proj: LocalProjection) -> GeometryArray:
    """Parse osmium hex WKB (lon/lat) into projected geometries; unparseable entries are None."""
    if not wkb:
        return np.empty(0, dtype=object)
    return project(shapely.from_wkb(np.asarray(wkb, dtype=object), on_invalid="ignore"), proj)


def polygon_parts(geom: shapely.Geometry) -> list[Polygon]:
    """Non-empty polygons inside any (multi)polygon or collection, in GEOS order."""
    parts = shapely.get_parts(np.asarray([geom], dtype=object))
    while np.isin(shapely.get_type_id(parts), MULTI_TYPE_IDS).any():
        parts = shapely.get_parts(parts)
    return [p for p in parts if isinstance(p, Polygon) and not p.is_empty]


def oriented_boxes(
    geoms: GeometryArray,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Sides (w >= d) of each minimum rotated rectangle and the angle of its w side from +x,
    in [0, pi)."""
    offsets, coords = ring_arrays(shapely.oriented_envelope(geoms), np.float64)
    starts, ends = offsets[:-1].astype(np.intp), offsets[1:].astype(np.intp)
    p0, p1, p2 = (coords[np.minimum(starts + k, ends - 1)] for k in range(3))
    e1, e2 = p1 - p0, p2 - p1
    l1, l2 = np.hypot(e1[:, 0], e1[:, 1]), np.hypot(e2[:, 0], e2[:, 1])
    long_side = np.where((l1 >= l2)[:, None], e1, e2)
    rot = np.mod(np.arctan2(long_side[:, 1], long_side[:, 0]), np.pi).astype(np.float32)
    rot[rot >= np.float32(np.pi)] = 0.0
    return np.maximum(l1, l2).astype(np.float32), np.minimum(l1, l2).astype(np.float32), rot


def ring_arrays(
    rings: GeometryArray, dtype: type[np.floating] = np.float32
) -> tuple[npt.NDArray[np.uint32], np.ndarray]:
    """CSR encoding of closed rings or lines: offsets (n+1) into flat xy (m, 2)."""
    coords, index = shapely.get_coordinates(rings, return_index=True)
    counts = np.bincount(index, minlength=len(rings))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    return offsets, coords.astype(dtype).reshape(-1, 2)


def encode_polygons(
    polygons: Sequence[Polygon],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
    """Polygons as poly_offsets (p+1) into rings, ring_offsets (r+1) into xy; exterior first,
    counter-clockwise, holes clockwise."""
    oriented = shapely.orient_polygons(np.asarray(polygons, dtype=object), exterior_cw=False)
    rings, owner = shapely.get_rings(oriented, return_index=True)
    poly_offsets = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=len(polygons)))])
    ring_offsets, xy = ring_arrays(rings)
    return poly_offsets.astype(np.uint32), ring_offsets, xy
