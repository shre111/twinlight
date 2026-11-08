"""Footprint squares and water tables in the shape the baked stage tables use."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray


def closed(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [*points, points[0]]


def water_table_of(
    polygons: Sequence[tuple[int, Sequence[Sequence[tuple[float, float]]]]],
) -> dict[str, NDArray[Any]]:
    """`polygons` are (kind, rings) with rings exactly as they are to be stored."""
    kinds = [p[0] for p in polygons]
    rings = [ring for _, rs in polygons for ring in rs]
    poly_offsets = np.concatenate([[0], np.cumsum([len(rs) for _, rs in polygons])])
    ring_offsets = np.concatenate([[0], np.cumsum([len(r) for r in rings])])
    xy = np.array([p for ring in rings for p in ring], np.float32).reshape(-1, 2)
    return {
        "poly_kind": np.array(kinds, np.uint8),
        "poly_offsets": poly_offsets.astype(np.uint32),
        "ring_offsets": ring_offsets.astype(np.uint32),
        "xy": xy,
    }


def square_ring(x: float, y: float, side: float) -> list[tuple[float, float]]:
    h = side / 2
    return closed([(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)])


def square(x: float, y: float, side: float) -> shapely.Polygon:
    return shapely.Polygon(square_ring(x, y, side))
