from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area
from shapely import Polygon

from twl_citybake.osm.geometry import encode_polygons, from_wkb, polygon_parts
from twl_citybake.projection import LocalProjection

MIN_LAKE_M2 = 10_000.0


class WaterKind(IntEnum):
    RIVER = 0
    LAKE = 1


@dataclass(frozen=True)
class Water:
    columns: dict[str, np.ndarray]
    river_area_m2: float
    lakes: int
    lake_area_m2: float


@dataclass
class WaterCollector:
    river: list[tuple[int, str]] = field(default_factory=list)
    lakes: list[tuple[int, str]] = field(default_factory=list)

    def add(self, area: Area, factory: WKBFactory) -> None:
        tags = area.tags
        is_river = tags.get("water") == "river" or tags.get("waterway") == "riverbank"
        if not is_river and tags.get("natural") != "water":
            return
        try:
            wkb = factory.create_multipolygon(area)
        except RuntimeError:
            return
        (self.river if is_river else self.lakes).append((area.id, wkb))

    def finish(self, proj: LocalProjection, clip: Polygon) -> Water:
        river = shapely.intersection(shapely.union_all(_valid(self.river, proj)), clip)
        river_parts = _ordered(polygon_parts(river))
        lakes = [
            clipped
            for lake in _valid(self.lakes, proj)
            if shapely.area(clipped := shapely.intersection(lake, clip)) >= MIN_LAKE_M2
        ]
        lake_parts = _ordered([part for lake in lakes for part in polygon_parts(lake)])
        polygons = river_parts + lake_parts
        poly_offsets, ring_offsets, xy = encode_polygons(polygons)
        kinds = [WaterKind.RIVER] * len(river_parts) + [WaterKind.LAKE] * len(lake_parts)
        return Water(
            columns={
                "poly_kind": np.asarray(kinds, np.uint8),
                "poly_offsets": poly_offsets,
                "ring_offsets": ring_offsets,
                "xy": xy,
            },
            river_area_m2=float(shapely.area(river)),
            lakes=len(lake_parts),
            lake_area_m2=float(sum(p.area for p in lake_parts)),
        )


def _valid(records: list[tuple[int, str]], proj: LocalProjection) -> np.ndarray:
    ordered = sorted(records)
    geoms = from_wkb([wkb for _, wkb in ordered], proj)
    geoms = geoms[~shapely.is_missing(geoms)]
    return np.asarray(shapely.make_valid(geoms), dtype=object)


def _ordered(polygons: list[Polygon]) -> list[Polygon]:
    """Largest first; ties broken by the first exterior vertex."""

    def key(polygon: Polygon) -> tuple[float, float, float]:
        x, y = polygon.exterior.coords[0]
        return -polygon.area, x, y

    return sorted(polygons, key=key)
