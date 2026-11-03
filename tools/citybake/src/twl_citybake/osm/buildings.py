from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area
from shapely import Polygon

from twl_citybake.osm.geometry import from_wkb, oriented_boxes, ring_arrays
from twl_citybake.osm.tags import HeightSource, building_kind, parse_levels, tagged_height
from twl_citybake.projection import LocalProjection

HEIGHT_KEYS = ("height", "building:height", "building:levels")
OSM_WAY, OSM_RELATION = 0, 1


@dataclass(frozen=True)
class Buildings:
    """Building footprints kept by the clip, sorted by (osm_type, osm_id), before defaults."""

    osm_type: npt.NDArray[np.uint8]
    osm_id: npt.NDArray[np.int64]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    w: npt.NDArray[np.float32]
    d: npt.NDArray[np.float32]
    rot: npt.NDArray[np.float32]
    area_m2: npt.NDArray[np.float32]
    kind: npt.NDArray[np.uint8]
    tagged_height_m: npt.NDArray[np.float64]
    height_source: npt.NDArray[np.uint8]
    levels: npt.NDArray[np.float64]
    rings: dict[str, np.ndarray]
    skipped_invalid: int

    def __len__(self) -> int:
        return len(self.osm_id)

    def table(
        self, height_m: np.ndarray, district: np.ndarray, bank: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {
            "osm_type": self.osm_type,
            "osm_id": self.osm_id,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "d": self.d,
            "rot": self.rot,
            "area_m2": self.area_m2,
            "height_m": np.asarray(height_m, np.float32),
            "height_source": self.height_source,
            "kind": self.kind,
            "district": np.asarray(district, np.uint16),
            "bank": np.asarray(bank, np.uint8),
        }


@dataclass
class BuildingCollector:
    wkb: list[str] = field(default_factory=list)
    osm_type: list[int] = field(default_factory=list)
    osm_id: list[int] = field(default_factory=list)
    kind: list[int] = field(default_factory=list)
    height_m: list[float] = field(default_factory=list)
    source: list[int] = field(default_factory=list)
    levels: list[float] = field(default_factory=list)
    invalid: int = 0

    def add(self, area: Area, factory: WKBFactory) -> None:
        value = area.tags.get("building")
        if value is None or (kind := building_kind(value)) is None:
            return
        try:
            wkb = factory.create_multipolygon(area)
        except RuntimeError:
            self.invalid += 1  # osmium hands over broken rings as empty areas
            return
        tags = {key: area.tags[key] for key in HEIGHT_KEYS if key in area.tags}
        height, source = tagged_height(tags) or (np.nan, HeightSource.DEFAULT)
        levels = (
            parse_levels(tags.get("building:levels")) if source == HeightSource.LEVELS else None
        )
        self.wkb.append(wkb)
        self.osm_type.append(OSM_WAY if area.from_way() else OSM_RELATION)
        self.osm_id.append(area.orig_id())
        self.kind.append(kind)
        self.height_m.append(height)
        self.source.append(source)
        self.levels.append(np.nan if levels is None else levels)

    def finish(self, proj: LocalProjection, clip: Polygon) -> Buildings:
        osm_type = np.asarray(self.osm_type, np.uint8)
        osm_id = np.asarray(self.osm_id, np.int64)
        order = np.lexsort((osm_id, osm_type))
        geoms = from_wkb([self.wkb[i] for i in order], proj)
        valid = shapely.is_valid(geoms) & (shapely.area(geoms) > 0)
        centroids = shapely.centroid(geoms[valid])
        x, y = shapely.get_x(centroids), shapely.get_y(centroids)
        inside = shapely.contains_xy(clip, x, y)
        keep = order[valid][inside]
        geoms = geoms[valid][inside]
        w, d, rot = oriented_boxes(geoms)
        ring_offsets, ring_xy = ring_arrays(_outer_rings(geoms))
        return Buildings(
            osm_type=osm_type[keep],
            osm_id=osm_id[keep],
            x=x[inside],
            y=y[inside],
            w=w,
            d=d,
            rot=rot,
            area_m2=shapely.area(geoms).astype(np.float32),
            kind=np.asarray(self.kind, np.uint8)[keep],
            tagged_height_m=np.asarray(self.height_m, np.float64)[keep],
            height_source=np.asarray(self.source, np.uint8)[keep],
            levels=np.asarray(self.levels, np.float64)[keep],
            rings={"offsets": ring_offsets, "xy": ring_xy},
            skipped_invalid=self.invalid + int((~valid).sum()),
        )


def _outer_rings(geoms: np.ndarray) -> np.ndarray:
    """Exterior ring, counter-clockwise, of the largest polygon of each footprint."""
    parts, owner = shapely.get_parts(geoms, return_index=True)
    by_size = np.lexsort((-shapely.area(parts), owner))
    first = by_size[np.searchsorted(owner[by_size], np.arange(len(geoms)))]
    largest = shapely.orient_polygons(parts[first], exterior_cw=False)
    return shapely.get_exterior_ring(largest)
