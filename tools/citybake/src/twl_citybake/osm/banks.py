from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import osmium
import shapely
from osmium.osm import Way
from shapely import Geometry, Polygon
from shapely.ops import split

from twl_citybake.projection import LocalProjection

WEST, EAST = 0, 1


@dataclass(frozen=True)
class Banks:
    """The clip polygon cut by the river centreline: regions[0] holds the north-west corner."""

    regions: tuple[Geometry, ...]

    @property
    def split(self) -> bool:
        return len(self.regions) > 1

    def of(self, x: np.ndarray, y: np.ndarray) -> npt.NDArray[np.uint8]:
        if not self.split:
            return np.zeros(len(x), np.uint8)
        west = self.regions[WEST]
        shapely.prepare(west)
        return np.where(shapely.contains_xy(west, x, y), WEST, EAST).astype(np.uint8)


@dataclass
class RiverCollector:
    names: tuple[str, ...]
    lines: list[tuple[int, list[tuple[float, float]]]] = field(default_factory=list)

    def add(self, way: Way) -> None:
        if way.tags.get("waterway") != "river" or way.tags.get("name") not in self.names:
            return
        try:
            coords = [(n.lon, n.lat) for n in way.nodes]
        except osmium.InvalidLocationError:
            return
        if len(coords) >= 2:
            self.lines.append((way.id, coords))

    def finish(self, proj: LocalProjection) -> Geometry:
        """Merged centreline in local metres (empty when no named river is mapped)."""
        lines = []
        for _, coords in sorted(self.lines):
            lon, lat = np.asarray(coords, np.float64).T
            lines.append(shapely.LineString(np.column_stack(proj.forward(lon, lat))))
        return shapely.line_merge(shapely.MultiLineString(lines))


def split_banks(clip: Polygon, river: Geometry, north_west: tuple[float, float]) -> Banks:
    """West is the piece holding the north-west corner, east the union of the rest. The unclipped
    centreline is used so its ends cross the clip boundary instead of stopping on it."""
    if shapely.is_empty(river) or not shapely.intersects(river, clip):
        return Banks((clip,))
    pieces = list(split(clip, river).geoms)
    if len(pieces) < 2:
        return Banks((clip,))
    corner = shapely.Point(north_west)
    west = int(np.argmin(shapely.distance(np.asarray(pieces, dtype=object), corner)))
    east = shapely.union_all([p for i, p in enumerate(pieces) if i != west])
    return Banks((pieces[west], east))
