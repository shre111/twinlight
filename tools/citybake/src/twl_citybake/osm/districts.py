from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area, Node
from scipy.spatial import KDTree
from shapely import Geometry, Polygon

from twl_citybake.osm.banks import Banks
from twl_citybake.osm.geometry import from_wkb, polygon_parts
from twl_citybake.projection import LocalProjection

NODE, WAY, RELATION = 0, 1, 2
FALLBACK_NAME = "City"


def normalise_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


@dataclass(frozen=True)
class Seeds:
    """District seeds sorted by normalised name; position i is district id i."""

    name: list[str]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.name)


@dataclass
class PlaceCollector:
    kinds: tuple[str, ...]
    nodes: list[tuple[int, int, str, float, float]] = field(default_factory=list)
    areas: list[tuple[int, int, str, str]] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        if (name := self._name(node.tags)) is not None and node.location.valid():
            self.nodes.append((NODE, node.id, name, node.location.lon, node.location.lat))

    def add_area(self, area: Area, factory: WKBFactory) -> None:
        if (name := self._name(area.tags)) is None:
            return
        try:
            wkb = factory.create_multipolygon(area)
        except RuntimeError:
            return
        self.areas.append((WAY if area.from_way() else RELATION, area.orig_id(), name, wkb))

    def _name(self, tags: Any) -> str | None:
        name = tags.get("name")
        return name if tags.get("place") in self.kinds and name and normalise_name(name) else None

    def finish(self, proj: LocalProjection, clip: Polygon) -> Seeds:
        kind = [r[0] for r in self.nodes] + [r[0] for r in self.areas]
        osm_id = [r[1] for r in self.nodes] + [r[1] for r in self.areas]
        name = [r[2] for r in self.nodes] + [r[2] for r in self.areas]
        node_x, node_y = proj.forward([r[3] for r in self.nodes], [r[4] for r in self.nodes])
        centroids = shapely.centroid(from_wkb([r[3] for r in self.areas], proj))
        x = np.concatenate([node_x, shapely.get_x(centroids)])
        y = np.concatenate([node_y, shapely.get_y(centroids)])
        inside = np.flatnonzero(shapely.contains_xy(clip, x, y))
        best: dict[str, tuple[tuple[int, int], int]] = {}
        for i in inside:
            key, rank = normalise_name(name[i]), (kind[i], osm_id[i])
            if key not in best or rank < best[key][0]:
                best[key] = (rank, int(i))
        chosen = [best[key][1] for key in sorted(best)]
        first_at: dict[tuple[int, int], int] = {}
        for i in sorted(chosen, key=lambda i: (kind[i], osm_id[i])):
            first_at.setdefault(_millimetres(x[i], y[i]), i)
        chosen = [i for i in chosen if first_at[_millimetres(x[i], y[i])] == i]
        return Seeds([name[i] for i in chosen], x[chosen], y[chosen])


def _millimetres(x: float, y: float) -> tuple[int, int]:
    return round(float(x) * 1000), round(float(y) * 1000)


def assign_districts(
    seeds: Seeds, banks: Banks, x: np.ndarray, y: np.ndarray, bank: np.ndarray
) -> tuple[npt.NDArray[np.uint16], list[dict[str, Any]]]:
    """Nearest seed on the building's own bank, or nearest overall when that bank has no seed.
    Each district's polygon is the matching Voronoi cells clipped to the banks, so polygons and
    assignments agree."""
    if not len(seeds):
        return np.zeros(len(x), np.uint16), [_city_district(banks, len(x))]
    seed_xy = np.column_stack([seeds.x, seeds.y])
    seed_bank = banks.of(seeds.x, seeds.y)
    district = np.zeros(len(x), np.uint16)
    cells: list[list[Geometry]] = [[] for _ in range(len(seeds))]
    for b, region in enumerate(banks.regions):
        members = np.flatnonzero(seed_bank == b)
        if not len(members):
            members = np.arange(len(seeds))
        on_bank = np.flatnonzero(bank == b)
        if len(on_bank):
            _, nearest = KDTree(seed_xy[members]).query(np.column_stack([x[on_bank], y[on_bank]]))
            district[on_bank] = members[nearest]
        for seed, cell in zip(members, _voronoi_cells(seed_xy[members], region), strict=True):
            cells[seed].append(cell)
    counts = np.bincount(district, minlength=len(seeds))
    return district, [
        {
            "id": i,
            "name": seeds.name[i],
            "seed": _rounded(seed_xy[i]),
            "bank": int(seed_bank[i]),
            "buildings": int(counts[i]),
            "polygon": _exteriors(shapely.union_all(cells[i])),
        }
        for i in range(len(seeds))
    ]


def _voronoi_cells(points: np.ndarray, region: Geometry) -> list[Geometry]:
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    if len(unique) == 1:
        return [region] * len(points)
    diagram = shapely.voronoi_polygons(shapely.multipoints(unique), extend_to=region, ordered=True)
    clipped = shapely.intersection(shapely.get_parts(diagram), region)
    return [clipped[i] for i in inverse.ravel()]


def _city_district(banks: Banks, buildings: int) -> dict[str, Any]:
    whole = shapely.union_all(banks.regions)
    centre = shapely.centroid(whole)
    return {
        "id": 0,
        "name": FALLBACK_NAME,
        "seed": _rounded(np.array([centre.x, centre.y])),
        "bank": 0,
        "buildings": buildings,
        "polygon": _exteriors(whole),
    }


def _exteriors(geom: Geometry) -> list[list[list[float]]]:
    return [_rounded(np.asarray(p.exterior.coords)) for p in polygon_parts(geom)]


def _rounded(coords: np.ndarray) -> Any:
    return np.round(coords, 1).tolist()
