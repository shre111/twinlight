from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area, Node
from scipy.spatial import KDTree
from shapely import Geometry, Polygon

from twl_citybake.config import DistrictConfig, normalise_name
from twl_citybake.osm.banks import Banks
from twl_citybake.osm.geometry import from_wkb, polygon_parts
from twl_citybake.projection import LocalProjection

EXTRA, NODE, WAY, RELATION = -1, 0, 1, 2
FALLBACK_NAME = "City"
RULES = ("extra_seeds", "merged", "renamed")


@dataclass(frozen=True)
class Seeds:
    """District seeds sorted by normalised display name; position i is district id i. Merged
    seeds still claim buildings and cells, which then belong to district `merged_into`.
    `applied` counts the seed-table rules that took effect."""

    name: list[str]
    name_local: list[str]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    merged_x: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    merged_y: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    merged_into: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, np.intp))
    applied: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RULES, 0))

    def __len__(self) -> int:
        return len(self.name)


@dataclass
class PlaceCollector:
    """Candidates as (kind, osm_id, display name, local name, location); configured extra
    seeds rank as kind EXTRA, ahead of every OSM place."""

    config: DistrictConfig
    nodes: list[tuple[int, int, str, str, float, float]] = field(default_factory=list)
    areas: list[tuple[int, int, str, str, str]] = field(default_factory=list)
    extra_at: dict[int, tuple[float, float]] = field(default_factory=dict)
    extra_nodes: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        self.extra_nodes = frozenset(seed.node for seed in self.config.extra_seeds)

    def add_node(self, node: Node) -> None:
        names = self._names(node.tags)
        extra = node.id in self.extra_nodes
        if (names is None and not extra) or not node.location.valid():
            return
        lon, lat = node.location.lon, node.location.lat
        if extra:
            self.extra_at[node.id] = (lon, lat)
        if names is not None:
            self.nodes.append((NODE, node.id, *names, lon, lat))

    def add_area(self, area: Area, factory: WKBFactory) -> None:
        if (names := self._names(area.tags)) is None:
            return
        try:
            wkb = factory.create_multipolygon(area)
        except RuntimeError:
            return
        self.areas.append((WAY if area.from_way() else RELATION, area.orig_id(), *names, wkb))

    def _names(self, tags: Any) -> tuple[str, str] | None:
        """`name:en` when it has letters or digits, else `name`; the local name is `name`,
        else the display name."""
        if tags.get("place") not in self.config.place_kinds:
            return None
        local, english = tags.get("name") or "", tags.get("name:en") or ""
        display = english if normalise_name(english) else local
        if not normalise_name(display):
            return None
        return display, local if normalise_name(local) else display

    def finish(self, proj: LocalProjection, clip: Polygon) -> Seeds:
        extras = sorted(self.config.extra_seeds, key=lambda seed: seed.node)
        if missing := [s.node for s in extras if s.node not in self.extra_at]:
            raise ValueError(f"districts.extra_seeds node {missing[0]} is not in the OSM file")
        points = [r[4:] for r in self.nodes] + [self.extra_at[s.node] for s in extras]
        point_x, point_y = proj.forward([p[0] for p in points], [p[1] for p in points])
        centroids = shapely.centroid(from_wkb([r[4] for r in self.areas], proj))
        x = np.concatenate([point_x, shapely.get_x(centroids)])
        y = np.concatenate([point_y, shapely.get_y(centroids)])
        kind = [r[0] for r in self.nodes] + [EXTRA] * len(extras) + [r[0] for r in self.areas]
        osm_id = [r[1] for r in self.nodes] + [s.node for s in extras] + [r[1] for r in self.areas]
        name = [r[2] for r in self.nodes] + [s.name for s in extras] + [r[2] for r in self.areas]
        local = [r[3] for r in self.nodes] + [s.name for s in extras] + [r[3] for r in self.areas]
        inside = shapely.contains_xy(clip, x, y)
        extra_inside = inside[len(self.nodes) : len(self.nodes) + len(extras)]
        if outside := [s.node for s, ok in zip(extras, extra_inside, strict=True) if not ok]:
            raise ValueError(f"districts.extra_seeds node {outside[0]} lies outside the bbox")
        best: dict[str, tuple[tuple[int, int], int]] = {}
        for i in np.flatnonzero(inside):
            key, rank = normalise_name(name[i]), (kind[i], osm_id[i])
            if key not in best or rank < best[key][0]:
                best[key] = (rank, int(i))
        chosen = [best[key][1] for key in sorted(best)]
        first_at: dict[tuple[int, int], int] = {}
        for i in sorted(chosen, key=lambda i: (kind[i], osm_id[i])):
            first_at.setdefault(_millimetres(x[i], y[i]), i)
        chosen = [i for i in chosen if first_at[_millimetres(x[i], y[i])] == i]
        final, merged_into, renamed = _seed_table(self.config, {i: name[i] for i in chosen})
        order = sorted(final, key=lambda i: normalise_name(final[i]))
        district_of = {i: d for d, i in enumerate(order)}
        merged = sorted(merged_into, key=lambda i: normalise_name(name[i]))
        at = np.asarray(merged, np.intp)
        applied = (sum(kind[i] == EXTRA for i in chosen), len(merged), renamed)
        return Seeds(
            name=[final[i] for i in order],
            name_local=[local[i] for i in order],
            x=x[order],
            y=y[order],
            merged_x=x[at],
            merged_y=y[at],
            merged_into=np.asarray([district_of[merged_into[i]] for i in merged], np.intp),
            applied=dict(zip(RULES, applied, strict=True)),
        )


def _seed_table(
    config: DistrictConfig, names: dict[int, str]
) -> tuple[dict[int, str], dict[int, int], int]:
    """Merges, then renames, over the deduped seeds keyed by candidate index: the final display
    names of the kept seeds, each merged seed's target, and the number of renames."""
    by_key = {normalise_name(n): i for i, n in names.items()}
    merge_keys = sorted(config.merge, key=normalise_name)
    targets = [_match("districts.merge target", config.merge[key], by_key) for key in merge_keys]
    sources = [_match("districts.merge key", key, by_key) for key in merge_keys]
    merged_into = dict(zip(sources, targets, strict=True))
    kept = {i: n for i, n in names.items() if i not in merged_into}
    kept_by_key = {normalise_name(n): i for i, n in kept.items()}
    renames = {
        _match("districts.rename key", key, kept_by_key): config.rename[key]
        for key in sorted(config.rename, key=normalise_name)
    }
    final = {i: renames.get(i, n) for i, n in kept.items()}
    holder: dict[str, int] = {}
    for i in sorted(final, key=lambda i: i in renames):
        if (other := holder.setdefault(normalise_name(final[i]), i)) != i:
            raise ValueError(
                f"districts.rename of {names[i]!r} to {final[i]!r} collides with {names[other]!r}"
            )
    return final, merged_into, len(renames)


def _match(what: str, name: str, by_key: dict[str, int]) -> int:
    if (i := by_key.get(normalise_name(name))) is None:
        raise ValueError(f"{what} {name!r} matches no district seed")
    return i


def _millimetres(x: float, y: float) -> tuple[int, int]:
    return round(float(x) * 1000), round(float(y) * 1000)


def assign_districts(
    seeds: Seeds, banks: Banks, x: np.ndarray, y: np.ndarray, bank: np.ndarray
) -> tuple[npt.NDArray[np.uint16], list[dict[str, Any]]]:
    """Nearest seed on the building's own bank, or nearest overall when that bank has no seed;
    a merged seed's buildings and cells then go to its target. Each district's polygon is its
    seeds' Voronoi cells clipped to the banks, so polygons and assignments agree."""
    if not len(seeds):
        return np.zeros(len(x), np.uint16), [_city_district(banks, len(x))]
    point_xy = np.column_stack(
        [np.concatenate([seeds.x, seeds.merged_x]), np.concatenate([seeds.y, seeds.merged_y])]
    )
    owner = np.concatenate([np.arange(len(seeds)), seeds.merged_into])
    point_bank = banks.of(point_xy[:, 0], point_xy[:, 1])
    district = np.zeros(len(x), np.uint16)
    cells: list[list[Geometry]] = [[] for _ in range(len(seeds))]
    for b, region in enumerate(banks.regions):
        members = np.flatnonzero(point_bank == b)
        if not len(members):
            members = np.arange(len(point_xy))
        on_bank = np.flatnonzero(bank == b)
        if len(on_bank):
            _, nearest = KDTree(point_xy[members]).query(np.column_stack([x[on_bank], y[on_bank]]))
            district[on_bank] = owner[members[nearest]]
        for point, cell in zip(members, _voronoi_cells(point_xy[members], region), strict=True):
            cells[owner[point]].append(cell)
    seed_xy, seed_bank = point_xy[: len(seeds)], point_bank[: len(seeds)]
    counts = np.bincount(district, minlength=len(seeds))
    return district, [
        {
            "id": i,
            "name": seeds.name[i],
            "name_local": seeds.name_local[i],
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
        "name_local": FALLBACK_NAME,
        "seed": _rounded(np.array([centre.x, centre.y])),
        "bank": 0,
        "buildings": buildings,
        "polygon": _exteriors(whole),
    }


def _exteriors(geom: Geometry) -> list[list[list[float]]]:
    return [_rounded(np.asarray(p.exterior.coords)) for p in polygon_parts(geom)]


def _rounded(coords: np.ndarray) -> Any:
    return np.round(coords, 1).tolist()
