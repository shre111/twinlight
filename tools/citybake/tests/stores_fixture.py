from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h3
import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import (
    CityConfig,
    DataDirs,
    StoresConfig,
    TravelBand,
    TravelConfig,
    load_city,
)
from twl_citybake.osm.tags import BuildingKind, RoadClass
from twl_citybake.projection import LocalProjection
from twl_citybake.store import write_json, write_table

Edge = tuple[int, int, float, int, int]
Resident = tuple[float, float, float]
Building = tuple[float, float, int, int]

SPEEDS = {
    "major": 40.0,
    "primary": 30.0,
    "secondary": 26.0,
    "tertiary": 22.0,
    "local": 16.0,
    "service": 12.0,
}
BANDS_SHUFFLED = (
    TravelBand(name="night", start_minute=1320, factor=1.25),
    TravelBand(name="morning_peak", start_minute=480, factor=0.8),
    TravelBand(name="midday", start_minute=660, factor=1.0),
    TravelBand(name="evening_peak", start_minute=1020, factor=0.7),
)
BANDS_SORTED = ("morning_peak", "midday", "evening_peak", "night")
DISTRICT_NAMES = ("Alpha", "Beta", "Gamma")
LOCAL = int(RoadClass.LOCAL)
H3_RESOLUTION = 9


def travel_config(**changes: Any) -> TravelConfig:
    base = TravelConfig(h3_resolution=H3_RESOLUTION, speed_kmh=SPEEDS, bands=BANDS_SHUFFLED)
    return replace(base, **changes)


def city_with(
    radius_m: float,
    coverage_target: float = 1.0,
    max_stores: int = 80,
    candidate_cell_m: float = 500.0,
    seed: int = 20260101,
    travel: TravelConfig | None = None,
) -> CityConfig:
    city = replace(load_city("ahmedabad"), seed=seed)
    stores = StoresConfig(
        radius_m=radius_m,
        coverage_target=coverage_target,
        max_stores=max_stores,
        candidate_cell_m=candidate_cell_m,
    )
    return replace(city, stores=stores, travel=travel or travel_config())


def projection(city: CityConfig) -> LocalProjection:
    return LocalProjection.for_bbox(city.bbox)


def hex_centre_xy(proj: LocalProjection, x: float, y: float) -> tuple[float, float]:
    lon, lat = proj.inverse(x, y)
    cell = h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION)
    clat, clon = h3.cell_to_latlng(cell)
    cx, cy = proj.forward(clon, clat)
    return float(cx), float(cy)


def road_tables(
    nodes: Sequence[tuple[float, float]], edges: Sequence[Edge]
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]:
    node_table = {
        "osm_id": np.arange(len(nodes), dtype=np.int64) + 1000,
        "x": np.array([p[0] for p in nodes], np.float64),
        "y": np.array([p[1] for p in nodes], np.float64),
    }
    edge_table = edge_columns(edges)
    return node_table, edge_table


def edge_columns(edges: Sequence[Edge]) -> dict[str, NDArray[Any]]:
    return {
        "u": np.array([e[0] for e in edges], np.uint32),
        "v": np.array([e[1] for e in edges], np.uint32),
        "length_m": np.array([e[2] for e in edges], np.float32),
        "road_class": np.array([e[3] for e in edges], np.uint8),
        "oneway": np.array([e[4] for e in edges], np.int8),
        "bridge": np.zeros(len(edges), np.uint8),
        "osm_way": np.arange(len(edges), dtype=np.int64) + 1,
    }


def two_way(u: int, v: int, length_m: float, road_class: int = LOCAL) -> Edge:
    return (u, v, length_m, road_class, 0)


@dataclass(frozen=True)
class Inputs:
    city: CityConfig
    dirs: DataDirs
    population: Path
    osm_layers: Path
    buildings_layers: Path
    residents: int


def build_inputs(
    root: Path,
    city: CityConfig,
    residents: Sequence[Resident],
    nodes: Sequence[tuple[float, float]],
    edges: Sequence[Edge],
    buildings: Sequence[Building] = ((0.0, 0.0, 0, 0),),
) -> Inputs:
    dirs = DataDirs(root)
    n = len(residents)
    x = np.array([r[0] for r in residents], np.float32)
    population = dirs.stage(city.slug, "population")
    write_table(
        population / "residents",
        {
            "household": np.arange(n, dtype=np.uint32),
            "building": np.zeros(n, np.uint32),
            "district": np.zeros(n, np.uint16),
            "bank": (x > 0).astype(np.uint8),
            "x": x,
            "y": np.array([r[1] for r in residents], np.float32),
            "z": np.zeros(n, np.float32),
            "weight": np.array([r[2] for r in residents], np.float32),
        },
    )
    layers = dirs.stage(city.slug, "osm") / "layers"
    node_table, edge_table = road_tables(nodes, edges)
    write_table(layers / "road_nodes", node_table)
    write_table(layers / "road_edges", edge_table)
    nb = len(buildings)
    bx = np.array([b[0] for b in buildings], np.float64)
    write_table(
        layers / "buildings",
        {
            "x": bx,
            "y": np.array([b[1] for b in buildings], np.float64),
            "area_m2": np.full(nb, 100.0, np.float32),
            "height_m": np.full(nb, 6.4, np.float32),
            "floors": np.full(nb, 2, np.uint8),
            "kind": np.full(nb, int(BuildingKind.RESIDENTIAL), np.uint8),
            "district": np.array([b[2] for b in buildings], np.uint16),
            "bank": np.array([b[3] for b in buildings], np.uint8),
        },
    )
    write_json(
        layers / "districts.json",
        [{"id": i, "name": name} for i, name in enumerate(DISTRICT_NAMES)],
    )
    return Inputs(city, dirs, population, layers, layers, n)


COLUMNS = (-1500.0, -900.0, -300.0, 300.0, 900.0)
ROWS = (-600.0, 0.0, 600.0)


def grid_index(column: int, row: int) -> int:
    return column * len(ROWS) + row


def grid_inputs(root: Path, city: CityConfig) -> Inputs:
    """5 x 3 two-way local grid at 600 m spacing with a resident group on every node, plus a
    two-node island and a one-way dead-end spur each placed at a resident hex centre."""
    proj = projection(city)
    nodes = [(cx, ry) for cx in COLUMNS for ry in ROWS]
    edges: list[Edge] = []
    for c in range(len(COLUMNS)):
        for r in range(len(ROWS)):
            if c + 1 < len(COLUMNS):
                edges.append(two_way(grid_index(c, r), grid_index(c + 1, r), 600.0))
            if r + 1 < len(ROWS):
                edges.append(two_way(grid_index(c, r), grid_index(c, r + 1), 600.0))
    ix, iy = hex_centre_xy(proj, COLUMNS[0], ROWS[1])
    nodes += [(ix, iy), (ix + 20.0, iy)]
    edges.append(two_way(15, 16, 20.0))
    sx, sy = hex_centre_xy(proj, COLUMNS[2], ROWS[1])
    nodes.append((sx, sy))
    edges.append((grid_index(2, 1), 17, float(np.hypot(sx - COLUMNS[2], sy - ROWS[1])), LOCAL, 1))
    residents: list[Resident] = []
    for i, (px, py) in enumerate(nodes[:15]):
        residents += [(px, py, 1.0 + 0.5 * (i % 3))] * (1 + i % 3)
    district = [0 if px < -1200 else 1 if px < 600 else 2 for px in COLUMNS]
    buildings = [
        (px + 50.0, py + 50.0, district[c], int(px + 50.0 > 0))
        for c, px in enumerate(COLUMNS)
        for py in ROWS
    ]
    return build_inputs(root, city, residents, nodes, edges, buildings)


def line_inputs(root: Path, city: CityConfig) -> Inputs:
    """Three nodes 600 m apart; residents sit 400 m west of the first node and on the last one."""
    nodes = [(400.0, 100.0), (1000.0, 100.0), (1600.0, 100.0)]
    edges = [two_way(0, 1, 600.0), two_way(1, 2, 600.0)]
    residents: list[Resident] = [(0.0, 0.0, 1.0)] * 3 + [(1600.0, 100.0, 1.0)]
    return build_inputs(root, city, residents, nodes, edges, [(400.0, 150.0, 0, 0)])


def heavy_inputs(root: Path, city: CityConfig) -> Inputs:
    """Ten light residents at one end of a 3 km road, three heavy ones at the other."""
    nodes = [(0.0, 0.0), (3000.0, 0.0)]
    edges = [two_way(0, 1, 3000.0, int(RoadClass.MAJOR))]
    residents: list[Resident] = [(50.0, 0.0, 1.0)] * 10 + [(3050.0, 0.0, 10.0)] * 3
    return build_inputs(root, city, residents, nodes, edges, [(0.0, 30.0, 0, 0)])


class StubBackend:
    name = "stub"

    def __init__(self, value: Callable[[NDArray[Any], NDArray[Any], int], NDArray[Any]]) -> None:
        self.value = value
        self.calls: list[tuple[int, int, str]] = []

    def minutes(self, sources: NDArray[Any], targets: NDArray[Any], band: TravelBand) -> Any:
        starts = sorted(b.start_minute for b in BANDS_SHUFFLED)
        self.calls.append((len(sources), len(targets), band.name))
        out = self.value(
            sources.astype(np.int64), targets.astype(np.int64), starts.index(band.start_minute)
        )
        assert out.shape == (len(sources), len(targets))
        return out.astype(np.float64)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }
