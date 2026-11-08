import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h3
import numpy as np
from numpy.typing import NDArray
from pack_shapes import closed, square_ring, water_table_of
from pytest import MonkeyPatch

from twl_citybake.cli import ward_file_path
from twl_citybake.config import (
    CityConfig,
    DataDirs,
    PopulationCheck,
    WardGroup,
    load_city,
)
from twl_citybake.osm.tags import BuildingKind
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table, table_digest, write_json, write_ragged, write_table

SLUG = "ahmedabad"
WARD_SOURCE = "Ward polygons: TESTWARDSRC"
WARD_NAMES = ("NORTH", "SOUTH", "EAST")
WARD_GROUP_NAMES = ("walled city", "east edge")
SEGMENTS = ["senior", "young_single", "other_adult"]
BANDS = [{"name": "morning_peak", "start_minute": 480}, {"name": "night", "start_minute": 1320}]
ATTRIBUTIONS = [
    "ATTR-BLD-A",
    "ATTR-BLD-B",
    "ATTR-OSM",
    "ATTR-PERS",
    "ATTR-POP",
    "ATTR-STORES",
]
TRAVEL_SHAPE = (2, 3, 2)

# (x, y, side_m, height_m, district); a square footprint centred on (x, y).
# 0 large; 1 large; 2 small and isolated; 3 and 4 small, 2.0 m apart; 5 large in the lake;
# 6 small and isolated in the lake; 7 and 8 large, near 2 (7 is nearer to 2's centre).
BUILDINGS: list[tuple[float, float, float, float, int]] = [
    (0.0, 0.0, 20.0, 9.6, 0),
    (100.0, 0.0, 15.0, 6.4, 0),
    (300.0, 300.0, 5.0, 3.2, 1),
    (500.0, 0.0, 5.0, 3.2, 1),
    (507.0, 0.0, 5.0, 3.2, 1),
    (-300.0, -300.0, 10.0, 6.4, 2),
    (-300.0, -340.0, 4.0, 3.2, 2),
    (340.0, 300.0, 12.0, 6.4, 1),
    (300.0, 350.0, 12.0, 6.4, 1),
]
KEPT = [0, 1, 3, 4, 7, 8]
DROPPED_IN_WATER = [5, 6]
DETAIL = [2]
FLOORS = [3, 2, 1, 1, 1, 2, 1, 2, 2]

# (home, x, y, floor)
RESIDENTS: list[tuple[int, float, float, int]] = [
    (0, 2.0, 3.0, 1),
    (2, 330.0, 340.0, 0),
    (5, -300.0, -300.0, 1),
    (1, 650.0, 100.0, 2),
    (6, -310.0, -345.0, 3),
    (0, -290.0, -310.0, 0),
    (0, -300.0, -240.0, 1),
    (2, 302.0, 298.0, 2),
    (3, 500.0, 0.0, 0),
    (4, 507.0, 1.0, 0),
]
# (baked building, x, y, floor) after the rules
EXPECTED_RESIDENTS: list[tuple[int, float, float, int]] = [
    (0, 2.0, 3.0, 1),
    (7, 330.0, 340.0, 0),
    (0, 0.0, 0.0, 1),
    (1, 100.0, 0.0, 1),
    (0, 0.0, 0.0, 2),
    (0, 0.0, 0.0, 0),
    (0, -300.0, -240.0, 1),
    (7, 302.0, 298.0, 2),
    (3, 500.0, 0.0, 0),
    (4, 507.0, 1.0, 0),
]
MOVED, REPOINTED, LOWERED = 4, 4, 2
REPOINT_DISTANCES = [40.0, float(np.hypot(300, 300)), float(np.hypot(300, 340)), 40.0]

ROAD_POINTS = [3, 2, 2, 2, 2, 2]
ROAD_CLASS = [0, 4, 2, 3, 1, 4]
ROAD_BRIDGE = [0, 0, 1, 0, 0, 1]
PACK_ROAD_CLASS = [1, 0, 2, 0, 1, 2]

DISTRICTS: list[dict[str, Any]] = [
    {
        "id": 0,
        "name": "Alpha",
        "name_local": "Alpha",
        "seed": [0.0, 0.0],
        "bank": 0,
        "buildings": 2,
        "polygon": [[[-50.0, -50.0], [150.0, -50.0], [150.0, 50.0], [-50.0, 50.0], [-50.0, -50.0]]],
    },
    {
        "id": 1,
        "name": "Beta",
        "name_local": "બીટા",
        "seed": [400.0, 150.0],
        "bank": 1,
        "buildings": 5,
        "polygon": [
            [[250.0, 250.0], [550.0, 250.0], [550.0, 400.0], [250.0, 400.0], [250.0, 250.0]],
            [[450.0, -20.0], [550.0, -20.0], [550.0, 20.0], [450.0, 20.0]],
        ],
    },
    {
        "id": 2,
        "name": "Ōtsu Kālupur",
        "name_local": "Ōtsu Kālupur",
        "seed": [-300.0, -320.0],
        "bank": 0,
        "buildings": 2,
        "polygon": [[[-400.0, -400.0], [-200.0, -400.0], [-200.0, -200.0], [-400.0, -200.0]]],
    },
]

LAKE_EXTERIOR = [(-400, -400), (-200, -400), (-200, -200), (-400, -200)]
LAKE_HOLE = [(-330, -250), (-330, -230), (-270, -230), (-270, -250)]
RIVER_EXTERIOR = [(600, -500), (700, -500), (700, 500), (600, 500)]

WardBox = tuple[str, float, float, float, float]
WARD_BOXES: list[WardBox] = [
    ("01 NORTH", -1000.0, 200.0, 1000.0, 1000.0),
    ("02 SOUTH", -1000.0, -1000.0, 1000.0, -100.0),
    ("03 EAST", 400.0, -100.0, 1000.0, 100.0),
]
# baked-area shares: walled city holds 2, 7, 8, 5, 6 and keeps 7, 8; east edge holds 3, 4
WALLED_FLOOR_KEPT = (144 * 2 + 144 * 2) / (25 + 288 + 288 + 200 + 16)
BAKED_FLOOR_AREA = [400 * 3, 225 * 2, 25, 25, 25, 200, 16, 288, 288]
FLOOR_KEPT = sum(BAKED_FLOOR_AREA[i] for i in KEPT) / sum(BAKED_FLOOR_AREA)


def fixture_water() -> dict[str, NDArray[Any]]:
    return water_table_of(
        [
            (0, [closed(RIVER_EXTERIOR)]),
            (1, [closed(LAKE_EXTERIOR), closed(LAKE_HOLE)]),
        ]
    )


def building_tables(
    order: Sequence[int] | None = None,
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]:
    """Baked row i is `BUILDINGS[order[i]]`."""
    order = list(range(len(BUILDINGS))) if order is None else list(order)
    rows = [BUILDINGS[i] for i in order]
    n = len(rows)
    side = np.array([b[2] for b in rows], np.float32)
    height = np.array([b[3] for b in rows], np.float32)
    rings = [square_ring(b[0], b[1], b[2]) for b in rows]
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rings])]).astype(np.uint32)
    table = {
        "osm_id": np.arange(n, dtype=np.int64) + 1,
        "x": np.array([b[0] for b in rows], np.float64),
        "y": np.array([b[1] for b in rows], np.float64),
        "w": side,
        "d": side,
        "rot": np.zeros(n, np.float32),
        "area_m2": side * side,
        "height_m": height,
        "floors": np.array([FLOORS[i] for i in order], np.uint8),
        "kind": np.full(n, int(BuildingKind.RESIDENTIAL), np.uint8),
        "district": np.array([b[4] for b in rows], np.uint16),
        "bank": np.array([1 if b[0] > 0 else 0 for b in rows], np.uint8),
    }
    ring_table = {
        "offsets": offsets,
        "xy": np.array([p for r in rings for p in r], np.float32).reshape(-1, 2),
    }
    return table, ring_table


def ward_geojson(proj: LocalProjection) -> dict[str, Any]:
    features = []
    for name, x0, y0, x1, y1 in WARD_BOXES:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        lon, lat = proj.inverse([c[0] for c in corners], [c[1] for c in corners])
        ring = [[float(a), float(b)] for a, b in zip(lon, lat, strict=True)]
        features.append(
            {
                "type": "Feature",
                "properties": {"Name": name},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def fixture_city(**pack_changes: Any) -> CityConfig:
    check = PopulationCheck(
        url="https://example.invalid/wards.geojson",
        md5="0" * 32,
        source=WARD_SOURCE,
        name_property="Name",
        ward_people_2011=1000,
        total_people_2011=3000,
        total_growth=(1.0, 2.0),
        ward_groups=(
            WardGroup(name="walled city", wards=("NORTH", "SOUTH")),
            WardGroup(name="east edge", wards=("EAST",)),
        ),
    )
    city = replace(load_city(SLUG), seed=20260101, population_check=check)
    return replace(city, pack=replace(city.pack, **pack_changes)) if pack_changes else city


@dataclass(frozen=True)
class PackInputs:
    city: CityConfig
    dirs: DataDirs
    buildings_layers: Path
    osm_layers: Path
    population: Path
    personas: Path
    graph: Path
    stores: Path
    wards: Path

    def args(self, wards: bool = False) -> tuple[Any, ...]:
        return (
            self.buildings_layers,
            self.osm_layers,
            self.population,
            self.personas,
            self.graph,
            self.stores,
            self.city,
            self.wards if wards else None,
        )


def build_pack_inputs(
    root: Path, city: CityConfig | None = None, building_order: Sequence[int] | None = None
) -> PackInputs:
    """`building_order` writes the baked buildings in another order; residents follow."""
    city = city or fixture_city()
    dirs = DataDirs(root)
    proj = LocalProjection.for_bbox(city.bbox)
    n = len(RESIDENTS)

    osm = dirs.stage(SLUG, "osm") / "layers"
    combined = dirs.stage(SLUG, "buildings") / "layers"
    buildings, rings = building_tables(building_order)
    baked_row = {b: row for row, b in enumerate(building_order or range(len(BUILDINGS)))}
    write_table(osm / "buildings", buildings)
    write_table(combined / "buildings", buildings)
    write_ragged(combined / "building_rings", rings, rows=len(BUILDINGS))
    write_json(combined / "districts.json", DISTRICTS)
    write_json(
        combined / "layer.json",
        {
            "attribution": ["ATTR-BLD-A", "ATTR-BLD-B"],
            "osm_buildings_sha256": table_digest(osm / "buildings"),
        },
    )

    xy = np.array(
        [
            *[(0, 0), (50, 0), (100, 0), (500, 0), (520, 10), (-100, 0), (-200, 20), (-300, 30)],
            *[(-100, 100), (600, 100), (650, 120), (700, 140), (-500, 300)],
        ],
        np.float32,
    )
    offsets = np.array([0, 3, 5, 7, 9, 11, 13], np.uint32)
    road_xy = xy[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    write_ragged(osm / "road_geom", {"offsets": offsets, "xy": road_xy}, rows=6)
    write_table(
        osm / "road_edges",
        {
            "u": np.arange(6, dtype=np.uint32),
            "v": np.arange(6, dtype=np.uint32) + 1,
            "length_m": np.full(6, 100.0, np.float32),
            "road_class": np.array(ROAD_CLASS, np.uint8),
            "oneway": np.zeros(6, np.int8),
            "bridge": np.array(ROAD_BRIDGE, np.uint8),
            "osm_way": np.arange(6, dtype=np.int64) + 1,
        },
    )
    water = fixture_water()
    write_ragged(osm / "water", water, rows=len(water["poly_kind"]))
    write_table(
        osm / "road_nodes",
        {"osm_id": np.arange(7, dtype=np.int64), "x": np.zeros(7), "y": np.zeros(7)},
    )
    write_json(osm / "layer.json", {"attribution": "ATTR-OSM"})

    population = dirs.stage(SLUG, "population")
    write_table(
        population / "residents",
        {
            "household": np.arange(n, dtype=np.uint32),
            "building": np.array([baked_row[r[0]] for r in RESIDENTS], np.uint32),
            "district": np.zeros(n, np.uint16),
            "bank": np.zeros(n, np.uint8),
            "x": np.array([r[1] for r in RESIDENTS], np.float32),
            "y": np.array([r[2] for r in RESIDENTS], np.float32),
            "z": np.array([(r[3] + 0.5) * 3.2 for r in RESIDENTS], np.float32),
            "weight": 1.0 + 0.25 * np.arange(n, dtype=np.float32),
        },
    )
    write_json(population / "layer.json", {"attribution": ["ATTR-POP", "ATTR-OSM"]})
    residents_sha = table_digest(population / "residents")

    personas = dirs.stage(SLUG, "personas")
    write_table(
        personas / "assignment",
        {
            "persona_ref": np.arange(n, dtype=np.uint32) + 1000,
            "segment": (np.arange(n) % 3).astype(np.uint16),
            "spouse": np.full(n, -1, np.int32),
        },
    )
    write_json(
        personas / "layer.json",
        {
            "attribution": ["ATTR-PERS"],
            "population_residents_sha256": residents_sha,
            "segments": SEGMENTS,
        },
    )

    graph = dirs.stage(SLUG, "graph")
    write_ragged(
        graph / "graph",
        {
            "offsets": np.arange(n + 1, dtype=np.uint32),
            "targets": ((np.arange(n) + 1) % n).astype(np.uint32),
            "kind": np.full(n, 2, np.uint8),
            "strength": np.full(n, 0.5, np.float32),
        },
        rows=n,
    )
    write_json(
        graph / "layer.json",
        {
            "population_residents_sha256": residents_sha,
            "persona_assignment_sha256": table_digest(personas / "assignment"),
            "buildings_sha256": table_digest(combined / "buildings"),
        },
    )

    stores = dirs.stage(SLUG, "stores")
    write_table(
        stores / "stores",
        {
            "node": np.array([0, 1], np.uint32),
            "x": np.array([50.0, 500.0], np.float32),
            "y": np.array([10.0, 20.0], np.float32),
            "hex": np.array([0, 2], np.uint32),
            "radius_m": np.full(2, 2500.0, np.float32),
            "district": np.array([0, 1], np.uint16),
        },
    )
    base = h3.str_to_int("8928308280fffff")
    write_table(stores / "hexes", {"h3": np.array([base, base + 2, base + 4], np.uint64)})
    write_table(stores / "resident_hex", {"hex": (np.arange(n) % 3).astype(np.uint32)})
    write_table(
        stores / "travel",
        {"minutes": np.arange(1, 13, dtype=np.uint16)},
    )
    write_json(stores / "bands.json", BANDS)
    write_json(
        stores / "layer.json",
        {
            "attribution": "ATTR-STORES",
            "population_residents_sha256": residents_sha,
            "buildings_sha256": table_digest(combined / "buildings"),
            "osm_road_nodes_sha256": table_digest(osm / "road_nodes"),
            "osm_road_edges_sha256": table_digest(osm / "road_edges"),
            "travel_shape": list(TRAVEL_SHAPE),
        },
    )

    wards = ward_file_path(SLUG, city.population_check, dirs)  # type: ignore[arg-type]
    wards.parent.mkdir(parents=True, exist_ok=True)
    wards.write_text(json.dumps(ward_geojson(proj)), encoding="utf-8")
    return PackInputs(city, dirs, combined, osm, population, personas, graph, stores, wards)


def rewrite_meta(folder: Path, **changes: Any) -> None:
    path = folder / "layer.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    write_json(path, {**meta, **changes})


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def delete(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    path.rmdir()


def patch_budget(monkeypatch: MonkeyPatch, name: str, value: int) -> None:
    """Set a budget constant wherever the pack code or the writer holds a copy of it."""
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith(("twl_pack", "twl_citybake.pack")) and hasattr(module, name):
            monkeypatch.setattr(module, name, value)


def baked_centres() -> NDArray[np.float64]:
    return np.array([[b[0], b[1]] for b in BUILDINGS], np.float64)


def residents_table(inputs: PackInputs) -> dict[str, NDArray[Any]]:
    return read_table(inputs.population / "residents")
