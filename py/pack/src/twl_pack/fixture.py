"""The cross-language golden pack in `fixtures/pack/v2/small`, and its probes."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_pack.format import NO_PERSONA
from twl_pack.reader import PackBuildings, PackReader, PackRoads
from twl_pack.types import (
    Band,
    BuildingColumns,
    GraphColumns,
    PackContents,
    ResidentColumns,
    ResidentEngineColumns,
    RiverColumns,
    RoadColumns,
    StoreColumns,
)
from twl_pack.writer import encode_pack, write_pack

_CONTENT_SEED = 25_001
_PROBE_SEED = 25_002

# The anchor roads reach these exactly, so the bbox is known and q runs the full 0..65535.
_HALF_X, _HALF_Y = 3000, 2000
_CRS = "+proj=tmerc +lat_0=23.03 +lon_0=72.58 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
_DISTRICT_IDS = (0, 1, 3)
_SEGMENTS = ("senior", "young_single", "working_graduate")
_HEX_COUNT = 12
# Shaped like H3 ids and above 2^53, so a reader that passes them through a double fails.
_FIRST_HEX = 0x0883B10000FFFFFF
_HEX_STEP = 1 << 28


def small_contents() -> PackContents:
    # Only integer draws and IEEE division here, no libm, so every CPU gets the same bits.
    rng = np.random.default_rng(_CONTENT_SEED)
    buildings = _buildings(rng)
    residents = _residents(rng, buildings)
    stores = _stores()
    hexes = np.array([_FIRST_HEX + i * _HEX_STEP for i in range(_HEX_COUNT)], dtype=np.uint64)
    bands = [Band("night", 0), Band("morning", 420), Band("evening", 1020)]
    return PackContents(
        kind="city",
        name="small",
        crs=_CRS,
        segments=list(_SEGMENTS),
        bands=bands,
        attributions=[
            "OSM ODbL",
            "GHS-POP CC BY 4.0",
            "Nemotron-Personas-India CC BY 4.0",
            "OSM ODbL",
        ],
        districts=_districts(),
        buildings=buildings,
        roads=_roads(rng),
        residents=residents,
        residents_engine=_residents_engine(rng, residents.x.size),
        stores=stores,
        river=_river(),
        graph=_graph(rng, residents.x.size),
        detail=_detail(rng),
        hexes=hexes,
        travel=_travel(rng, stores.x.size * hexes.size * len(bands)),
        tile_level=3,
    )


def write_fixture(out_dir: Path) -> None:
    write_pack(encode_pack(small_contents()), out_dir)
    probes = _probes(PackReader(out_dir))
    (out_dir / "probes.json").write_bytes(
        (json.dumps(probes, sort_keys=True, indent=2) + "\n").encode()
    )


def _tenths(rng: np.random.Generator, low: float, high: float, size: int) -> NDArray[np.float64]:
    return rng.integers(int(low * 10), int(high * 10), size) / 10


def _districts() -> list[dict[str, Any]]:
    return [
        {
            "id": 0,
            "name": "Navrangpura",
            "name_local": "નવરંગપુરા",
            "x": -1500.0,
            "y": 900.0,
            "east": False,
            "polygons": [[[-3000.0, 0.0], [0.0, 0.0], [0.0, 2000.0], [-3000.0, 2000.0]]],
        },
        {
            "id": 1,
            "name": "Maninagar",
            "name_local": "મણિનગર",
            "x": 1500.0,
            "y": -900.0,
            "east": True,
            "polygons": [
                [[0.0, -2000.0], [3000.0, -2000.0], [3000.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [3000.0, 0.0], [3000.0, 2000.0], [0.0, 2000.0]],
            ],
        },
        {
            "id": 3,
            "name": "Sabarmati",
            "name_local": "સાબરમતી",
            "x": -1500.0,
            "y": -900.0,
            "east": False,
            "polygons": [[[-3000.0, -2000.0], [0.0, -2000.0], [0.0, 0.0], [-3000.0, 0.0]]],
        },
    ]


def _buildings(rng: np.random.Generator) -> BuildingColumns:
    # x, y, width, depth, height, rotation, district. Corners give codes 0 and 0xFFFFFFFF; the next
    # five tie on Morton code, three at one spot and two only once quantised; 451.5625 m is the
    # largest size that fits, 512 m and 600 m clip.
    hand = np.array(
        [
            [-3000.0, -2000.0, 10.0, 10.0, 6.4, 0.0, 3],
            [3000.0, 2000.0, 10.0, 10.0, 6.4, 3.0, 1],
            [120.5, -340.25, 10.0, 11.0, 12.0, 0.5, 1],
            [120.5, -340.25, 20.0, 21.0, 22.0, 1.0, 1],
            [120.5, -340.25, 30.0, 31.0, 32.0, 1.5, 1],
            [-700.25, 640.5, 40.0, 41.0, 42.0, -0.5, 0],
            [-700.2501, 640.5001, 50.0, 51.0, 52.0, -1.5, 0],
            [800.0, 600.0, 512.0, 40.0, 30.0, 7.0, 1],
            [-800.0, -600.0, 20.0, 451.5625, 30.0, -4.0, 3],
            [900.0, -700.0, 30.0, 30.0, 600.0, 3.2, 1],
            [-900.0, 700.0, 0.0, 12.0, 3.2, 0.25, 0],
        ]
    )
    # Nothing is drawn past x = 1500, so the two eastern columns of tiles stay almost empty.
    drawn = 230
    return BuildingColumns(
        x=np.concatenate((hand[:, 0], _tenths(rng, -2900, 1500, drawn))),
        y=np.concatenate((hand[:, 1], _tenths(rng, -1900, 1900, drawn))),
        width=np.concatenate((hand[:, 2], _tenths(rng, 3, 60, drawn))),
        depth=np.concatenate((hand[:, 3], _tenths(rng, 3, 60, drawn))),
        height=np.concatenate((hand[:, 4], _tenths(rng, 3, 90, drawn))),
        rotation=np.concatenate((hand[:, 5], rng.integers(-40_000, 40_000, drawn) / 10_000)),
        district=np.concatenate((hand[:, 6], rng.choice(_DISTRICT_IDS, drawn))).astype(np.int64),
    )


def _detail(rng: np.random.Generator) -> BuildingColumns:
    rows = 20
    x, y = _tenths(rng, -2900, 2900, rows), _tenths(rng, -1900, 1900, rows)
    x[1], y[1] = x[0], y[0]
    depth = _tenths(rng, 3, 8, rows)
    depth[2] = 600.0
    return BuildingColumns(
        x=x,
        y=y,
        width=_tenths(rng, 3, 8, rows),
        depth=depth,
        height=_tenths(rng, 3, 6, rows),
        rotation=rng.integers(0, 31_415, rows) / 10_000,
        district=rng.choice(_DISTRICT_IDS, rows),
    )


def _roads(rng: np.random.Generator) -> RoadColumns:
    # q(0.01) = 32768 and q(-0.05) = 32767 on both axes, so these hit both ends of the signed wrap:
    # the first point starts above 32767, -3000 to 3000 is a jump of 65535, 0 to 32768 wraps to
    # -32768, and the one-point road sits between two of them.
    anchors = [
        (1, [(2500.0, 1500.0), (2600.0, 1400.0), (-3000.0, -2000.0), (3000.0, 2000.0)]),
        (0, [(-3000.0, -2000.0)]),
        (2, [(0.01, 0.01), (-3000.0, -2000.0), (-0.05, -0.05), (3000.0, 2000.0)]),
        (0, [(-2800.0, 1900.0), (2800.0, -1900.0)]),
    ]
    streets = 40
    points = rng.integers(2, 7, streets)
    total = int(points.sum())
    start_x = np.repeat(rng.integers(-28_000, 28_000, streets), points)
    start_y = np.repeat(rng.integers(-18_000, 18_000, streets), points)
    walk_x = np.clip(start_x + rng.integers(-3000, 3000, total), -29_500, 29_500) / 10
    walk_y = np.clip(start_y + rng.integers(-3000, 3000, total), -19_500, 19_500) / 10
    anchor_xy = np.array([point for _, line in anchors for point in line], dtype=np.float64)
    last_xy = np.array([[1234.5, -987.5]], dtype=np.float64)
    return RoadColumns(
        points=np.concatenate(([len(line) for _, line in anchors], points, [1])),
        road_class=np.concatenate(
            ([road_class for road_class, _ in anchors], rng.integers(0, 3, streets), [1])
        ),
        x=np.concatenate((anchor_xy[:, 0], walk_x, last_xy[:, 0])),
        y=np.concatenate((anchor_xy[:, 1], walk_y, last_xy[:, 1])),
    )


def _residents(rng: np.random.Generator, buildings: BuildingColumns) -> ResidentColumns:
    rows = 150
    home = rng.integers(0, buildings.x.size, rows)
    home[:5] = np.arange(2, 7)
    floor = rng.integers(0, 12, rows)
    floor[-1] = 255
    return ResidentColumns(
        x=np.clip(buildings.x[home] + _tenths(rng, -20, 20, rows), -_HALF_X, _HALF_X),
        y=np.clip(buildings.y[home] + _tenths(rng, -20, 20, rows), -_HALF_Y, _HALF_Y),
        floor=floor,
        building=home,
        segment=rng.integers(0, len(_SEGMENTS), rows),
    )


def _residents_engine(rng: np.random.Generator, rows: int) -> ResidentEngineColumns:
    persona_ref = rng.integers(0, 5000, rows)
    persona_ref[-1] = NO_PERSONA
    return ResidentEngineColumns(
        hex=rng.integers(0, _HEX_COUNT, rows),
        persona_ref=persona_ref,
        weight=rng.integers(1, 40, rows) / 10,
    )


def _stores() -> StoreColumns:
    return StoreColumns(
        x=np.array([-1200.5, -300.25, 450.125, 1450.75]),
        y=np.array([800.5, -1100.25, 300.75, -600.1]),
        hex=np.array([0, 3, 7, 11]),
        radius_m=np.array([2500.0, 1800.5, 2200.25, 3000.1]),
        district=np.array([0, 3, 1, 1]),
    )


def _river() -> RiverColumns:
    # The river runs past the bbox. The holes run clockwise: up the west side and back over the top.
    rings = [
        [(-6000.0, -300.0), (6000.0, -500.0), (6000.0, 200.0), (-6000.0, 400.0)],
        [(-200.5, -100.25), (-220.125, 80.1), (150.75, 90.5), (170.1, -110.3)],
        [(1800.1, 1200.2), (2200.3, 1250.4), (2000.5, 1600.6)],
        [
            (-2500.0, -1800.0),
            (-1500.0, -1800.0),
            (-1200.0, -1300.0),
            (-2000.0, -900.0),
            (-2700.0, -1300.0),
        ],
        [(-2200.0, -1500.0), (-2150.0, -1300.0), (-2000.0, -1500.0)],
        [(-1800.0, -1600.0), (-1750.0, -1450.0), (-1650.0, -1600.0)],
    ]
    flat = np.array([point for ring in rings for point in ring], dtype=np.float64)
    return RiverColumns(
        offsets=np.concatenate(([0], np.cumsum([len(ring) for ring in rings]))),
        x=flat[:, 0],
        y=flat[:, 1],
        kind=np.array([0, 1, 1]),
        rings=np.array([2, 1, 3]),
    )


def _graph(rng: np.random.Generator, residents: int) -> GraphColumns:
    degree = rng.integers(0, 5, residents)
    degree[3] = 0
    ties = int(degree.sum())
    return GraphColumns(
        offsets=np.concatenate(([0], np.cumsum(degree))),
        targets=rng.integers(0, residents, ties),
        kind=rng.integers(0, 4, ties),
        strength=rng.integers(1, 1000, ties) / 1000,
    )


def _travel(rng: np.random.Generator, rows: int) -> NDArray[np.int64]:
    minutes = rng.integers(1, 61, rows)
    minutes[0], minutes[-1] = 1, 65535
    return minutes


def _probes(reader: PackReader) -> dict[str, Any]:
    rng = np.random.default_rng(_PROBE_SEED)
    buildings = reader.buildings()
    detail = reader.detail()
    assert buildings.tile_start is not None
    boundaries = _interior_tiles(buildings.tile_start)
    return {
        "pack_hash": reader.pack_hash,
        "tile_start": _tile_rows(buildings.tile_start, boundaries),
        "buildings": _building_rows(reader, buildings, boundaries, rng),
        "detail": [] if detail is None else _building_rows(reader, detail, [], rng),
        "roads": _road_rows(reader.roads(), rng),
        "residents": _resident_rows(reader, rng),
        "river": _river_rows(reader),
        "stores": _store_rows(reader),
    }


def _ends_and_picks(size: int, picks: int, rng: np.random.Generator) -> list[int]:
    return [0, size - 1, *rng.integers(0, size, picks).tolist()]


def _interior_tiles(tile_start: NDArray[np.uint32]) -> list[int]:
    """Tiles with buildings on both sides of their start: the first, middle and last of those."""
    interior = np.flatnonzero((tile_start > 0) & (tile_start < tile_start[-1]))
    return [int(interior[i]) for i in (0, interior.size // 2, -1)]


def _tile_rows(tile_start: NDArray[np.uint32], boundaries: list[int]) -> list[dict[str, int]]:
    empty = np.flatnonzero(tile_start[1:] == tile_start[:-1])
    tiles = np.unique([0, *boundaries, int(empty[0]), tile_start.size - 1])
    return [{"tile": int(t), "start": int(tile_start[t])} for t in tiles]


def _building_rows(
    reader: PackReader, b: PackBuildings, boundaries: list[int], rng: np.random.Generator
) -> list[dict[str, Any]]:
    around_tiles: list[int] = []
    if b.tile_start is not None:
        for tile in boundaries:
            start = int(b.tile_start[tile])
            around_tiles += [start - 1, start]
    ties = np.flatnonzero(b.code[1:] == b.code[:-1])
    rows = np.unique(
        [
            *_ends_and_picks(b.code.size, 8, rng),
            *around_tiles,
            *(ties[:3] if ties.size else []),
            int(b.width.argmax()),
            int(b.depth.argmax()),
            int(b.height.argmax()),
        ]
    )
    x_m, y_m = reader.x_m(b.qx), reader.y_m(b.qy)
    width_m, depth_m, height_m, rotation_rad = b.width_m, b.depth_m, b.height_m, b.rotation_rad
    return [
        {
            "index": int(i),
            "qx": int(b.qx[i]),
            "qy": int(b.qy[i]),
            "width": int(b.width[i]),
            "depth": int(b.depth[i]),
            "height": int(b.height[i]),
            "rotation": int(b.rotation[i]),
            "district": int(b.district[i]),
            "x_m": float(x_m[i]),
            "y_m": float(y_m[i]),
            "width_m": float(width_m[i]),
            "depth_m": float(depth_m[i]),
            "height_m": float(height_m[i]),
            "rotation_rad": float(rotation_rad[i]),
        }
        for i in rows
    ]


def _road_rows(r: PackRoads, rng: np.random.Generator) -> list[dict[str, Any]]:
    offsets = r.offsets
    road_of_point = np.repeat(np.arange(r.points.size), r.points)
    jumps: list[int] = []
    for q in (r.qx, r.qy):
        after = np.flatnonzero(np.abs(np.diff(q.astype(np.int64))) > 32767)[:8] + 1
        jumps += [*road_of_point[after - 1].tolist(), *road_of_point[after].tolist()]
    rows = np.unique(
        [
            *_ends_and_picks(r.points.size, 4, rng),
            *np.flatnonzero(r.points == 1).tolist(),
            *jumps,
        ]
    )
    return [
        {
            "road": int(i),
            "class": int(r.road_class[i]),
            "points": [
                [int(r.qx[p]), int(r.qy[p])] for p in range(int(offsets[i]), int(offsets[i + 1]))
            ],
        }
        for i in rows
    ]


def _resident_rows(reader: PackReader, rng: np.random.Generator) -> list[dict[str, Any]]:
    r = reader.residents()
    z_m = r.z_m
    return [
        {
            "index": int(i),
            "qx": int(r.qx[i]),
            "qy": int(r.qy[i]),
            "floor": int(r.floor[i]),
            "z_m": float(z_m[i]),
            "building": int(r.building[i]),
            "segment": int(r.segment[i]),
        }
        for i in np.unique(_ends_and_picks(r.qx.size, 10, rng))
    ]


def _river_rows(reader: PackReader) -> list[dict[str, Any]]:
    r = reader.river()
    offsets = r.offsets.astype(np.int64)
    ring_sizes = np.diff(offsets)
    first_ring = np.cumsum(r.rings, dtype=np.int64) - r.rings
    xy = np.stack([r.x, r.y], axis=1).tolist()
    return [
        {
            "polygon": polygon,
            "kind": int(r.kind[polygon]),
            "rings": ring_sizes[start : start + r.rings[polygon]].tolist(),
            "first_point": [float(r.x[offsets[start]]), float(r.y[offsets[start]])],
            "points": [
                xy[offsets[ring] : offsets[ring + 1]]
                for ring in range(start, start + int(r.rings[polygon]))
            ],
        }
        for polygon, start in enumerate(first_ring.tolist())
    ]


def _store_rows(reader: PackReader) -> list[dict[str, Any]]:
    s = reader.stores()
    return [
        {
            "index": i,
            "x": float(s.x[i]),
            "y": float(s.y[i]),
            "hex": int(s.hex[i]),
            "radius_m": float(s.radius_m[i]),
            "district": int(s.district[i]),
        }
        for i in range(s.x.size)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m twl_pack.fixture",
        description="Write the small golden pack and its probes.",
    )
    parser.add_argument("out_dir", type=Path, help="directory to write the pack to")
    args = parser.parse_args(argv)
    write_fixture(args.out_dir)


if __name__ == "__main__":
    main()
