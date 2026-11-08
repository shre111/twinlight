"""Encode a pack in memory and write it out (ARCHITECTURE.md section 5)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from blake3 import blake3
from numpy.typing import NDArray

from twl_pack.checks import check_contents
from twl_pack.codecs import (
    compress,
    delta_wrap_encode,
    morton_encode,
    pad,
    quantise,
    rotation_encode,
    size_encode,
    tile_starts,
    to_planes,
)
from twl_pack.format import (
    COMPRESSION,
    DETAIL_FILE,
    ENGINE_BUDGET,
    ENGINE_FILES,
    FIRST_LOAD_BUDGET,
    FIRST_LOAD_FILES,
    PACK_FORMAT,
    PACK_VERSION,
    PackError,
)
from twl_pack.types import (
    BuildingColumns,
    GraphColumns,
    PackContents,
    ResidentColumns,
    ResidentEngineColumns,
    RiverColumns,
    RoadColumns,
    StoreColumns,
)

Bbox = tuple[int, int, int, int]
_SIZE_COLUMNS = ("width", "depth", "height")


@dataclass(frozen=True, eq=False)
class EncodedPack:
    manifest: bytes
    pack_hash: str
    raw: dict[str, bytes]
    objects: dict[str, bytes]
    building_order: NDArray[np.int64]
    detail_order: NDArray[np.int64]
    clipped: dict[str, int]


def encode_pack(contents: PackContents) -> EncodedPack:
    c = check_contents(contents)
    bbox = _bbox(c)
    buildings, building_order, clipped = _encode_buildings(c.buildings, bbox, c.tile_level)
    detail_order = np.empty(0, dtype=np.int64)
    raw = {
        "buildings.bin": buildings,
        "districts.json": _json("districts.json", c.districts),
        "graph.bin": _encode_graph(c.graph),
        "residents.bin": _encode_residents(c.residents, bbox, building_order),
        "residents_engine.bin": _encode_residents_engine(c.residents_engine),
        "river.bin": _encode_river(c.river),
        "roads.bin": _encode_roads(c.roads, bbox),
        "stores.bin": _encode_stores(c.stores),
    }
    if c.detail is not None:
        raw[DETAIL_FILE], detail_order, detail_clipped = _encode_buildings(c.detail, bbox, None)
        clipped = {name: clipped[name] + detail_clipped[name] for name in _SIZE_COLUMNS}
    if c.hexes is not None and c.travel is not None:
        raw["hexes.bin"] = pad(c.hexes.tobytes())
        raw["travel.bin"] = pad(c.travel.tobytes())
    raw = dict(sorted(raw.items()))
    objects = {name: compress(data) for name, data in raw.items()}
    manifest = _json("manifest.json", _manifest(c, bbox, raw, objects))
    return EncodedPack(
        manifest=manifest,
        pack_hash=blake3(manifest).hexdigest(),
        raw=raw,
        objects=objects,
        building_order=building_order,
        detail_order=detail_order,
        clipped=clipped,
    )


def write_pack(encoded: EncodedPack, out_dir: Path) -> None:
    """Write the compressed objects and the manifest; other files in `out_dir` stay as they are."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in encoded.objects.items():
        (out_dir / name).write_bytes(data)
    (out_dir / "manifest.json").write_bytes(encoded.manifest)


def _bbox(c: PackContents) -> Bbox:
    with_positions = [c.buildings, c.roads, c.residents, c.stores]
    if c.detail is not None:
        with_positions.append(c.detail)
    filled = [p for p in with_positions if p.x.size]
    min_x = math.floor(min(float(p.x.min()) for p in filled))
    min_y = math.floor(min(float(p.y.min()) for p in filled))
    max_x = math.ceil(max(float(p.x.max()) for p in filled))
    max_y = math.ceil(max(float(p.y.max()) for p in filled))
    return (min_x, min_y, max(max_x, min_x + 1), max(max_y, min_y + 1))


def _quantised(
    x: NDArray[Any], y: NDArray[Any], bbox: Bbox
) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    return quantise(x, bbox[0], bbox[2]), quantise(y, bbox[1], bbox[3])


def _join(*columns: bytes) -> bytes:
    return b"".join(pad(column) for column in columns)


def _encode_buildings(
    b: BuildingColumns, bbox: Bbox, tile_level: int | None
) -> tuple[bytes, NDArray[np.int64], dict[str, int]]:
    """Columns in Morton order, the input row of each pack row, and how many sizes clipped."""
    code = morton_encode(*_quantised(b.x, b.y, bbox))
    order = np.argsort(code, kind="stable").astype(np.int64)
    sorted_code = code[order]
    tiles = [] if tile_level is None else [tile_starts(sorted_code, tile_level).tobytes()]
    width, clipped_width = size_encode(b.width[order])
    depth, clipped_depth = size_encode(b.depth[order])
    height, clipped_height = size_encode(b.height[order])
    data = _join(
        *tiles,
        to_planes(np.diff(sorted_code.astype(np.int64), prepend=0), 4),
        width.tobytes(),
        depth.tobytes(),
        height.tobytes(),
        rotation_encode(b.rotation[order]).tobytes(),
        b.district[order].tobytes(),
    )
    clipped = {"width": clipped_width, "depth": clipped_depth, "height": clipped_height}
    return data, order, clipped


def _encode_roads(r: RoadColumns, bbox: Bbox) -> bytes:
    qx, qy = _quantised(r.x, r.y, bbox)
    return _join(
        r.points.tobytes(),
        r.road_class.tobytes(),
        to_planes(delta_wrap_encode(qx), 2),
        to_planes(delta_wrap_encode(qy), 2),
    )


def _encode_residents(r: ResidentColumns, bbox: Bbox, building_order: NDArray[np.int64]) -> bytes:
    qx, qy = _quantised(r.x, r.y, bbox)
    pack_row = np.empty(building_order.size, dtype=np.int64)
    pack_row[building_order] = np.arange(building_order.size, dtype=np.int64)
    return _join(
        qx.tobytes(),
        qy.tobytes(),
        r.floor.tobytes(),
        pack_row[r.building].astype("<u4").tobytes(),
        r.segment.tobytes(),
    )


def _encode_residents_engine(r: ResidentEngineColumns) -> bytes:
    return _join(r.hex.tobytes(), r.persona_ref.tobytes(), r.weight.tobytes(), r.tier.tobytes())


def _encode_stores(s: StoreColumns) -> bytes:
    return _join(
        s.x.astype("<f4").tobytes(),
        s.y.astype("<f4").tobytes(),
        s.hex.tobytes(),
        s.radius_m.tobytes(),
        s.district.tobytes(),
    )


def _encode_river(r: RiverColumns) -> bytes:
    return _join(
        r.offsets.tobytes(),
        r.x.tobytes(),
        r.y.tobytes(),
        r.kind.tobytes(),
        r.rings.tobytes(),
    )


def _encode_graph(g: GraphColumns) -> bytes:
    return _join(
        g.offsets.tobytes(),
        g.targets.tobytes(),
        g.kind.tobytes(),
        g.strength.tobytes(),
    )


def _manifest(
    c: PackContents, bbox: Bbox, raw: dict[str, bytes], objects: dict[str, bytes]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "attributions": sorted(set(c.attributions)),
        "bands": [{"name": b.name, "start_minute": int(b.start_minute)} for b in c.bands],
        "bbox": list(bbox),
        "compression": COMPRESSION,
        "counts": _counts(c),
        "crs": c.crs,
        "files": {name: blake3(data).hexdigest() for name, data in raw.items()},
        "format": PACK_FORMAT,
        "kind": c.kind,
        "name": c.name,
        "segments": list(c.segments),
        "sizes": _sizes(objects),
        "timezone": c.timezone,
        "version": PACK_VERSION,
        "weekend_days": [int(day) for day in c.weekend_days],
    }
    if c.seed is not None:
        manifest["seed"] = int(c.seed)
    return manifest


def _counts(c: PackContents) -> dict[str, int]:
    return {
        "bands": len(c.bands),
        "buildingTileLevel": int(c.tile_level),
        "buildings": int(c.buildings.x.size),
        "detailBuildings": 0 if c.detail is None else int(c.detail.x.size),
        "hexes": 0 if c.hexes is None else int(c.hexes.size),
        "residents": int(c.residents.x.size),
        "riverPoints": int(c.river.x.size),
        "riverPolygons": int(c.river.kind.size),
        "riverRings": int(c.river.offsets.size - 1),
        "roadPoints": int(c.roads.x.size),
        "roads": int(c.roads.points.size),
        "stores": int(c.stores.x.size),
        "ties": int(c.graph.targets.size),
    }


def _sizes(objects: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    first_load = sorted(name for name in FIRST_LOAD_FILES if name in objects)
    engine = sorted(name for name in ENGINE_FILES if name in objects)
    detail = [DETAIL_FILE] if DETAIL_FILE in objects else []
    return {
        "detail": {"bytes": _total(objects, detail), "files": detail},
        "engine": {"budget": ENGINE_BUDGET, "bytes": _total(objects, engine), "files": engine},
        "first_load": {
            "budget": FIRST_LOAD_BUDGET,
            "bytes": _total(objects, first_load),
            "files": first_load,
        },
    }


def _total(objects: dict[str, bytes], names: list[str]) -> int:
    return sum(len(objects[name]) for name in names)


def _json(name: str, value: object) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PackError(f"{name} cannot be written as JSON: {error}") from error
    return text.encode()
