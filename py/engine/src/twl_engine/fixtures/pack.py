"""Toy city packs in the v1 binary layout (ARCHITECTURE.md section 5)."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from blake3 import blake3
from numpy.typing import NDArray

from twl_engine.fixtures.types import ToyCity

PACK_FORMAT = "twinlight-pack"
PACK_VERSION = 1
COLUMN_ALIGN = 8

F4, U4, U2, U1 = "<f4", "<u4", "<u2", "<u1"

Column = tuple[NDArray[Any], str, int]


def write_pack(city: ToyCity, out_dir: Path) -> str:
    """Write the pack files and manifest (other files in `out_dir` are left as they are).

    Returns `pack_hash`, the blake3 of the manifest bytes.
    """
    counts = _counts(city)
    files = _files(city, counts)
    manifest = _json(
        {
            "attributions": [],
            "bbox": _bbox(city),
            "counts": counts,
            "crs": "local",
            "files": {name: blake3(data).hexdigest() for name, data in sorted(files.items())},
            "format": PACK_FORMAT,
            "kind": "toy",
            "name": "toy",
            "seed": int(city.seed),
            "version": PACK_VERSION,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(files.items()):
        (out_dir / name).write_bytes(data)
    (out_dir / "manifest.json").write_bytes(manifest)
    return blake3(manifest).hexdigest()


def _counts(city: ToyCity) -> dict[str, int]:
    return {
        "buildings": int(city.buildings.x.size),
        "residents": int(city.residents.x.size),
        "ties": int(city.ties.targets.size),
        "stores": int(city.stores.x.size),
        "roads": int(city.roads.kind.size),
        "roadPoints": int(city.roads.x.size),
        "riverPolygons": int(city.river.offsets.size - 1),
        "riverPoints": int(city.river.x.size),
    }


def _files(city: ToyCity, counts: dict[str, int]) -> dict[str, bytes]:
    b, r, t, s = city.buildings, city.residents, city.ties, city.stores
    roads, river = city.roads, city.river
    nb, nr, nt, ns = counts["buildings"], counts["residents"], counts["ties"], counts["stores"]
    road_points, river_points = counts["roadPoints"], counts["riverPoints"]
    return {
        "buildings.bin": _columns(
            [
                (b.x, F4, nb),
                (b.y, F4, nb),
                (b.width, F4, nb),
                (b.depth, F4, nb),
                (b.height, F4, nb),
                (b.rotation, F4, nb),
                (b.district, U2, nb),
            ]
        ),
        "residents.bin": _columns(
            [
                (r.x, F4, nr),
                (r.y, F4, nr),
                (r.z, F4, nr),
                (r.building, U4, nr),
                (r.hex, U4, nr),
                (r.segment, U2, nr),
                (r.persona_ref, U4, nr),
                (r.weight, F4, nr),
            ]
        ),
        "graph.bin": _columns(
            [
                (t.offsets, U4, nr + 1),
                (t.targets, U4, nt),
                (t.kind, U1, nt),
                (t.strength, F4, nt),
            ]
        ),
        "stores.bin": _columns(
            [
                (s.x, F4, ns),
                (s.y, F4, ns),
                (s.hex, U4, ns),
                (s.radius_m, F4, ns),
                (s.district, U2, ns),
            ]
        ),
        "roads.bin": _columns(
            [
                (roads.offsets, U4, counts["roads"] + 1),
                (roads.x, F4, road_points),
                (roads.y, F4, road_points),
                (roads.kind, U1, counts["roads"]),
            ]
        ),
        "river.bin": _columns(
            [
                (river.offsets, U4, counts["riverPolygons"] + 1),
                (river.x, F4, river_points),
                (river.y, F4, river_points),
            ]
        ),
        "districts.json": _json([dataclasses.asdict(d) for d in city.districts]),
    }


def _columns(columns: list[Column]) -> bytes:
    """Little-endian columns, each zero-padded to a multiple of 8 bytes."""
    out = bytearray()
    for values, dtype, length in columns:
        if values.shape != (length,):
            raise ValueError(f"column of shape {values.shape} where ({length},) was expected")
        out += np.ascontiguousarray(values.astype(dtype, casting="safe")).tobytes()
        out += bytes(-len(out) % COLUMN_ALIGN)
    return bytes(out)


def _bbox(city: ToyCity) -> list[float]:
    """Bounds of building centres, roads, residents and stores; the river runs on past them."""
    parts = (city.buildings, city.roads, city.residents, city.stores)
    x = np.concatenate([p.x for p in parts])
    y = np.concatenate([p.y for p in parts])
    return [
        float(math.floor(x.min())),
        float(math.floor(y.min())),
        float(math.ceil(x.max())),
        float(math.ceil(y.max())),
    ]


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
