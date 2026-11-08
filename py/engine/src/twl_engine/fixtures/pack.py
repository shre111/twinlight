"""Toy city packs, written by `twl_pack` in the v2 layout (ARCHITECTURE.md section 5)."""

import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import twl_pack
from twl_engine.fixtures.types import ToyCity
from twl_pack.format import FLOOR_HEIGHT_M

SEGMENTS = ("age_16_24", "age_25_34", "age_35_49", "age_50_64", "age_65_80")
TILE_LEVEL = 4
TOY_TIER = 1


def write_pack(city: ToyCity, out_dir: Path) -> str:
    """Write the pack files and manifest (other files in `out_dir` are left as they are).

    Returns `pack_hash`, the blake3 of the manifest bytes.
    """
    encoded = twl_pack.encode_pack(toy_contents(city))
    twl_pack.write_pack(encoded, out_dir)
    return encoded.pack_hash


def pack_hash(city: ToyCity) -> str:
    """The `pack_hash` `write_pack` would return, without writing anything."""
    return twl_pack.encode_pack(toy_contents(city)).pack_hash


def toy_contents(city: ToyCity) -> twl_pack.PackContents:
    b, r, s = city.buildings, city.residents, city.stores
    return twl_pack.PackContents(
        kind="toy",
        name="toy",
        crs="local",
        timezone=city.timezone,
        weekend_days=tuple(city.weekend_days),
        seed=int(city.seed),
        segments=list(SEGMENTS),
        bands=[],
        attributions=[],
        districts=[dataclasses.asdict(d) for d in city.districts],
        buildings=twl_pack.BuildingColumns(
            x=b.x,
            y=b.y,
            width=b.width,
            depth=b.depth,
            height=b.height,
            rotation=b.rotation,
            district=b.district,
        ),
        roads=twl_pack.RoadColumns(
            points=np.diff(city.roads.offsets),
            road_class=city.roads.kind,
            x=city.roads.x,
            y=city.roads.y,
        ),
        residents=twl_pack.ResidentColumns(
            x=r.x, y=r.y, floor=pack_floors(r.z), building=r.building, segment=r.segment
        ),
        residents_engine=twl_pack.ResidentEngineColumns(
            hex=r.hex,
            persona_ref=r.persona_ref,
            weight=r.weight,
            tier=np.full(r.x.size, TOY_TIER, dtype=np.uint8),
        ),
        stores=twl_pack.StoreColumns(
            x=s.x, y=s.y, hex=s.hex, radius_m=s.radius_m, district=s.district
        ),
        river=twl_pack.RiverColumns(
            offsets=city.river.offsets,
            x=city.river.x,
            y=city.river.y,
            kind=city.river.kind,
            rings=np.ones(city.river.kind.size, dtype=np.uint16),
        ),
        graph=twl_pack.GraphColumns(
            offsets=city.ties.offsets,
            targets=city.ties.targets,
            kind=city.ties.kind,
            strength=city.ties.strength,
        ),
        tile_level=TILE_LEVEL,
    )


def pack_floors(z: NDArray[np.float32]) -> NDArray[np.uint8]:
    """The highest pack floor, `z = (floor + 0.5) * 3.2`, at or below each height (floor 0 below
    1.6 m), so a resident who was under their roof stays under it."""
    floors = np.floor(z.astype(np.float64) / FLOOR_HEIGHT_M - 0.5)
    return np.maximum(floors, 0).astype(np.uint8)
