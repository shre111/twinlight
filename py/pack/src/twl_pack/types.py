"""What a pack is built from, in natural units; arrays may be any numeric dtype."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from twl_pack.format import DEFAULT_TILE_LEVEL


@dataclass(frozen=True, eq=False)
class BuildingColumns:
    x: NDArray[Any]
    y: NDArray[Any]
    width: NDArray[Any]
    depth: NDArray[Any]
    height: NDArray[Any]
    rotation: NDArray[Any]
    district: NDArray[Any]


@dataclass(frozen=True, eq=False)
class RoadColumns:
    points: NDArray[Any]
    road_class: NDArray[Any]
    x: NDArray[Any]
    y: NDArray[Any]


@dataclass(frozen=True, eq=False)
class ResidentColumns:
    """`building` indexes the `BuildingColumns` as given, not the sorted pack order."""

    x: NDArray[Any]
    y: NDArray[Any]
    floor: NDArray[Any]
    building: NDArray[Any]
    segment: NDArray[Any]


@dataclass(frozen=True, eq=False)
class ResidentEngineColumns:
    """`tier` is the household affluence tier (D-035): 0 low, 1 middle, 2 high."""

    hex: NDArray[Any]
    persona_ref: NDArray[Any]
    weight: NDArray[Any]
    tier: NDArray[Any]


@dataclass(frozen=True, eq=False)
class StoreColumns:
    x: NDArray[Any]
    y: NDArray[Any]
    hex: NDArray[Any]
    radius_m: NDArray[Any]
    district: NDArray[Any]


@dataclass(frozen=True, eq=False)
class RiverColumns:
    """`offsets` has one more entry than there are rings; `kind` and `rings` are per polygon."""

    offsets: NDArray[Any]
    x: NDArray[Any]
    y: NDArray[Any]
    kind: NDArray[Any]
    rings: NDArray[Any]


@dataclass(frozen=True, eq=False)
class GraphColumns:
    offsets: NDArray[Any]
    targets: NDArray[Any]
    kind: NDArray[Any]
    strength: NDArray[Any]


@dataclass(frozen=True)
class Band:
    name: str
    start_minute: int


@dataclass(frozen=True, eq=False)
class PackContents:
    """`timezone` is an IANA name and `weekend_days` ISO weekdays, ascending (D-036)."""

    kind: str
    name: str
    crs: str
    timezone: str
    weekend_days: Sequence[int]
    segments: list[str]
    bands: list[Band]
    attributions: list[str]
    districts: list[dict[str, Any]]
    buildings: BuildingColumns
    roads: RoadColumns
    residents: ResidentColumns
    residents_engine: ResidentEngineColumns
    stores: StoreColumns
    river: RiverColumns
    graph: GraphColumns
    detail: BuildingColumns | None = None
    hexes: NDArray[Any] | None = None
    travel: NDArray[Any] | None = None
    seed: int | None = None
    tile_level: int = DEFAULT_TILE_LEVEL
