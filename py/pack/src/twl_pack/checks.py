"""Everything `encode_pack` refuses, each raising `PackError` naming the column."""

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from twl_pack.codecs import as_uint
from twl_pack.format import MAX_DISTRICTS, MAX_TILE_LEVEL, PackError
from twl_pack.local_time import check_timezone, check_weekend_days
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

_U8_MAX = 0xFF
_U16_MAX = 0xFFFF
_U32_MAX = 0xFFFFFFFF
_U64_MAX = 0xFFFFFFFFFFFFFFFF
_MINUTES_PER_DAY = 1440
_ROAD_CLASS_MAX = 2
_TIER_MAX = 2
_TIE_KIND_MAX = 3
_RIVER_KIND_MAX = 1
_MIN_RING_POINTS = 3
_F32_MAX = float(np.finfo(np.float32).max)
_DISTRICT_KEYS = ("id", "name", "x", "y", "east")


def check_contents(contents: PackContents) -> PackContents:
    """A copy with integer and `f32` columns in their stored dtype and the rest in `float64`.

    Building, road and resident positions stay `float64` until quantised, and store positions
    until written, so that `bbox` is taken over the values as given.
    """
    c = contents
    _check_tile_level(c.tile_level)
    timezone = check_timezone(c.timezone)
    weekend_days = check_weekend_days(c.weekend_days)
    district_ids = _check_districts(c.districts)
    _check_bands(c.bands)
    buildings = _buildings("buildings", c.buildings, district_ids)
    if buildings.x.size == 0:
        raise PackError("buildings is empty: a pack needs at least one building")
    detail = None if c.detail is None else _buildings("detail", c.detail, district_ids)
    if detail is not None and detail.x.size == 0:
        detail = None
    _check_kind(c, detail)

    hexes = None if c.hexes is None else _uints("hexes", c.hexes, None, "<u8", high=_U64_MAX)
    hex_high = 0 if hexes is None else hexes.size - 1
    residents = _residents(c.residents, buildings.x.size, len(c.segments))
    residents_engine = _residents_engine(c.residents_engine, residents.x.size, hex_high)
    stores = _stores(c.stores, district_ids, hex_high)
    travel = None
    if c.travel is not None and hexes is not None:
        rows = stores.x.size * hexes.size * len(c.bands)
        travel = _uints("travel", c.travel, rows, "<u2", low=1, high=_U16_MAX)
    return replace(
        c,
        timezone=timezone,
        weekend_days=weekend_days,
        buildings=buildings,
        detail=detail,
        roads=_roads(c.roads),
        residents=residents,
        residents_engine=residents_engine,
        stores=stores,
        river=_river(c.river),
        graph=_graph(c.graph, residents.x.size),
        hexes=hexes,
        travel=travel,
    )


def _column(column: str, values: ArrayLike, length: int | None) -> NDArray[Any]:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise PackError(f"{column} must be numeric, got {array.dtype}")
    if array.ndim != 1:
        raise PackError(f"{column} must be one-dimensional, got shape {array.shape}")
    if length is not None and array.size != length:
        raise PackError(f"{column} has {array.size} rows, expected {length}")
    return array


def _floats(column: str, values: ArrayLike, length: int | None = None) -> NDArray[np.float64]:
    stored = _column(column, values, length).astype(np.float64, copy=False)
    if not np.isfinite(stored).all():
        raise PackError(f"{column} has NaN or infinity")
    return stored


def _fits_f32(column: str, values: ArrayLike, length: int | None = None) -> NDArray[np.float64]:
    stored = _floats(column, values, length)
    if stored.size and np.abs(stored).max() > _F32_MAX:
        raise PackError(f"{column} has values too large for f32")
    return stored


def _f32(column: str, values: ArrayLike, length: int | None = None) -> NDArray[np.float32]:
    return _fits_f32(column, values, length).astype("<f4")


def _uints(
    column: str,
    values: ArrayLike,
    length: int | None,
    dtype: DTypeLike,
    *,
    low: int = 0,
    high: int,
) -> NDArray[Any]:
    return as_uint(_column(column, values, length), dtype, column, low=low, high=high)


def _check_tile_level(level: int) -> None:
    if not isinstance(level, int | np.integer) or not 0 <= level <= MAX_TILE_LEVEL:
        raise PackError(f"tile_level must be in 0..{MAX_TILE_LEVEL}, got {level}")


def _check_kind(c: PackContents, detail: BuildingColumns | None) -> None:
    if c.kind == "toy":
        if c.seed is None:
            raise PackError("seed is required for a toy pack")
        extras = (("detail", detail), ("hexes", c.hexes), ("travel", c.travel))
        for name, value in extras:
            if value is not None:
                raise PackError(f"a toy pack cannot have {name}")
        if c.bands:
            raise PackError("a toy pack cannot have bands")
    elif c.kind == "city":
        if c.seed is not None:
            raise PackError("seed is only for a toy pack")
        if c.hexes is None or np.size(c.hexes) == 0:
            raise PackError("a city pack needs hexes")
        if c.travel is None:
            raise PackError("a city pack needs travel")
        if not c.bands:
            raise PackError("a city pack needs bands")
    else:
        raise PackError(f"kind must be toy or city, got {c.kind!r}")


def _check_bands(bands: list[Band]) -> None:
    previous = -1
    for band in bands:
        minute = band.start_minute
        if not isinstance(minute, int | np.integer):
            raise PackError(f"bands start_minute must be an integer, got {minute!r}")
        if not 0 <= minute < _MINUTES_PER_DAY:
            raise PackError(f"bands start_minute {minute} is outside 0..{_MINUTES_PER_DAY - 1}")
        if minute <= previous:
            raise PackError(f"bands must ascend strictly by start_minute, got {minute}")
        previous = int(minute)


def _check_districts(districts: list[dict[str, Any]]) -> NDArray[np.int64]:
    if len(districts) > MAX_DISTRICTS:
        raise PackError(f"districts has {len(districts)} entries, the most is {MAX_DISTRICTS}")
    previous = -1
    for district in districts:
        for key in _DISTRICT_KEYS:
            if key not in district:
                raise PackError(f"districts entry {district!r} is missing {key}")
        district_id = district["id"]
        if not isinstance(district_id, int) or isinstance(district_id, bool):
            raise PackError(f"districts id must be an integer, got {district_id!r}")
        if district_id <= previous:
            raise PackError(f"districts ids must ascend strictly and be >= 0, got {district_id}")
        if district_id > _U8_MAX:
            raise PackError(f"districts id {district_id} does not fit the u8 building column")
        previous = district_id
    return np.array([d["id"] for d in districts], dtype=np.int64)


def _district_column(
    column: str, values: ArrayLike, length: int, dtype: DTypeLike, high: int, ids: NDArray[np.int64]
) -> NDArray[Any]:
    stored = _uints(column, values, length, dtype, high=high)
    if not np.isin(stored, ids).all():
        raise PackError(f"{column} holds a value that is not a district id")
    return stored


def _size_column(column: str, values: ArrayLike, length: int) -> NDArray[np.float64]:
    stored = _floats(column, values, length)
    if (stored < 0).any():
        raise PackError(f"{column} has a negative size")
    return stored


def _buildings(name: str, b: BuildingColumns, ids: NDArray[np.int64]) -> BuildingColumns:
    x = _floats(f"{name}.x", b.x)
    n = x.size
    return BuildingColumns(
        x=x,
        y=_floats(f"{name}.y", b.y, n),
        width=_size_column(f"{name}.width", b.width, n),
        depth=_size_column(f"{name}.depth", b.depth, n),
        height=_size_column(f"{name}.height", b.height, n),
        rotation=_floats(f"{name}.rotation", b.rotation, n),
        district=_district_column(f"{name}.district", b.district, n, "<u1", _U8_MAX, ids),
    )


def _roads(r: RoadColumns) -> RoadColumns:
    points = _uints("roads.points", r.points, None, "<u2", low=1, high=_U16_MAX)
    road_class = _uints("roads.road_class", r.road_class, points.size, "<u1", high=_ROAD_CLASS_MAX)
    x = _floats("roads.x", r.x)
    y = _floats("roads.y", r.y, x.size)
    total = int(points.sum(dtype=np.int64))
    if total != x.size:
        raise PackError(f"roads.points sums to {total} but there are {x.size} road points")
    return RoadColumns(points=points, road_class=road_class, x=x, y=y)


def _residents(r: ResidentColumns, buildings: int, segments: int) -> ResidentColumns:
    x = _floats("residents.x", r.x)
    n = x.size
    return ResidentColumns(
        x=x,
        y=_floats("residents.y", r.y, n),
        floor=_uints("residents.floor", r.floor, n, "<u1", high=_U8_MAX),
        building=_uints("residents.building", r.building, n, "<u4", high=buildings - 1),
        segment=_uints("residents.segment", r.segment, n, "<u1", high=min(segments - 1, _U8_MAX)),
    )


def _residents_engine(
    r: ResidentEngineColumns, residents: int, hex_high: int
) -> ResidentEngineColumns:
    weight = _f32("residents_engine.weight", r.weight, residents)
    if not (weight > 0).all():
        raise PackError("residents_engine.weight must be positive")
    return ResidentEngineColumns(
        hex=_uints("residents_engine.hex", r.hex, residents, "<u4", high=hex_high),
        persona_ref=_uints(
            "residents_engine.persona_ref", r.persona_ref, residents, "<u4", high=_U32_MAX
        ),
        weight=weight,
        tier=_uints("residents_engine.tier", r.tier, residents, "<u1", high=_TIER_MAX),
    )


def _stores(s: StoreColumns, ids: NDArray[np.int64], hex_high: int) -> StoreColumns:
    x = _fits_f32("stores.x", s.x)
    n = x.size
    return StoreColumns(
        x=x,
        y=_fits_f32("stores.y", s.y, n),
        hex=_uints("stores.hex", s.hex, n, "<u4", high=hex_high),
        radius_m=_f32("stores.radius_m", s.radius_m, n),
        district=_district_column("stores.district", s.district, n, "<u2", _U16_MAX, ids),
    )


def _graph(g: GraphColumns, residents: int) -> GraphColumns:
    offsets = _uints("graph.offsets", g.offsets, residents + 1, "<u4", high=_U32_MAX)
    targets = _uints("graph.targets", g.targets, None, "<u4", high=residents - 1)
    ties = targets.size
    if offsets[0] != 0:
        raise PackError(f"graph.offsets must start at 0, got {offsets[0]}")
    if offsets[-1] != ties:
        raise PackError(f"graph.offsets must end at the tie count {ties}, got {offsets[-1]}")
    if (offsets[1:] < offsets[:-1]).any():
        raise PackError("graph.offsets must not decrease")
    return GraphColumns(
        offsets=offsets,
        targets=targets,
        kind=_uints("graph.kind", g.kind, ties, "<u1", high=_TIE_KIND_MAX),
        strength=_f32("graph.strength", g.strength, ties),
    )


def _river(r: RiverColumns) -> RiverColumns:
    rings = _uints("river.rings", r.rings, None, "<u2", low=1, high=_U16_MAX)
    kind = _uints("river.kind", r.kind, rings.size, "<u1", high=_RIVER_KIND_MAX)
    x = _f32("river.x", r.x)
    y = _f32("river.y", r.y, x.size)
    offsets = _uints("river.offsets", r.offsets, None, "<u4", high=_U32_MAX)
    ring_count = offsets.size - 1
    if ring_count < 0:
        raise PackError("river.offsets needs at least one entry")
    total_rings = int(rings.sum(dtype=np.int64))
    if total_rings != ring_count:
        raise PackError(f"river.rings sums to {total_rings} but there are {ring_count} rings")
    if offsets[0] != 0 or offsets[-1] != x.size:
        raise PackError(f"river.offsets must run from 0 to the point count {x.size}")
    if (offsets[1:] < offsets[:-1]).any():
        raise PackError("river.offsets must not decrease")
    if ring_count > 0:
        _check_rings(offsets.astype(np.int64), x, y, rings)
    return RiverColumns(offsets=offsets, x=x, y=y, kind=kind, rings=rings)


def _check_rings(
    offsets: NDArray[np.int64],
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    rings: NDArray[np.uint16],
) -> None:
    starts, ends = offsets[:-1], offsets[1:]
    short = np.flatnonzero(ends - starts < _MIN_RING_POINTS)
    if short.size:
        raise PackError(f"river ring {short[0]} has under {_MIN_RING_POINTS} points")
    last = ends - 1
    closed = np.flatnonzero((x[starts] == x[last]) & (y[starts] == y[last]))
    if closed.size:
        raise PackError(f"river ring {closed[0]} is closed: its last point repeats its first")
    area = _ring_areas(starts, last, x, y)
    is_exterior = np.zeros(area.size, dtype=np.bool_)
    is_exterior[np.cumsum(rings, dtype=np.int64) - rings] = True
    backwards = np.flatnonzero(is_exterior & (area <= 0))
    if backwards.size:
        raise PackError(f"river ring {backwards[0]} is an exterior but not counter-clockwise")
    forwards = np.flatnonzero(~is_exterior & (area >= 0))
    if forwards.size:
        raise PackError(f"river ring {forwards[0]} is a hole but not clockwise")


def _ring_areas(
    starts: NDArray[np.int64],
    last: NDArray[np.int64],
    x: NDArray[np.float32],
    y: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Twice the shoelace area of each ring, positive when counter-clockwise.

    Taken over the stored `f32` values: their products are exact in `float64`.
    """
    following = np.arange(1, x.size + 1, dtype=np.int64)
    following[last] = starts
    px, py = x.astype(np.float64), y.astype(np.float64)
    return np.add.reduceat(px * py[following] - px[following] * py, starts)
