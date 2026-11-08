"""Read and verify a pack directory (ARCHITECTURE.md section 5)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from blake3 import blake3
from numpy.typing import ArrayLike, NDArray

from twl_pack.codecs import (
    content_size,
    decompress,
    delta_wrap_decode,
    dequantise,
    floor_z,
    from_planes,
    morton_decode,
    rotation_decode,
    size_decode,
)
from twl_pack.format import (
    COLUMN_ALIGN,
    COMPRESSION,
    DETAIL_FILE,
    ENGINE_FILES,
    FIRST_LOAD_FILES,
    MAX_TILE_LEVEL,
    PACK_FORMAT,
    PACK_VERSION,
    PackError,
)
from twl_pack.local_time import check_timezone, check_weekend_days

_U32_MAX = 0xFFFFFFFF
_MANIFEST_KEYS = (
    "attributions",
    "bands",
    "bbox",
    "compression",
    "counts",
    "crs",
    "files",
    "format",
    "kind",
    "name",
    "segments",
    "sizes",
    "timezone",
    "version",
    "weekend_days",
)
_COUNT_KEYS = (
    "bands",
    "buildingTileLevel",
    "buildings",
    "detailBuildings",
    "hexes",
    "residents",
    "riverPoints",
    "riverPolygons",
    "riverRings",
    "roadPoints",
    "roads",
    "stores",
    "ties",
)
_TOY_OMITS = ("hexes.bin", "travel.bin")


@dataclass(frozen=True, eq=False)
class PackBuildings:
    tile_start: NDArray[np.uint32] | None
    code: NDArray[np.uint32]
    qx: NDArray[np.uint16]
    qy: NDArray[np.uint16]
    width: NDArray[np.uint8]
    depth: NDArray[np.uint8]
    height: NDArray[np.uint8]
    rotation: NDArray[np.uint8]
    district: NDArray[np.uint8]

    @property
    def width_m(self) -> NDArray[np.float64]:
        return size_decode(self.width)

    @property
    def depth_m(self) -> NDArray[np.float64]:
        return size_decode(self.depth)

    @property
    def height_m(self) -> NDArray[np.float64]:
        return size_decode(self.height)

    @property
    def rotation_rad(self) -> NDArray[np.float64]:
        return rotation_decode(self.rotation)


@dataclass(frozen=True, eq=False)
class PackRoads:
    points: NDArray[np.uint16]
    road_class: NDArray[np.uint8]
    qx: NDArray[np.uint16]
    qy: NDArray[np.uint16]

    @property
    def offsets(self) -> NDArray[np.int64]:
        return np.concatenate(([0], np.cumsum(self.points, dtype=np.int64)), dtype=np.int64)


@dataclass(frozen=True, eq=False)
class PackResidents:
    qx: NDArray[np.uint16]
    qy: NDArray[np.uint16]
    floor: NDArray[np.uint8]
    building: NDArray[np.uint32]
    segment: NDArray[np.uint8]

    @property
    def z_m(self) -> NDArray[np.float64]:
        return floor_z(self.floor)


@dataclass(frozen=True, eq=False)
class PackResidentsEngine:
    hex: NDArray[np.uint32]
    persona_ref: NDArray[np.uint32]
    weight: NDArray[np.float32]
    tier: NDArray[np.uint8]


@dataclass(frozen=True, eq=False)
class PackStores:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    hex: NDArray[np.uint32]
    radius_m: NDArray[np.float32]
    district: NDArray[np.uint16]


@dataclass(frozen=True, eq=False)
class PackRiver:
    offsets: NDArray[np.uint32]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    kind: NDArray[np.uint8]
    rings: NDArray[np.uint16]


@dataclass(frozen=True, eq=False)
class PackGraph:
    offsets: NDArray[np.uint32]
    targets: NDArray[np.uint32]
    kind: NDArray[np.uint8]
    strength: NDArray[np.float32]


@dataclass(frozen=True)
class _Column:
    name: str
    dtype: str
    count: int
    planes: bool = False

    @property
    def nbytes(self) -> int:
        return self.count * np.dtype(self.dtype).itemsize

    @property
    def padded(self) -> int:
        return self.nbytes + (-self.nbytes % COLUMN_ALIGN)


def _building_layout(rows: int, tile_level: int | None) -> list[_Column]:
    tiles = [] if tile_level is None else [_Column("tile_start", "<u4", 4**tile_level + 1)]
    sizes = ("width", "depth", "height", "rotation", "district")
    return [
        *tiles,
        _Column("position_delta", "<u4", rows, planes=True),
        *(_Column(name, "<u1", rows) for name in sizes),
    ]


def _layout(name: str, counts: dict[str, int]) -> list[_Column] | None:
    """The columns of a `.bin` as `counts` implies them; `None` when the file has no fixed size."""
    residents, stores, ties = counts["residents"], counts["stores"], counts["ties"]
    roads, road_points = counts["roads"], counts["roadPoints"]
    rings, river_points = counts["riverRings"], counts["riverPoints"]
    match name:
        case "buildings.bin":
            return _building_layout(counts["buildings"], counts["buildingTileLevel"])
        case "buildings_detail.bin":
            return _building_layout(counts["detailBuildings"], None)
        case "roads.bin":
            return [
                _Column("points", "<u2", roads),
                _Column("road_class", "<u1", roads),
                _Column("dx", "<u2", road_points, planes=True),
                _Column("dy", "<u2", road_points, planes=True),
            ]
        case "residents.bin":
            return [
                _Column("qx", "<u2", residents),
                _Column("qy", "<u2", residents),
                _Column("floor", "<u1", residents),
                _Column("building", "<u4", residents),
                _Column("segment", "<u1", residents),
            ]
        case "residents_engine.bin":
            return [
                _Column("hex", "<u4", residents),
                _Column("persona_ref", "<u4", residents),
                _Column("weight", "<f4", residents),
                _Column("tier", "<u1", residents),
            ]
        case "stores.bin":
            return [
                _Column("x", "<f4", stores),
                _Column("y", "<f4", stores),
                _Column("hex", "<u4", stores),
                _Column("radius_m", "<f4", stores),
                _Column("district", "<u2", stores),
            ]
        case "river.bin":
            return [
                _Column("offsets", "<u4", rings + 1),
                _Column("x", "<f4", river_points),
                _Column("y", "<f4", river_points),
                _Column("kind", "<u1", counts["riverPolygons"]),
                _Column("rings", "<u2", counts["riverPolygons"]),
            ]
        case "graph.bin":
            return [
                _Column("offsets", "<u4", residents + 1),
                _Column("targets", "<u4", ties),
                _Column("kind", "<u1", ties),
                _Column("strength", "<f4", ties),
            ]
        case "hexes.bin":
            return [_Column("h3", "<u8", counts["hexes"])]
        case "travel.bin":
            return [_Column("minutes", "<u2", stores * counts["hexes"] * counts["bands"])]
        case _:
            return None


class PackReader:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        try:
            data = path.read_bytes()
        except OSError as error:
            raise PackError(f"cannot read {path}: {error}") from error
        self.pack_hash: str = blake3(data).hexdigest()
        self.manifest: dict[str, Any] = _parse_manifest(data)
        self.counts: dict[str, int] = self.manifest["counts"]
        low_x, low_y, high_x, high_y = self.manifest["bbox"]
        self.bbox: tuple[float, float, float, float] = (
            float(low_x),
            float(low_y),
            float(high_x),
            float(high_y),
        )

    def buildings(self) -> PackBuildings:
        return _pack_buildings(self._columns("buildings.bin"))

    def detail(self) -> PackBuildings | None:
        if self.counts["detailBuildings"] == 0:
            return None
        return _pack_buildings(self._columns(DETAIL_FILE))

    def roads(self) -> PackRoads:
        c = self._columns("roads.bin")
        return PackRoads(
            points=c["points"],
            road_class=c["road_class"],
            qx=delta_wrap_decode(c["dx"]),
            qy=delta_wrap_decode(c["dy"]),
        )

    def residents(self) -> PackResidents:
        return PackResidents(**self._columns("residents.bin"))

    def residents_engine(self) -> PackResidentsEngine:
        return PackResidentsEngine(**self._columns("residents_engine.bin"))

    def stores(self) -> PackStores:
        return PackStores(**self._columns("stores.bin"))

    def river(self) -> PackRiver:
        return PackRiver(**self._columns("river.bin"))

    def graph(self) -> PackGraph:
        return PackGraph(**self._columns("graph.bin"))

    def hexes(self) -> NDArray[np.uint64] | None:
        if self.manifest["kind"] == "toy":
            return None
        return self._columns("hexes.bin")["h3"]

    def travel(self) -> NDArray[np.uint16] | None:
        if self.manifest["kind"] == "toy":
            return None
        shape = (self.counts["stores"], self.counts["hexes"], self.counts["bands"])
        return self._columns("travel.bin")["minutes"].reshape(shape)

    def districts(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self._load("districts.json"))
        except ValueError as error:
            raise PackError(f"districts.json is not valid JSON: {error}") from error

    def x_m(self, qx: ArrayLike) -> NDArray[np.float64]:
        return dequantise(qx, self.bbox[0], self.bbox[2])

    def y_m(self, qy: ArrayLike) -> NDArray[np.float64]:
        return dequantise(qy, self.bbox[1], self.bbox[3])

    def verify(self) -> None:
        """Run every file check; a file the pack needs must also be listed in the manifest."""
        files = self.manifest["files"]
        for name in self._required_files():
            if name not in files:
                raise PackError(f"{name} is not listed in the manifest")
        for name in sorted(files):
            self._load(name)

    def _required_files(self) -> list[str]:
        required = [*FIRST_LOAD_FILES, *ENGINE_FILES]
        if self.manifest["kind"] == "toy":
            required = [name for name in required if name not in _TOY_OMITS]
        if self.counts["detailBuildings"] > 0:
            required.append(DETAIL_FILE)
        return required

    def _load(self, name: str) -> bytes:
        """Decompressed file bytes, once size, hash and padding have all checked out."""
        digest = self.manifest["files"].get(name)
        if digest is None:
            raise PackError(f"{name} is not listed in the manifest")
        try:
            data = (self.directory / name).read_bytes()
        except OSError as error:
            raise PackError(f"{name} is missing: {error}") from error
        layout = _layout(name, self.counts)
        if layout is not None:
            expected = sum(column.padded for column in layout)
            declared = content_size(data)
            if declared != expected:
                raise PackError(f"{name} has wrong length {declared}, counts imply {expected}")
        raw = decompress(data)
        if blake3(raw).hexdigest() != digest:
            raise PackError(f"{name} has wrong hash")
        if layout is not None:
            _check_padding(name, raw, layout)
        return raw

    def _columns(self, name: str) -> dict[str, NDArray[Any]]:
        raw = self._load(name)
        layout = _layout(name, self.counts)
        if layout is None:
            raise PackError(f"{name} is not a column file")
        out: dict[str, NDArray[Any]] = {}
        position = 0
        for column in layout:
            if column.planes:
                width = np.dtype(column.dtype).itemsize
                stored = raw[position : position + column.nbytes]
                out[column.name] = from_planes(stored, column.count, width)
            else:
                out[column.name] = np.frombuffer(
                    raw, dtype=column.dtype, count=column.count, offset=position
                )
            position += column.padded
        return out


def _parse_manifest(data: bytes) -> dict[str, Any]:
    try:
        manifest = json.loads(data)
    except ValueError as error:
        raise PackError(f"manifest.json is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise PackError("manifest.json is not an object")
    if manifest.get("format") != PACK_FORMAT:
        raise PackError(f"format must be {PACK_FORMAT}, got {manifest.get('format')!r}")
    if manifest.get("version") != PACK_VERSION:
        raise PackError(f"version must be {PACK_VERSION}, got {manifest.get('version')!r}")
    for key in _MANIFEST_KEYS:
        if key not in manifest:
            raise PackError(f"manifest.json is missing {key}")
    if manifest["compression"] != COMPRESSION:
        raise PackError(f"compression must be {COMPRESSION}, got {manifest['compression']!r}")
    kind = manifest["kind"]
    if kind not in ("toy", "city"):
        raise PackError(f"kind must be toy or city, got {kind!r}")
    if kind == "toy" and "seed" not in manifest:
        raise PackError("manifest.json is missing seed")
    check_timezone(manifest["timezone"])
    check_weekend_days(manifest["weekend_days"])
    _check_counts(manifest["counts"])
    if not isinstance(manifest["files"], dict):
        raise PackError("manifest.json files is not an object")
    bbox = manifest["bbox"]
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, int | float) for value in bbox)
    ):
        raise PackError(f"manifest.json bbox must have 4 numbers, got {bbox!r}")
    return manifest


def _check_counts(counts: Any) -> None:
    if not isinstance(counts, dict):
        raise PackError("manifest.json counts is not an object")
    for key in _COUNT_KEYS:
        if key not in counts:
            raise PackError(f"manifest.json counts is missing {key}")
        if not isinstance(counts[key], int) or counts[key] < 0:
            raise PackError(f"manifest.json counts {key} is not a count: {counts[key]!r}")
    if counts["buildingTileLevel"] > MAX_TILE_LEVEL:
        raise PackError(f"manifest.json buildingTileLevel {counts['buildingTileLevel']} too high")


def _check_padding(name: str, raw: bytes, layout: list[_Column]) -> None:
    position = 0
    for column in layout:
        if any(raw[position + column.nbytes : position + column.padded]):
            raise PackError(f"{name} has non-zero padding after {column.name}")
        position += column.padded


def _pack_buildings(c: dict[str, NDArray[Any]]) -> PackBuildings:
    code = np.cumsum(c["position_delta"], dtype=np.uint64)
    if code.size and code[-1] > _U32_MAX:
        raise PackError("position_delta runs past the largest Morton code")
    qx, qy = morton_decode(code)
    return PackBuildings(
        tile_start=c.get("tile_start"),
        code=code.astype("<u4"),
        qx=qx,
        qy=qy,
        width=c["width"],
        depth=c["depth"],
        height=c["height"],
        rotation=c["rotation"],
        district=c["district"],
    )
