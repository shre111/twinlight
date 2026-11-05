import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Any

from twl_citybake.osm.tags import BuildingKind


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180 <= self.west < self.east <= 180 and -90 <= self.south < self.north <= 90):
            raise ValueError(f"invalid bbox {self}")

    @property
    def centre(self) -> tuple[float, float]:
        return (self.west + self.east) / 2, (self.south + self.north) / 2

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north


@dataclass(frozen=True)
class PinnedDownload:
    url: str
    md5: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class OsmSource(PinnedDownload):
    pass


@dataclass(frozen=True)
class PopulationRaster(PinnedDownload):
    """A gridded population product in EPSG:4326, persons per cell. When the pinned download is a
    zip, `member` names the GeoTIFF inside it."""

    source: str
    licence: str
    member: str = ""

    def __post_init__(self) -> None:
        if "/" in self.member or "\\" in self.member or self.member.startswith(".."):
            raise ValueError(f"population_raster.member must be a plain name, got {self.member!r}")

    @property
    def raster_name(self) -> str:
        return self.member or self.filename


@dataclass(frozen=True)
class PopulationTotal:
    """A sourced head count that overrides the raster's total inside an OSM boundary, so a census
    figure can replace a projection. `polygon` is `relation/<id>` or `way/<id>`."""

    people: int
    source: str
    polygon: str
    as_of: str

    def __post_init__(self) -> None:
        if self.people <= 0:
            raise ValueError(f"population_total.people must be positive, got {self.people}")
        if not re.fullmatch(r"(relation|way)/[0-9]+", self.polygon):
            raise ValueError(
                f"population_total.polygon must be relation/<id> or way/<id>, got {self.polygon!r}"
            )
        if not self.source or not self.as_of:
            raise ValueError("population_total needs a source and an as_of")


@dataclass(frozen=True)
class RiverConfig:
    """Names of the `waterway=river` centrelines that split the city into banks."""

    names: tuple[str, ...] = ()


def normalise_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


@dataclass(frozen=True)
class ExtraSeed:
    """A district seed named `name` at OSM node `node`, for a district OSM has no place for."""

    name: str
    node: int


@dataclass(frozen=True)
class DistrictConfig:
    """`place` values whose named nodes and areas seed districts, then the seed table:
    `extra_seeds` add seeds, `merge` drops a district into another, `rename` changes display
    names. Names match by normalise_name."""

    place_kinds: tuple[str, ...] = ("suburb", "neighbourhood", "quarter", "village", "town")
    extra_seeds: tuple[ExtraSeed, ...] = ()
    merge: Mapping[str, str] = field(default_factory=dict)
    rename: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_names("districts.extra_seeds name", [s.name for s in self.extra_seeds], True)
        nodes = [s.node for s in self.extra_seeds]
        if bad := [n for n in nodes if not isinstance(n, int) or isinstance(n, bool)]:
            raise ValueError(f"districts.extra_seeds node {bad[0]!r} is not an OSM node id")
        if repeats := [n for i, n in enumerate(nodes) if n in nodes[:i]]:
            raise ValueError(f"districts.extra_seeds repeat node {repeats[0]}")
        _check_names("districts.merge key", list(self.merge), True)
        _check_names("districts.merge target", list(self.merge.values()))
        _check_names("districts.rename key", list(self.rename), True)
        _check_names("districts.rename value", list(self.rename.values()))
        merged = {normalise_name(k) for k in self.merge}
        if chained := sorted(t for t in self.merge.values() if normalise_name(t) in merged):
            raise ValueError(f"districts.merge target {chained[0]!r} is itself merged away")


def _check_names(what: str, names: list[Any], unique: bool = False) -> None:
    seen: dict[str, str] = {}
    for name in names:
        if not isinstance(name, str) or not (key := normalise_name(name)):
            raise ValueError(f"{what} {name!r} has no letters or digits")
        if unique and key in seen:
            raise ValueError(f"{what} {name!r} repeats {seen[key]!r}")
        seen.setdefault(key, name)


DEFAULT_RESIDENTIAL_SHARE = {
    "unknown": 0.9,
    "residential": 1.0,
    "commercial": 0.2,
    "industrial": 0.05,
    "civic": 0.1,
    "religious": 0.0,
    "other": 0.0,
}


@dataclass(frozen=True)
class PopulationConfig:
    """`residential_share` is the share of floor area that is housing, keyed by lowercase
    BuildingKind name."""

    min_floor_m2_per_person: float = 4.0
    adult_share: float = 0.69
    adult_share_source: str = ""
    residents: int = 100_000
    shopper_share: float = 0.27
    residential_share: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RESIDENTIAL_SHARE)
    )

    def __post_init__(self) -> None:
        names = sorted(kind.name.lower() for kind in BuildingKind)
        if sorted(self.residential_share) != names:
            raise ValueError(f"residential_share needs exactly the kinds {names}")
        if not all(0 <= v <= 1 for v in self.residential_share.values()):
            raise ValueError("residential shares must be in [0, 1]")
        if not (0 < self.adult_share <= 1 and 0 < self.shopper_share <= 1):
            raise ValueError("adult_share and shopper_share must be in (0, 1]")
        if self.min_floor_m2_per_person <= 0 or self.residents < 0:
            raise ValueError("min_floor_m2_per_person must be positive, residents non-negative")

    def share_by_kind(self) -> dict[BuildingKind, float]:
        return {kind: float(self.residential_share[kind.name.lower()]) for kind in BuildingKind}


@dataclass(frozen=True)
class HouseholdConfig:
    """Census households by size: `sizes` are the representative sizes of the bins.
    `size_shrink` scales the mean household size for a census older than the population it is
    applied to (1.0 keeps the census)."""

    source: str
    sizes: tuple[int, ...]
    counts: tuple[int, ...]
    size_shrink: float = 1.0

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.counts) or not self.sizes:
            raise ValueError("household sizes and counts must be non-empty and the same length")
        if self.sizes[0] < 1 or any(a >= b for a, b in pairwise(self.sizes)):
            raise ValueError("household sizes must be positive and increasing")
        if self.sizes[-1] > 255 or min(self.counts) < 0 or sum(self.counts) == 0:
            raise ValueError("household sizes must fit u8, counts be non-negative and not all 0")
        smallest = min(s for s, n in zip(self.sizes, self.counts, strict=True) if n > 0)
        if not 0 < self.size_shrink <= 1 or self.mean_size * self.size_shrink < smallest:
            raise ValueError(
                "size_shrink must be in (0, 1] and keep the mean above the smallest counted size"
            )

    @property
    def mean_size(self) -> float:
        persons = sum(s * n for s, n in zip(self.sizes, self.counts, strict=True))
        return persons / sum(self.counts)

    def shrunk_counts(self) -> tuple[float, ...]:
        """Counts exponentially tilted towards small households, w_k = n_k exp(theta s_k), with
        theta set by bisection so the mean size is `size_shrink` times the census mean; the
        least-informative reweighting that hits the new mean."""
        if self.size_shrink == 1.0:
            return tuple(float(n) for n in self.counts)
        sizes = [float(s) for s in self.sizes]
        goal = self.mean_size * self.size_shrink

        def weights(theta: float) -> list[float]:
            top = max(sizes)
            return [
                n * math.exp(theta * (s - top)) for s, n in zip(sizes, self.counts, strict=True)
            ]

        def mean(theta: float) -> float:
            w = weights(theta)
            return sum(wk * s for wk, s in zip(w, sizes, strict=True)) / sum(w)

        lo, hi = -700.0 / max(1.0, max(sizes) - min(sizes)), 0.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean(mid) < goal else (lo, mid)
        theta = (lo + hi) / 2
        if not abs(mean(theta) - goal) <= 1e-9 * goal:
            raise ValueError(f"cannot shrink the mean household size to {goal:.3f}")
        return tuple(weights(theta))


@dataclass(frozen=True)
class FootprintsConfig:
    """D-021 gap filling: the pinned VIDA file (`vida_etag` is the HTTP ETag, quotes included)
    and the Open Buildings 2.5D heights manifest."""

    vida_url: str
    vida_etag: str
    heights_manifest: str
    vida_snapshot: str = ""
    overlap_drop: float = 0.3
    raster_min_height_m: float = 1.0
    raster_cell_m: float = 4.0

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_drop <= 1:
            raise ValueError("overlap_drop must be in [0, 1]")
        if self.raster_min_height_m < 0 or self.raster_cell_m <= 0:
            raise ValueError("raster_min_height_m must be non-negative, raster_cell_m positive")


@dataclass(frozen=True)
class CityConfig:
    slug: str
    name: str
    bbox: BBox
    osm: OsmSource
    river: RiverConfig = field(default_factory=RiverConfig)
    districts: DistrictConfig = field(default_factory=DistrictConfig)
    seed: int = 0
    population_raster: PopulationRaster | None = None
    population_total: PopulationTotal | None = None
    population: PopulationConfig = field(default_factory=PopulationConfig)
    households: HouseholdConfig | None = None
    footprints: FootprintsConfig | None = None


def load_city(slug: str) -> CityConfig:
    resource = files("twl_citybake.cities") / f"{slug}.toml"
    if not re.fullmatch(r"[a-z0-9-]+", slug) or not resource.is_file():
        raise ValueError(f"unknown city {slug!r}")
    raw = tomllib.loads(resource.read_text())
    return CityConfig(
        slug=raw["slug"],
        name=raw["name"],
        bbox=BBox(*raw["bbox"]),
        osm=OsmSource(url=raw["osm"]["url"], md5=raw["osm"]["md5"]),
        river=RiverConfig(names=tuple(raw.get("river", {}).get("names", ()))),
        districts=_districts(raw.get("districts", {})),
        seed=int(raw.get("seed", 0)),
        population_raster=_population_raster(raw.get("population_raster")),
        population_total=_population_total(raw.get("population_total")),
        population=PopulationConfig(**raw.get("population", {})),
        households=_households(raw.get("households")),
        footprints=_footprints(raw.get("footprints")),
    )


def _districts(raw: dict[str, Any]) -> DistrictConfig:
    _known_keys("[districts]", raw, {"place_kinds", "extra_seeds", "merge", "rename"})
    for table in ("merge", "rename"):
        if not isinstance(raw.get(table, {}), dict):
            raise ValueError(f"districts.{table} must be a table of names")
    return DistrictConfig(
        place_kinds=tuple(raw.get("place_kinds", DistrictConfig.place_kinds)),
        extra_seeds=tuple(_extra_seed(entry) for entry in raw.get("extra_seeds", [])),
        merge=dict(raw.get("merge", {})),
        rename=dict(raw.get("rename", {})),
    )


def _extra_seed(raw: Any) -> ExtraSeed:
    if not isinstance(raw, dict):
        raise ValueError(f"districts.extra_seeds entry {raw!r} must be a table")
    _known_keys("districts.extra_seeds entry", raw, {"name", "node"})
    if "name" not in raw or "node" not in raw:
        raise ValueError(f"districts.extra_seeds entry {raw!r} needs a name and a node")
    return ExtraSeed(name=raw["name"], node=raw["node"])


def _known_keys(where: str, raw: dict[str, Any], known: set[str]) -> None:
    if unknown := sorted(set(raw) - known):
        raise ValueError(f"{where} has unknown key {unknown[0]!r}")


def _population_raster(raw: dict[str, Any] | None) -> PopulationRaster | None:
    if raw is None:
        return None
    _known_keys("[population_raster]", raw, {"url", "md5", "member", "source", "licence"})
    if missing := sorted({"url", "md5", "source", "licence"} - set(raw)):
        raise ValueError(f"[population_raster] needs {missing[0]!r}")
    return PopulationRaster(**raw)


def _population_total(raw: dict[str, Any] | None) -> PopulationTotal | None:
    if raw is None:
        return None
    _known_keys("[population_total]", raw, {"people", "source", "polygon", "as_of"})
    if missing := sorted({"people", "source", "polygon", "as_of"} - set(raw)):
        raise ValueError(f"[population_total] needs {missing[0]!r}")
    return PopulationTotal(**raw)


def _households(raw: dict[str, Any] | None) -> HouseholdConfig | None:
    if raw is None:
        return None
    return HouseholdConfig(
        source=raw["source"],
        sizes=tuple(raw["sizes"]),
        counts=tuple(raw["counts"]),
        size_shrink=float(raw.get("size_shrink", 1.0)),
    )


def _footprints(raw: dict[str, Any] | None) -> FootprintsConfig | None:
    return None if raw is None else FootprintsConfig(**raw)


@dataclass(frozen=True)
class DataDirs:
    root: Path

    @classmethod
    def from_env(cls, override: Path | None = None) -> "DataDirs":
        return cls(override or Path(os.environ.get("TWL_DATA_DIR", "data")))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    def stage(self, city: str, stage: str) -> Path:
        return self.root / "cache" / "citybake" / city / stage
