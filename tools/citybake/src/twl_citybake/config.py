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
class WorldPopSource(PinnedDownload):
    pass


@dataclass(frozen=True)
class RiverConfig:
    """Names of the `waterway=river` centrelines that split the city into banks."""

    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DistrictConfig:
    """`place` values whose named nodes and areas seed districts."""

    place_kinds: tuple[str, ...] = ("suburb", "neighbourhood", "quarter", "village", "town")


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
        if not 0 < self.size_shrink <= 1 or self.mean_size * self.size_shrink < self.sizes[0]:
            raise ValueError(
                "size_shrink must be in (0, 1] and keep the mean above the smallest size"
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

        lo, hi = -50.0, 0.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean(mid) < goal else (lo, mid)
        return tuple(weights((lo + hi) / 2))


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
    worldpop: WorldPopSource | None = None
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
        districts=DistrictConfig(
            place_kinds=tuple(
                raw.get("districts", {}).get("place_kinds", DistrictConfig.place_kinds)
            )
        ),
        seed=int(raw.get("seed", 0)),
        worldpop=_worldpop(raw.get("worldpop")),
        population=PopulationConfig(**raw.get("population", {})),
        households=_households(raw.get("households")),
        footprints=_footprints(raw.get("footprints")),
    )


def _worldpop(raw: dict[str, Any] | None) -> WorldPopSource | None:
    return None if raw is None else WorldPopSource(url=raw["url"], md5=raw["md5"])


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
