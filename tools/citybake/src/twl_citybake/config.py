import os
import re
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path


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
class OsmSource:
    url: str
    md5: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class RiverConfig:
    """Names of the `waterway=river` centrelines that split the city into banks."""

    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DistrictConfig:
    """`place` values whose named nodes and areas seed districts."""

    place_kinds: tuple[str, ...] = ("suburb", "neighbourhood", "quarter", "village", "town")


@dataclass(frozen=True)
class CityConfig:
    slug: str
    name: str
    bbox: BBox
    osm: OsmSource
    river: RiverConfig = field(default_factory=RiverConfig)
    districts: DistrictConfig = field(default_factory=DistrictConfig)


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
        river=RiverConfig(names=tuple(raw["river"]["names"])),
        districts=DistrictConfig(place_kinds=tuple(raw["districts"]["place_kinds"])),
    )


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
