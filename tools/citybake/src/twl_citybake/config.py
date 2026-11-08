import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Any

from twl_citybake.osm.tags import BuildingKind, RoadClass
from twl_citybake.population.tier import check_shares
from twl_pack.format import MAX_TILE_LEVEL
from twl_pack.local_time import check_timezone, check_weekend_days

MINUTES_PER_DAY = 1440


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
    def cache_key(self) -> str:
        """Stable text for equal bboxes, whether the TOML wrote integers or floats."""
        return repr(tuple(float(v) for v in (self.west, self.south, self.east, self.north)))

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


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_band(bounds: object) -> bool:
    return (
        isinstance(bounds, tuple | list)
        and len(bounds) == 2
        and all(_is_number(b) for b in bounds)
        and bounds[0] <= bounds[1]
    )


def _is_polygon_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"(relation|way)/[0-9]+", value) is not None


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class PopulationCorrection:
    """D-030: each raster cell becomes raster^(1-w) * fit^w, where fit is people per housing floor
    area by built-form class, fitted on `reference_polygons` and applied everywhere. `weight` is a
    stated value with its evidence; no code chooses it. Buildings inside OSM areas with an
    `exclude_landuse` or `exclude_amenity` value house nobody. Without this section a city is
    uncorrected.
    """

    weight: float
    weight_range: tuple[float, float]
    weight_evidence: str
    reference_polygons: tuple[str, ...]
    height_edges: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0)
    footprint_edges: tuple[float, ...] = (100.0, 250.0, 600.0, 1500.0)
    coverage_edges: tuple[float, ...] = (0.10, 0.25, 0.45)
    fit_min_m2_per_person: float = 8.0
    min_class_share: float = 0.002
    exclude_landuse: tuple[str, ...] = ()
    exclude_amenity: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bounds = self.weight_range
        if not _is_band(bounds) or bounds[0] < 0 or bounds[1] > 1:
            raise ValueError(
                f"population_correction.weight_range must be an ordered [low, high] inside "
                f"[0, 1], got {bounds!r}"
            )
        if not _is_number(self.weight) or not bounds[0] <= self.weight <= bounds[1]:
            raise ValueError(
                f"population_correction.weight must lie within weight_range {bounds!r}, "
                f"got {self.weight!r}"
            )
        if not _is_text(self.weight_evidence):
            raise ValueError("population_correction.weight_evidence must say why this weight")
        if not self.reference_polygons:
            raise ValueError("population_correction.reference_polygons needs at least one polygon")
        for polygon in self.reference_polygons:
            if not _is_polygon_id(polygon):
                raise ValueError(
                    "population_correction.reference_polygons entries must be relation/<id> or "
                    f"way/<id>, got {polygon!r}"
                )
        for name in ("height_edges", "footprint_edges", "coverage_edges"):
            edges = getattr(self, name)
            if not _is_ascending(edges):
                raise ValueError(f"population_correction.{name} must be strictly ascending")
        if not _is_positive(self.fit_min_m2_per_person):
            raise ValueError("population_correction.fit_min_m2_per_person must be positive")
        if not _is_number(self.min_class_share) or not 0 <= self.min_class_share < 1:
            raise ValueError("population_correction.min_class_share must be in [0, 1)")
        for name in ("exclude_landuse", "exclude_amenity"):
            values = getattr(self, name)
            if values and not _is_name_list(values):
                raise ValueError(f"population_correction.{name} must be a list of tag values")


@dataclass(frozen=True)
class WardGroup:
    """Named wards checked together. Without `mean_ratio` the group is reported, not asserted;
    `ward_ratio` flags single wards, and a ward in `known_misses` is expected to fall outside it."""

    name: str
    wards: tuple[str, ...]
    mean_ratio: tuple[float, float] | None = None
    ward_ratio: tuple[float, float] | None = None
    known_misses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_text(self.name):
            raise ValueError(f"population_check.ward_groups name {self.name!r} is empty")
        if not _is_name_list(self.wards):
            raise ValueError(f"population_check.ward_groups {self.name}: wards must be a list")
        if len(set(self.wards)) != len(self.wards):
            raise ValueError(f"population_check.ward_groups {self.name}: a ward is repeated")
        for key in ("mean_ratio", "ward_ratio"):
            band = getattr(self, key)
            if band is not None and not _is_band(band):
                raise ValueError(
                    f"population_check.ward_groups {self.name}: {key} must be an ordered "
                    f"[low, high], got {band!r}"
                )
        if strays := [ward for ward in self.known_misses if ward not in self.wards]:
            raise ValueError(
                f"population_check.ward_groups {self.name}: known_misses {strays[0]!r} "
                "is not one of its wards"
            )


@dataclass(frozen=True)
class CheckRegion:
    """A band of people over an OSM boundary (`polygon`) or a box of local metres (`box`, as x0, y0,
    x1, y1); exactly one of the two."""

    name: str
    people: tuple[float, float]
    polygon: str = ""
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not _is_text(self.name):
            raise ValueError(f"population_check.regions name {self.name!r} is empty")
        if not _is_band(self.people):
            raise ValueError(
                f"population_check.regions {self.name}: people must be an ordered [low, high], "
                f"got {self.people!r}"
            )
        if bool(self.polygon) == (self.box is not None):
            raise ValueError(
                f"population_check.regions {self.name} needs exactly one of polygon and box"
            )
        if self.polygon and not _is_polygon_id(self.polygon):
            raise ValueError(
                f"population_check.regions {self.name}: polygon must be relation/<id> or "
                f"way/<id>, got {self.polygon!r}"
            )
        if self.box is not None and not _is_local_box(self.box):
            raise ValueError(
                f"population_check.regions {self.name}: box must be x0, y0, x1, y1 with "
                f"x0 < x1 and y0 < y1, got {self.box!r}"
            )


def _is_ascending(edges: object) -> bool:
    return (
        isinstance(edges, tuple | list)
        and len(edges) > 0
        and all(_is_number(edge) for edge in edges)
        and all(a < b for a, b in pairwise(edges))
    )


def _is_name_list(names: object) -> bool:
    return (
        isinstance(names, tuple | list) and len(names) > 0 and all(_is_text(name) for name in names)
    )


def _is_local_box(box: object) -> bool:
    return (
        isinstance(box, tuple | list)
        and len(box) == 4
        and all(_is_number(v) for v in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


TOTAL_TARGET = "48-ward total"
ALL_TARGETS = "all"


@dataclass(frozen=True)
class PopulationCheck(PinnedDownload):
    """Validation only (D-030): ward polygons the bake is checked against. They never enter a stage
    table, `layer.json`'s attribution or a pack; `report.json` credits `source`."""

    source: str
    name_property: str
    ward_people_2011: int
    total_people_2011: int
    total_growth: tuple[float, float]
    ward_groups: tuple[WardGroup, ...] = ()
    regions: tuple[CheckRegion, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source", "name_property"):
            if not _is_text(getattr(self, name)):
                raise ValueError(f"population_check.{name} must not be empty")
        for name in ("ward_people_2011", "total_people_2011"):
            if not _is_positive(getattr(self, name)):
                raise ValueError(f"population_check.{name} must be positive")
        if not _is_band(self.total_growth):
            raise ValueError(
                f"population_check.total_growth must be an ordered [low, high], "
                f"got {self.total_growth!r}"
            )
        names = [*(g.name for g in self.ward_groups), *(r.name for r in self.regions)]
        if repeats := [name for i, name in enumerate(names) if name in names[:i]]:
            raise ValueError(f"population_check names repeat {repeats[0]!r}")
        if reserved := [name for name in names if name in (TOTAL_TARGET, ALL_TARGETS)]:
            raise ValueError(f"population_check name {reserved[0]!r} is reserved")


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
class SegmentRule:
    """One row of the D-026 table; a persona matches when every set condition holds."""

    name: str
    min_age: int | None = None
    max_age: int | None = None
    sex: str | None = None
    marital: tuple[str, ...] = ()
    education: tuple[str, ...] = ()
    occupied: bool | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a segment rule needs a name")
        if self.sex not in (None, "Female", "Male"):
            raise ValueError(f"segment {self.name} has unknown sex {self.sex!r}")
        if self.min_age is not None and self.max_age is not None and self.min_age > self.max_age:
            raise ValueError(f"segment {self.name} has min_age above max_age")

    @property
    def is_catch_all(self) -> bool:
        return (
            self.min_age is None
            and self.max_age is None
            and self.sex is None
            and not self.marital
            and not self.education
            and self.occupied is None
        )


@dataclass(frozen=True)
class PersonasConfig:
    """The pinned Nemotron slice and the ordered first-match segment table (D-026)."""

    revision: str
    state: str
    district: str
    no_occupation: str
    married: str
    segments_version: int
    segments: tuple[SegmentRule, ...]

    def __post_init__(self) -> None:
        if len(self.segments) < 2:
            raise ValueError("personas.segments needs at least two rules")
        names = [rule.name for rule in self.segments]
        if repeats := [name for i, name in enumerate(names) if name in names[:i]]:
            raise ValueError(f"personas.segments repeats {repeats[0]!r}")
        if not self.segments[-1].is_catch_all:
            raise ValueError("the last personas.segments rule must have no conditions")
        if early := [rule.name for rule in self.segments[:-1] if rule.is_catch_all]:
            raise ValueError(f"personas.segments rule {early[0]!r} shadows the rules after it")


@dataclass(frozen=True)
class GraphTargets:
    """What the graph report is checked against: `clustering` is the average local clustering over
    all ties. Ranges live here, not in code, so another city can state its own."""

    mean_degree: tuple[float, float] = (5.0, 8.0)
    clustering: tuple[float, float] = (0.10, 0.50)
    cross_river_non_household: tuple[float, float] = (0.05, 0.25)
    largest_component_min: float = 0.95
    isolated_max: float = 0.02

    def __post_init__(self) -> None:
        ranges = {
            "mean_degree": self.mean_degree,
            "clustering": self.clustering,
            "cross_river_non_household": self.cross_river_non_household,
        }
        for name, bounds in ranges.items():
            if not _is_ordered_range(bounds):
                raise ValueError(f"graph.targets.{name} must be an ordered [low, high], {bounds}")
        if not (0 <= self.largest_component_min <= 1 and 0 <= self.isolated_max <= 1):
            raise ValueError("graph.targets largest_component_min and isolated_max must be shares")


@dataclass(frozen=True)
class GraphConfig:
    """Tie generation (P-023). Degrees are mean ties per resident; `work_degree` is per occupied
    resident. Workplaces are tagged commercial, industrial and civic buildings, plus untagged
    footprints at least `workplace_min_area_m2` in area and at most `workplace_max_floors` tall."""

    neighbour_degree: float = 1.6
    neighbour_scale_m: float = 250.0
    neighbour_candidates: int = 64
    friend_degree: float = 1.2
    friend_scale_m: float = 3000.0
    friend_homophily: float = 4.0
    friend_candidates: int = 16
    work_degree: float = 1.0
    work_scale_m: float = 5000.0
    work_cell_m: float = 500.0
    workplace_candidates: int = 16
    workplace_min_area_m2: float = 1000.0
    workplace_max_floors: int = 3
    targets: GraphTargets = field(default_factory=GraphTargets)

    def __post_init__(self) -> None:
        for name in ("neighbour_scale_m", "friend_scale_m", "work_scale_m", "work_cell_m"):
            if not getattr(self, name) > 0:
                raise ValueError(f"graph.{name} must be positive")
        for name in ("neighbour_degree", "friend_degree", "work_degree"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"graph.{name} must be non-negative")
        for name in ("neighbour_candidates", "friend_candidates", "workplace_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"graph.{name} must be a whole number of at least 2")
        if not self.friend_homophily > 0:
            raise ValueError("graph.friend_homophily must be positive")
        if not self.workplace_min_area_m2 >= 0:
            raise ValueError("graph.workplace_min_area_m2 must be non-negative")
        if type(self.workplace_max_floors) is not int or self.workplace_max_floors < 1:
            raise ValueError("graph.workplace_max_floors must be a whole number of at least 1")


def _is_positive(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


@dataclass(frozen=True)
class StoresConfig:
    """Greedy max-coverage placement of synthetic dark stores (P-024, D-013). Candidates are road
    nodes, at most one per `candidate_cell_m` grid cell; placing stops once `coverage_target` of
    resident weight lies within `radius_m` of a store."""

    radius_m: float = 2500.0
    coverage_target: float = 0.92
    max_stores: int = 80
    candidate_cell_m: float = 500.0

    def __post_init__(self) -> None:
        for name in ("radius_m", "candidate_cell_m"):
            if not _is_positive(getattr(self, name)):
                raise ValueError(f"stores.{name} must be positive, got {getattr(self, name)!r}")
        if not _is_positive(self.coverage_target) or self.coverage_target > 1:
            raise ValueError(f"stores.coverage_target must be in (0, 1], {self.coverage_target!r}")
        if type(self.max_stores) is not int or self.max_stores < 1:
            raise ValueError(f"stores.max_stores must be a whole number >= 1, {self.max_stores!r}")


@dataclass(frozen=True)
class TravelBand:
    """A time of day from `start_minute` to the next band's start, wrapping midnight. `factor`
    scales the speed of every road class."""

    name: str
    start_minute: int
    factor: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"travel.bands name must be non-empty text, got {self.name!r}")
        if type(self.start_minute) is not int or not 0 <= self.start_minute <= MINUTES_PER_DAY - 1:
            raise ValueError(
                f"travel.bands start_minute of {self.name} must be a whole number in "
                f"0..{MINUTES_PER_DAY - 1}, got {self.start_minute!r}"
            )
        if not _is_positive(self.factor):
            raise ValueError(f"travel.bands factor of {self.name} must be positive")


DEFAULT_SPEED_KMH = {
    "major": 40.0,
    "primary": 30.0,
    "secondary": 26.0,
    "tertiary": 22.0,
    "local": 16.0,
    "service": 12.0,
}
DEFAULT_TRAVEL_BANDS = (
    TravelBand("night", 1320, 1.25),
    TravelBand("morning_peak", 480, 0.8),
    TravelBand("midday", 660, 1.0),
    TravelBand("evening_peak", 1020, 0.7),
)


@dataclass(frozen=True)
class TravelConfig:
    """Store-to-hex travel minutes (D-022). `speed_kmh` is two-wheeler speed by lowercase RoadClass
    name at a band factor of 1; `bands` cover the day, each running to the next start."""

    h3_resolution: int = 9
    speed_kmh: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SPEED_KMH))
    bands: tuple[TravelBand, ...] = DEFAULT_TRAVEL_BANDS

    def __post_init__(self) -> None:
        if type(self.h3_resolution) is not int or not 0 <= self.h3_resolution <= 15:
            raise ValueError(
                f"travel.h3_resolution must be a whole number in 0..15, {self.h3_resolution!r}"
            )
        classes = sorted(road.name.lower() for road in RoadClass)
        if sorted(self.speed_kmh) != classes:
            raise ValueError(f"travel.speed_kmh needs exactly the road classes {classes}")
        for name, speed in self.speed_kmh.items():
            if not _is_positive(speed):
                raise ValueError(f"travel.speed_kmh.{name} must be positive, got {speed!r}")
        if not self.bands:
            raise ValueError("travel.bands needs at least one band")
        starts = [band.start_minute for band in self.bands]
        if repeats := [start for i, start in enumerate(starts) if start in starts[:i]]:
            raise ValueError(f"travel.bands repeat start_minute {repeats[0]}")
        names = [band.name for band in self.bands]
        if repeats := [name for i, name in enumerate(names) if name in names[:i]]:
            raise ValueError(f"travel.bands repeat name {repeats[0]!r}")

    def bands_by_start(self) -> tuple[TravelBand, ...]:
        return tuple(sorted(self.bands, key=lambda band: band.start_minute))


@dataclass(frozen=True)
class PackConfig:
    """Pack-time rules (D-031). A footprint under `min_footprint_m2` with no other building within
    `isolated_m` goes to the detail file. `arterial_classes` are lowercase RoadClass names."""

    min_footprint_m2: float = 30.0
    isolated_m: float = 3.0
    tile_level: int = 6
    arterial_classes: tuple[str, ...] = ("major", "primary", "secondary")

    def __post_init__(self) -> None:
        for name in ("min_footprint_m2", "isolated_m"):
            if not _is_positive(getattr(self, name)):
                raise ValueError(f"pack.{name} must be positive, got {getattr(self, name)!r}")
        if type(self.tile_level) is not int or not 0 <= self.tile_level <= MAX_TILE_LEVEL:
            raise ValueError(
                f"pack.tile_level must be a whole number in 0..{MAX_TILE_LEVEL}, "
                f"got {self.tile_level!r}"
            )
        known = {road.name.lower() for road in RoadClass}
        if unknown := [name for name in self.arterial_classes if name not in known]:
            raise ValueError(f"pack.arterial_classes has unknown road class {unknown[0]!r}")


@dataclass(frozen=True)
class TierConfig:
    """D-035: housed buildings are ranked by floor area per person and cut so that `low_share` of
    households are low tier and `high_share` high; each share states its source."""

    low_share: float
    high_share: float
    low_share_source: str
    high_share_source: str
    attribution: str

    def __post_init__(self) -> None:
        check_shares(self.low_share, self.high_share)
        for name in ("low_share_source", "high_share_source", "attribution"):
            if not _is_text(getattr(self, name)):
                raise ValueError(f"tier.{name} must not be blank")


@dataclass(frozen=True)
class CityConfig:
    """`timezone` (IANA) and `weekend_days` (ISO weekdays) go into the pack manifest (D-036); a city
    without them cannot be packed."""

    slug: str
    name: str
    bbox: BBox
    osm: OsmSource
    river: RiverConfig = field(default_factory=RiverConfig)
    districts: DistrictConfig = field(default_factory=DistrictConfig)
    seed: int = 0
    timezone: str | None = None
    weekend_days: tuple[int, ...] | None = None
    population_raster: PopulationRaster | None = None
    population_total: PopulationTotal | None = None
    population_correction: PopulationCorrection | None = None
    population_check: PopulationCheck | None = None
    population: PopulationConfig = field(default_factory=PopulationConfig)
    households: HouseholdConfig | None = None
    tier: TierConfig | None = None
    footprints: FootprintsConfig | None = None
    personas: PersonasConfig | None = None
    graph: GraphConfig = field(default_factory=GraphConfig)
    stores: StoresConfig = field(default_factory=StoresConfig)
    travel: TravelConfig = field(default_factory=TravelConfig)
    pack: PackConfig = field(default_factory=PackConfig)


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
        timezone=None if "timezone" not in raw else check_timezone(raw["timezone"]),
        weekend_days=None if "weekend_days" not in raw else check_weekend_days(raw["weekend_days"]),
        population_raster=_population_raster(raw.get("population_raster")),
        population_total=_population_total(raw.get("population_total")),
        population_correction=_population_correction(raw.get("population_correction")),
        population_check=_population_check(raw.get("population_check")),
        population=PopulationConfig(**raw.get("population", {})),
        households=_households(raw.get("households")),
        tier=_tier(raw.get("tier")),
        footprints=_footprints(raw.get("footprints")),
        personas=_personas(raw.get("personas")),
        graph=_graph(raw.get("graph", {})),
        stores=_stores(raw.get("stores", {})),
        travel=_travel(raw.get("travel", {})),
        pack=_pack(raw.get("pack", {})),
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


def _tuples(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}


def _population_correction(raw: dict[str, Any] | None) -> PopulationCorrection | None:
    if raw is None:
        return None
    _known_keys("[population_correction]", raw, {f.name for f in fields(PopulationCorrection)})
    required = {"weight", "weight_range", "weight_evidence", "reference_polygons"}
    if missing := sorted(required - set(raw)):
        raise ValueError(f"[population_correction] needs {missing[0]!r}")
    return PopulationCorrection(**_tuples(raw))


def _population_check(raw: dict[str, Any] | None) -> PopulationCheck | None:
    if raw is None:
        return None
    _known_keys("[population_check]", raw, {f.name for f in fields(PopulationCheck)})
    required = {
        "url",
        "md5",
        "source",
        "name_property",
        "ward_people_2011",
        "total_people_2011",
        "total_growth",
    }
    if missing := sorted(required - set(raw)):
        raise ValueError(f"[population_check] needs {missing[0]!r}")
    return PopulationCheck(
        **{
            **raw,
            "total_growth": tuple(raw["total_growth"]),
            "ward_groups": tuple(_ward_group(group) for group in raw.get("ward_groups", ())),
            "regions": tuple(_check_region(region) for region in raw.get("regions", ())),
        }
    )


def _ward_group(raw: dict[str, Any]) -> WardGroup:
    _known_keys("[[population_check.ward_groups]]", raw, {f.name for f in fields(WardGroup)})
    if missing := sorted({"name", "wards"} - set(raw)):
        raise ValueError(f"[[population_check.ward_groups]] {raw!r} needs {missing[0]!r}")
    return WardGroup(**_tuples(raw))


def _check_region(raw: dict[str, Any]) -> CheckRegion:
    _known_keys("[[population_check.regions]]", raw, {f.name for f in fields(CheckRegion)})
    if missing := sorted({"name", "people"} - set(raw)):
        raise ValueError(f"[[population_check.regions]] {raw!r} needs {missing[0]!r}")
    return CheckRegion(**_tuples(raw))


def _households(raw: dict[str, Any] | None) -> HouseholdConfig | None:
    if raw is None:
        return None
    return HouseholdConfig(
        source=raw["source"],
        sizes=tuple(raw["sizes"]),
        counts=tuple(raw["counts"]),
        size_shrink=float(raw.get("size_shrink", 1.0)),
    )


def _tier(raw: Any) -> TierConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("[tier] must be a table")
    required = {f.name for f in fields(TierConfig)}
    _known_keys("[tier]", raw, required)
    if missing := sorted(required - set(raw)):
        raise ValueError(f"[tier] needs {missing[0]!r}")
    return TierConfig(**raw)


def _footprints(raw: dict[str, Any] | None) -> FootprintsConfig | None:
    return None if raw is None else FootprintsConfig(**raw)


def _personas(raw: dict[str, Any] | None) -> PersonasConfig | None:
    if raw is None:
        return None
    required = {
        "revision",
        "state",
        "district",
        "no_occupation",
        "married",
        "segments_version",
        "segments",
    }
    _known_keys("[personas]", raw, required)
    if missing := sorted(required - set(raw)):
        raise ValueError(f"[personas] needs {missing[0]!r}")
    return PersonasConfig(**{**raw, "segments": tuple(_segment_rule(r) for r in raw["segments"])})


def _segment_rule(raw: dict[str, Any]) -> SegmentRule:
    known = {"name", "min_age", "max_age", "sex", "marital", "education", "occupied"}
    _known_keys("[[personas.segments]]", raw, known)
    if "name" not in raw:
        raise ValueError(f"[[personas.segments]] entry {raw!r} needs a name")
    for key in ("marital", "education"):
        if not isinstance(raw.get(key, []), list):
            raise ValueError(f"segment {raw['name']}: {key} must be a list of names")
    if not isinstance(raw.get("occupied", False), bool):
        raise ValueError(f"segment {raw['name']}: occupied must be true or false")
    return SegmentRule(
        **{
            **raw,
            "marital": tuple(raw.get("marital", ())),
            "education": tuple(raw.get("education", ())),
        }
    )


def _is_ordered_range(bounds: object) -> bool:
    return (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and all(isinstance(b, int | float) for b in bounds)
        and bounds[0] <= bounds[1]
    )


def _graph(raw: dict[str, Any]) -> GraphConfig:
    _known_keys("[graph]", raw, {f.name for f in fields(GraphConfig)})
    targets = raw.get("targets", {})
    _known_keys("[graph.targets]", targets, {f.name for f in fields(GraphTargets)})
    bounds: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in targets.items()}
    return GraphConfig(**{**raw, "targets": GraphTargets(**bounds)})


def _stores(raw: dict[str, Any]) -> StoresConfig:
    _known_keys("[stores]", raw, {f.name for f in fields(StoresConfig)})
    return StoresConfig(**raw)


def _pack(raw: dict[str, Any]) -> PackConfig:
    _known_keys("[pack]", raw, {f.name for f in fields(PackConfig)})
    return PackConfig(**_tuples(raw))


def _travel(raw: dict[str, Any]) -> TravelConfig:
    _known_keys("[travel]", raw, {f.name for f in fields(TravelConfig)})
    if not isinstance(raw.get("speed_kmh", {}), dict):
        raise ValueError("travel.speed_kmh must be a table of road classes")
    if "bands" not in raw:
        return TravelConfig(**raw)
    return TravelConfig(**{**raw, "bands": tuple(_travel_band(band) for band in raw["bands"])})


def _travel_band(raw: Any) -> TravelBand:
    if not isinstance(raw, dict):
        raise ValueError(f"travel.bands entry {raw!r} must be a table")
    required = {f.name for f in fields(TravelBand)}
    _known_keys("[[travel.bands]]", raw, required)
    if missing := sorted(required - set(raw)):
        raise ValueError(f"[[travel.bands]] entry {raw!r} needs {missing[0]!r}")
    return TravelBand(**raw)


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

    def pack(self, city: str) -> Path:
        return self.root / "cache" / "packs" / city
