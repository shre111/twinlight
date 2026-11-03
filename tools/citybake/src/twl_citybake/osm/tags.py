import re
from collections.abc import Mapping
from enum import IntEnum

LEVEL_HEIGHT_M = 3.2
MAX_HEIGHT_M = 400.0
MAX_LEVELS = 80.0
FOOT_M = 0.3048
INCH_M = 0.0254

_NUMBER = r"(\d+(?:[.,]\d+)?|[.,]\d+)"
_LENGTH = re.compile(rf"{_NUMBER}\s*(m|metres?|meters?|ft|feet|foot|')?", re.IGNORECASE)
_FEET_INCHES = re.compile(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*(?:\"|'')")
_LEVELS = re.compile(r"\d+(?:\.\d+)?")


class HeightSource(IntEnum):
    HEIGHT = 0
    LEVELS = 1
    DEFAULT = 2


class BuildingKind(IntEnum):
    UNKNOWN = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    CIVIC = 4
    RELIGIOUS = 5
    OTHER = 6


class RoadClass(IntEnum):
    MAJOR = 0
    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3
    LOCAL = 4
    SERVICE = 5


NOT_BUILDINGS = frozenset({"no", "proposed", "demolished", "destroyed", "razed"})

_KINDS: dict[str, BuildingKind] = {
    value: kind
    for kind, values in {
        BuildingKind.RESIDENTIAL: (
            "apartments residential house detached semidetached_house terrace bungalow "
            "dormitory hut farm cabin"
        ),
        BuildingKind.COMMERCIAL: "commercial retail office supermarket kiosk hotel",
        BuildingKind.INDUSTRIAL: "industrial warehouse factory manufacture",
        BuildingKind.CIVIC: (
            "school university college kindergarten hospital public government civic "
            "train_station transportation fire_station"
        ),
        BuildingKind.RELIGIOUS: (
            "religious temple mosque church chapel shrine cathedral synagogue gurudwara"
        ),
        BuildingKind.OTHER: (
            "roof garage garages shed toilets water_tower service carport construction ruins "
            "greenhouse stadium"
        ),
    }.items()
    for value in values.split()
}

_ROAD_CLASSES: dict[str, RoadClass] = {
    value: road
    for road, values in {
        RoadClass.MAJOR: "motorway motorway_link trunk trunk_link",
        RoadClass.PRIMARY: "primary primary_link",
        RoadClass.SECONDARY: "secondary secondary_link",
        RoadClass.TERTIARY: "tertiary tertiary_link",
        RoadClass.LOCAL: "residential unclassified living_street road",
        RoadClass.SERVICE: "service track",
    }.items()
    for value in values.split()
}


def _length_m(item: str) -> float | None:
    if match := _FEET_INCHES.fullmatch(item):
        return int(match[1]) * FOOT_M + float(match[2]) * INCH_M
    if not (match := _LENGTH.fullmatch(item)):
        return None
    value = float(match[1].replace(",", "."))
    unit = (match[2] or "m").lower()
    return value * FOOT_M if unit in ("ft", "feet", "foot", "'") else value


def _levels(item: str) -> float | None:
    return float(item) if _LEVELS.fullmatch(item) else None


def parse_length_m(value: str | None) -> float | None:
    if value is None:
        return None
    items = (_length_m(item.strip()) for item in value.split(";"))
    return max((v for v in items if v is not None and 0 < v <= MAX_HEIGHT_M), default=None)


def parse_levels(value: str | None) -> float | None:
    if value is None:
        return None
    items = (_levels(item.strip()) for item in re.split(r"[;,]", value))
    return max((v for v in items if v is not None and 1 <= v <= MAX_LEVELS), default=None)


def tagged_height(tags: Mapping[str, str]) -> tuple[float, HeightSource] | None:
    for key in ("height", "building:height"):
        if (height := parse_length_m(tags.get(key))) is not None:
            return height, HeightSource.HEIGHT
    if (levels := parse_levels(tags.get("building:levels"))) is not None:
        return levels * LEVEL_HEIGHT_M, HeightSource.LEVELS
    return None


def building_kind(value: str) -> BuildingKind | None:
    key = value.strip().lower()
    if key in NOT_BUILDINGS:
        return None
    return _KINDS.get(key, BuildingKind.UNKNOWN)


def road_class(highway: str) -> RoadClass | None:
    return _ROAD_CLASSES.get(highway.strip().lower())


def oneway(tags: Mapping[str, str]) -> int:
    match tags.get("oneway", "").strip().lower():
        case "yes" | "true" | "1":
            return 1
        case "-1" | "reverse":
            return -1
        case "no" | "false" | "0":
            return 0
    if tags.get("junction") in ("roundabout", "circular"):
        return 1
    return 1 if tags.get("highway") in ("motorway", "motorway_link") else 0
