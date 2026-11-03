import pytest

from twl_citybake.osm.tags import (
    LEVEL_HEIGHT_M,
    BuildingKind,
    HeightSource,
    RoadClass,
    building_kind,
    oneway,
    parse_length_m,
    parse_levels,
    road_class,
    tagged_height,
)

FOOT_M = 0.3048


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12", 12.0),
        ("12.5", 12.5),
        ("12 m", 12.0),
        ("12m", 12.0),
        ("12.5m", 12.5),
        ("12,5", 12.5),
        ("  12 m ", 12.0),
        ("40 ft", 40 * FOOT_M),
        ("40'", 40 * FOOT_M),
        ("40 feet", 40 * FOOT_M),
        ("12;15", 15.0),
        ("15;12", 15.0),
        ("400", 400.0),
    ],
)
def test_parse_length_accepts_osm_notations(value: str, expected: float) -> None:
    assert parse_length_m(value) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "value",
    [None, "", "unknow", "0", "0.0", "-3", "400.1", "1000", "nan", "inf", "-inf", "m"],
)
def test_parse_length_rejects_missing_unparseable_and_out_of_range(value: str | None) -> None:
    assert parse_length_m(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("4", 4.0),
        ("1", 1.0),
        ("4.5", 4.5),
        ("80", 80.0),
        ("1,2,3", 3.0),
        ("1;2;5", 5.0),
        ("5;2", 5.0),
        ("4,5", 5.0),
    ],
)
def test_parse_levels_accepts_numbers_and_lists(value: str, expected: float) -> None:
    assert parse_levels(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value", [None, "", "x", "0", "0.5", "-2", "81", "500", "nan", "inf", "two"]
)
def test_parse_levels_rejects_missing_unparseable_and_out_of_range(value: str | None) -> None:
    assert parse_levels(value) is None


def test_level_height_is_three_point_two_metres() -> None:
    assert LEVEL_HEIGHT_M == 3.2


def test_height_source_values_are_stable() -> None:
    assert [int(s) for s in (HeightSource.HEIGHT, HeightSource.LEVELS, HeightSource.DEFAULT)] == [
        0,
        1,
        2,
    ]


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"height": "15"}, (15.0, HeightSource.HEIGHT)),
        ({"height": "15 m", "building:levels": "2"}, (15.0, HeightSource.HEIGHT)),
        ({"building:height": "8"}, (8.0, HeightSource.HEIGHT)),
        ({"height": "20", "building:height": "8"}, (20.0, HeightSource.HEIGHT)),
        ({"height": "unknow", "building:height": "12"}, (12.0, HeightSource.HEIGHT)),
        ({"building:height": "40 ft"}, (40 * FOOT_M, HeightSource.HEIGHT)),
        ({"building:levels": "4"}, (12.8, HeightSource.LEVELS)),
        ({"building:levels": "1,2,3"}, (9.6, HeightSource.LEVELS)),
        ({"building:levels": "2.5"}, (8.0, HeightSource.LEVELS)),
        ({"height": "0", "building:levels": "2"}, (6.4, HeightSource.LEVELS)),
        ({"height": "500", "building:levels": "3"}, (9.6, HeightSource.LEVELS)),
        ({"height": "unknow", "building:levels": "5"}, (16.0, HeightSource.LEVELS)),
    ],
)
def test_tagged_height_prefers_height_then_building_height_then_levels(
    tags: dict[str, str], expected: tuple[float, HeightSource]
) -> None:
    result = tagged_height(tags)
    assert result is not None
    assert result[0] == pytest.approx(expected[0], abs=1e-3)
    assert result[1] == expected[1]


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {"building": "yes"},
        {"height": "unknow"},
        {"height": "0"},
        {"building:levels": "0"},
        {"building:levels": "many"},
        {"height": "-4", "building:levels": "81"},
    ],
)
def test_tagged_height_is_none_when_nothing_parses(tags: dict[str, str]) -> None:
    assert tagged_height(tags) is None


def words(kind: BuildingKind, text: str) -> list[tuple[str, BuildingKind]]:
    return [(word, kind) for word in text.split()]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("yes", BuildingKind.UNKNOWN),
        ("something_unlisted", BuildingKind.UNKNOWN),
        *words(
            BuildingKind.RESIDENTIAL,
            "apartments residential house detached semidetached_house terrace bungalow "
            "dormitory hut farm cabin",
        ),
        *words(BuildingKind.COMMERCIAL, "commercial retail office supermarket kiosk hotel"),
        *words(BuildingKind.INDUSTRIAL, "industrial warehouse factory manufacture"),
        *words(
            BuildingKind.CIVIC,
            "school university college kindergarten hospital public government civic "
            "train_station transportation fire_station",
        ),
        *words(
            BuildingKind.RELIGIOUS,
            "religious temple mosque church chapel shrine cathedral synagogue gurudwara",
        ),
        *words(
            BuildingKind.OTHER,
            "roof garage garages shed toilets water_tower service carport construction ruins "
            "greenhouse stadium",
        ),
        ("Apartments", BuildingKind.RESIDENTIAL),
        ("YES", BuildingKind.UNKNOWN),
        ("School", BuildingKind.CIVIC),
    ],
)
def test_building_kind_maps_values(value: str, expected: BuildingKind) -> None:
    assert building_kind(value) == expected


@pytest.mark.parametrize(
    "value", ["no", "NO", "No", "proposed", "demolished", "destroyed", "razed", "Razed"]
)
def test_building_kind_is_none_for_non_buildings(value: str) -> None:
    assert building_kind(value) is None


def test_building_kind_numeric_values_are_stable() -> None:
    assert [int(k) for k in BuildingKind] == [0, 1, 2, 3, 4, 5, 6]
    assert BuildingKind.UNKNOWN == 0
    assert BuildingKind.OTHER == 6


@pytest.mark.parametrize(
    ("highway", "expected"),
    [
        ("motorway", RoadClass.MAJOR),
        ("motorway_link", RoadClass.MAJOR),
        ("trunk", RoadClass.MAJOR),
        ("trunk_link", RoadClass.MAJOR),
        ("primary", RoadClass.PRIMARY),
        ("primary_link", RoadClass.PRIMARY),
        ("secondary", RoadClass.SECONDARY),
        ("secondary_link", RoadClass.SECONDARY),
        ("tertiary", RoadClass.TERTIARY),
        ("tertiary_link", RoadClass.TERTIARY),
        ("residential", RoadClass.LOCAL),
        ("unclassified", RoadClass.LOCAL),
        ("living_street", RoadClass.LOCAL),
        ("road", RoadClass.LOCAL),
        ("service", RoadClass.SERVICE),
        ("track", RoadClass.SERVICE),
    ],
)
def test_road_class_maps_drivable_highways(highway: str, expected: RoadClass) -> None:
    assert road_class(highway) == expected


@pytest.mark.parametrize(
    "highway",
    [
        "footway",
        "path",
        "steps",
        "cycleway",
        "pedestrian",
        "construction",
        "proposed",
        "platform",
        "bus_guideway",
        "raceway",
        "bridleway",
        "corridor",
        "elevator",
        "",
    ],
)
def test_road_class_is_none_for_non_drivable_highways(highway: str) -> None:
    assert road_class(highway) is None


def test_road_class_numeric_values_are_stable() -> None:
    assert [int(c) for c in RoadClass] == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("value", ["yes", "true", "1"])
def test_oneway_forward_values(value: str) -> None:
    assert oneway({"oneway": value}) == 1


@pytest.mark.parametrize("value", ["-1", "reverse"])
def test_oneway_reverse_values(value: str) -> None:
    assert oneway({"oneway": value}) == -1


@pytest.mark.parametrize("value", ["no", "false", "0"])
def test_oneway_explicit_no_is_two_way(value: str) -> None:
    assert oneway({"oneway": value}) == 0


@pytest.mark.parametrize(
    "tags",
    [
        {"junction": "roundabout"},
        {"junction": "circular"},
        {"highway": "motorway"},
        {"highway": "motorway_link"},
        {"highway": "motorway", "oneway": "alternating"},
        {"junction": "roundabout", "highway": "residential"},
    ],
)
def test_oneway_implied_by_junction_or_motorway(tags: dict[str, str]) -> None:
    assert oneway(tags) == 1


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {"highway": "residential"},
        {"highway": "trunk"},
        {"oneway": "reversible", "highway": "residential"},
        {"oneway": "alternating"},
        {"oneway": "no", "junction": "roundabout"},
        {"oneway": "no", "highway": "motorway"},
        {"junction": "yes"},
    ],
)
def test_oneway_defaults_to_two_way(tags: dict[str, str]) -> None:
    assert oneway(tags) == 0


def test_oneway_explicit_reverse_beats_roundabout() -> None:
    assert oneway({"oneway": "-1", "junction": "roundabout"}) == -1
