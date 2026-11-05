from dataclasses import replace
from importlib.resources import files
from typing import Any

import pytest

from twl_citybake.config import StoresConfig, TravelBand, TravelConfig, load_city
from twl_citybake.osm.tags import RoadClass

SPEEDS = {cls.name.lower(): 20.0 for cls in RoadClass}


def test_stores_defaults_are_the_documented_values() -> None:
    config = StoresConfig()
    assert config.radius_m == 2500.0
    assert config.coverage_target == 0.92
    assert config.max_stores == 80
    assert config.candidate_cell_m == 500.0


def test_travel_defaults_are_the_approved_speeds_and_bands() -> None:
    config = TravelConfig()
    assert config.h3_resolution == 9
    assert dict(config.speed_kmh) == {
        "major": 40.0,
        "primary": 30.0,
        "secondary": 26.0,
        "tertiary": 22.0,
        "local": 16.0,
        "service": 12.0,
    }
    assert {(b.name, b.start_minute, b.factor) for b in config.bands} == {
        ("night", 1320, 1.25),
        ("morning_peak", 480, 0.8),
        ("midday", 660, 1.0),
        ("evening_peak", 1020, 0.7),
    }


def test_ahmedabad_carries_the_stores_and_travel_sections() -> None:
    city = load_city("ahmedabad")
    assert city.stores == StoresConfig()
    assert city.travel.h3_resolution == 9
    assert {(b.name, b.start_minute, b.factor) for b in city.travel.bands} == {
        (b.name, b.start_minute, b.factor) for b in TravelConfig().bands
    }
    assert dict(city.travel.speed_kmh) == dict(TravelConfig().speed_kmh)


def test_ahmedabad_toml_says_the_bands_and_speeds_are_assumptions() -> None:
    text = (files("twl_citybake.cities") / "ahmedabad.toml").read_text(encoding="utf-8")
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    assert any("assum" in line.lower() for line in comments)


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"radius_m": 0.0}, "radius_m"),
        ({"radius_m": -1.0}, "radius_m"),
        ({"radius_m": float("nan")}, "radius_m"),
        ({"candidate_cell_m": 0.0}, "candidate_cell_m"),
        ({"candidate_cell_m": -5.0}, "candidate_cell_m"),
        ({"candidate_cell_m": float("nan")}, "candidate_cell_m"),
        ({"coverage_target": 0.0}, "coverage_target"),
        ({"coverage_target": -0.1}, "coverage_target"),
        ({"coverage_target": 1.0001}, "coverage_target"),
        ({"coverage_target": float("nan")}, "coverage_target"),
        ({"max_stores": 0}, "max_stores"),
        ({"max_stores": -3}, "max_stores"),
        ({"max_stores": 80.0}, "max_stores"),
        ({"max_stores": True}, "max_stores"),
    ],
)
def test_stores_config_rejects_invalid_values_naming_the_key(
    change: dict[str, Any], key: str
) -> None:
    with pytest.raises(ValueError, match=key):
        replace(StoresConfig(), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"coverage_target": 1.0},
        {"coverage_target": 1e-9},
        {"max_stores": 1},
        {"radius_m": 1.0},
    ],
)
def test_stores_config_accepts_boundary_values(change: dict[str, Any]) -> None:
    replace(StoresConfig(), **change)


@pytest.mark.parametrize("resolution", [-1, 16, 9.0, True, "9"])
def test_h3_resolution_must_be_an_integer_in_zero_to_fifteen(resolution: Any) -> None:
    with pytest.raises(ValueError, match="h3_resolution"):
        replace(TravelConfig(), h3_resolution=resolution)


@pytest.mark.parametrize("resolution", [0, 15])
def test_h3_resolution_accepts_both_ends(resolution: int) -> None:
    assert replace(TravelConfig(), h3_resolution=resolution).h3_resolution == resolution


@pytest.mark.parametrize("bad", [0.0, -12.0, float("nan")])
@pytest.mark.parametrize("cls", [c.name.lower() for c in RoadClass])
def test_every_road_class_speed_must_be_positive(cls: str, bad: float) -> None:
    with pytest.raises(ValueError, match="speed_kmh"):
        replace(TravelConfig(), speed_kmh={**SPEEDS, cls: bad})


@pytest.mark.parametrize("cls", [c.name.lower() for c in RoadClass])
def test_a_missing_road_class_speed_is_rejected(cls: str) -> None:
    speeds = {k: v for k, v in SPEEDS.items() if k != cls}
    with pytest.raises(ValueError, match="speed_kmh"):
        replace(TravelConfig(), speed_kmh=speeds)


def bands_of(spec: tuple[tuple[str, int, float], ...]) -> TravelConfig:
    bands = tuple(TravelBand(name=n, start_minute=s, factor=f) for n, s, f in spec)
    return replace(TravelConfig(), bands=bands)


@pytest.mark.parametrize(
    "spec",
    [
        (),
        (("a", 0, 1.0), ("b", 0, 1.0)),
        (("a", 600, 1.0), ("b", 700, 1.0), ("c", 600, 1.0)),
        (("a", 1440, 1.0),),
        (("a", -1, 1.0),),
        (("a", 0, 1.0), ("b", 1500, 1.0)),
        (("a", 0, 0.0),),
        (("a", 0, -0.5),),
    ],
    ids=[
        "empty",
        "duplicate-start",
        "duplicate-later",
        "start-1440",
        "start-negative",
        "past-day",
        "zero-factor",
        "negative-factor",
    ],
)
def test_bands_need_distinct_starts_within_the_day_and_positive_factors(
    spec: tuple[tuple[str, int, float], ...],
) -> None:
    with pytest.raises(ValueError, match="bands"):
        bands_of(spec)


@pytest.mark.parametrize("start", [0, 600, 1439])
def test_a_single_band_anywhere_in_the_day_covers_the_day_by_wrapping(start: int) -> None:
    config = bands_of((("all", start, 1.0),))
    assert [b.start_minute for b in config.bands] == [start]


def test_stores_and_travel_configs_are_frozen() -> None:
    with pytest.raises(AttributeError):
        StoresConfig().radius_m = 1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        TravelConfig().h3_resolution = 8  # type: ignore[misc]
