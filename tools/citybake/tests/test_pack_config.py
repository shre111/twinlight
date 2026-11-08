import re
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from twl_citybake import config as config_module
from twl_citybake.config import DataDirs, PackConfig, load_city
from twl_citybake.osm.tags import RoadClass

AHMEDABAD = (files("twl_citybake.cities") / "ahmedabad.toml").read_text(encoding="utf-8")
PACK_SECTION = re.compile(r"(?ms)^\[pack\]\n.*?(?=^\[|\Z)")


class FakeResource:
    def __init__(self, text: str) -> None:
        self.text = text

    def __truediv__(self, name: str) -> "FakeResource":
        return self

    def is_file(self) -> bool:
        return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
        return self.text


def load_with(monkeypatch: pytest.MonkeyPatch, pack_toml: str | None) -> Any:
    text = PACK_SECTION.sub("", AHMEDABAD)
    if pack_toml is not None:
        text = text.rstrip("\n") + "\n\n[pack]\n" + pack_toml + "\n"
    monkeypatch.setattr(config_module, "files", lambda package: FakeResource(text))
    return load_city("ahmedabad")


def test_pack_defaults_are_the_documented_values() -> None:
    config = PackConfig()
    assert config.min_footprint_m2 == 30.0
    assert config.isolated_m == 3.0
    assert config.tile_level == 6
    assert tuple(config.arterial_classes) == ("major", "primary", "secondary")


def test_a_city_without_a_pack_table_gets_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_with(monkeypatch, None).pack == PackConfig()


def test_the_pack_table_overrides_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    city = load_with(
        monkeypatch,
        'min_footprint_m2 = 50.0\nisolated_m = 5.0\ntile_level = 4\narterial_classes = ["major"]',
    )
    assert city.pack == PackConfig(
        min_footprint_m2=50.0, isolated_m=5.0, tile_level=4, arterial_classes=("major",)
    )
    assert isinstance(city.pack.arterial_classes, tuple)


def test_a_partial_pack_table_keeps_the_other_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    city = load_with(monkeypatch, "isolated_m = 4.0")
    assert city.pack == replace(PackConfig(), isolated_m=4.0)


def test_an_unknown_pack_key_is_refused_naming_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="bogus"):
        load_with(monkeypatch, "bogus = 1")


def test_an_unknown_arterial_class_in_the_toml_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="motorway"):
        load_with(monkeypatch, 'arterial_classes = ["major", "motorway"]')


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"min_footprint_m2": 0.0}, "min_footprint_m2"),
        ({"min_footprint_m2": -1.0}, "min_footprint_m2"),
        ({"min_footprint_m2": float("nan")}, "min_footprint_m2"),
        ({"min_footprint_m2": float("inf")}, "min_footprint_m2"),
        ({"min_footprint_m2": True}, "min_footprint_m2"),
        ({"isolated_m": -0.5}, "isolated_m"),
        ({"isolated_m": float("nan")}, "isolated_m"),
        ({"isolated_m": float("inf")}, "isolated_m"),
        ({"tile_level": -1}, "tile_level"),
        ({"tile_level": 9}, "tile_level"),
        ({"tile_level": 6.5}, "tile_level"),
        ({"tile_level": True}, "tile_level"),
        ({"arterial_classes": ("major", "motorway")}, "motorway"),
    ],
)
def test_pack_config_rejects_bad_values_naming_the_key(change: dict[str, Any], key: str) -> None:
    with pytest.raises(ValueError, match=key):
        replace(PackConfig(), **change)


@pytest.mark.parametrize("name", [road.name.lower() for road in RoadClass])
def test_every_lowercased_road_class_name_is_a_valid_arterial_class(name: str) -> None:
    assert replace(PackConfig(), arterial_classes=(name,)).arterial_classes == (name,)


def test_pack_config_is_frozen() -> None:
    with pytest.raises(AttributeError):
        PackConfig().tile_level = 3  # type: ignore[misc]


def test_ahmedabad_carries_the_d031_values() -> None:
    pack = load_city("ahmedabad").pack
    assert pack.min_footprint_m2 == 30.0
    assert pack.isolated_m == 3.0
    assert pack.tile_level == 6
    assert tuple(pack.arterial_classes) == ("major", "primary", "secondary")


def test_ahmedabad_toml_has_a_pack_table_with_a_comment_citing_d031() -> None:
    section = PACK_SECTION.search(AHMEDABAD)
    assert section is not None
    lines = section.group(0).splitlines()
    comments = [line for line in lines if line.lstrip().startswith("#")]
    assert any("D-031" in line for line in comments)


def test_the_pack_directory_is_under_cache_packs() -> None:
    assert DataDirs(Path("/data")).pack("ahmedabad") == Path("/data/cache/packs/ahmedabad")
