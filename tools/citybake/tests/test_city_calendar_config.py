import re
from importlib.resources import files
from typing import Any

import pytest
from test_pack_config import FakeResource

from twl_citybake import config as config_module
from twl_citybake.config import load_city

AHMEDABAD = (files("twl_citybake.cities") / "ahmedabad.toml").read_text(encoding="utf-8")
CALENDAR_LINE = re.compile(r"(?m)^(timezone|weekend_days)\s*=.*\n")
FIRST_TABLE = re.compile(r"(?m)^\[")


def load_with(monkeypatch: pytest.MonkeyPatch, top_level: str) -> Any:
    text = CALENDAR_LINE.sub("", AHMEDABAD)
    head = FIRST_TABLE.search(text)
    assert head is not None
    text = text[: head.start()] + top_level + "\n" + text[head.start() :]
    monkeypatch.setattr(config_module, "files", lambda package: FakeResource(text))
    return load_city("ahmedabad")


def test_ahmedabad_is_in_kolkata_with_a_saturday_sunday_weekend() -> None:
    city = load_city("ahmedabad")
    assert city.timezone == "Asia/Kolkata"
    assert city.weekend_days == (6, 7)
    assert isinstance(city.weekend_days, tuple)


def test_a_city_without_the_keys_gets_none(monkeypatch: pytest.MonkeyPatch) -> None:
    city = load_with(monkeypatch, "")
    assert city.timezone is None
    assert city.weekend_days is None


def test_present_keys_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    city = load_with(monkeypatch, 'timezone = "Asia/Dubai"\nweekend_days = [5, 6]')
    assert city.timezone == "Asia/Dubai"
    assert city.weekend_days == (5, 6)


def test_an_empty_weekend_is_loaded_as_an_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_with(monkeypatch, "weekend_days = []").weekend_days == ()


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ('timezone = "Mars/Olympus"', "timezone"),
        ('timezone = "asia/kolkata"', "timezone"),
        ('timezone = ""', "timezone"),
        ("timezone = 5", "timezone"),
        ("weekend_days = [7, 6]", "weekend_days"),
        ("weekend_days = [6, 6]", "weekend_days"),
        ("weekend_days = [0, 6]", "weekend_days"),
        ("weekend_days = [6, 8]", "weekend_days"),
        ("weekend_days = [6.0, 7]", "weekend_days"),
        ("weekend_days = [true]", "weekend_days"),
        ('weekend_days = "67"', "weekend_days"),
    ],
)
def test_bad_calendar_values_are_refused_naming_the_key(
    monkeypatch: pytest.MonkeyPatch, line: str, key: str
) -> None:
    with pytest.raises(ValueError, match=key):
        load_with(monkeypatch, line)
