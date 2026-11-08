import json
import re
from dataclasses import FrozenInstanceError
from importlib.resources import files
from typing import Any

import pytest

from twl_citybake import config as config_module
from twl_citybake.config import CityConfig, TierTiltConfig, load_city

AHMEDABAD = (files("twl_citybake.cities") / "ahmedabad.toml").read_text(encoding="utf-8")
TIER_SECTION = re.compile(r"(?ms)^\[tier\]\n.*?(?=^\[|\Z)")
TILT_SECTION = re.compile(r"(?ms)^\[personas\.tier_tilt\]\n.*?(?=^\[|\Z)")

VALID: dict[str, Any] = {
    "education_10plus": ["Matric/Secondary", "Graduate & above"],
    "quintile_population": [3.2, 7.2, 15.5, 28.6, 45.5],
    "female_10plus": [7.5, 15.5, 23.9, 34.6, 53.4],
    "male_10plus": [12.0, 23.1, 33.7, 45.2, 63.6],
    "source": "a source",
}

SPEC_EDUCATION = (
    "Matric/Secondary",
    "Higher Secondary/Intermediate Pre-University/Senior Secondary",
    "Graduate & above",
    "Technical diploma or certificate not equal to degree",
    "Non-technical diploma or certificate not equal to degree",
)


class FakeResource:
    def __init__(self, text: str) -> None:
        self.text = text

    def __truediv__(self, name: str) -> "FakeResource":
        return self

    def is_file(self) -> bool:
        return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
        return self.text


def load_with(
    monkeypatch: pytest.MonkeyPatch, tilt: dict[str, Any] | None, keep_tier: bool = True
) -> CityConfig:
    text = TILT_SECTION.sub("", AHMEDABAD)
    if not keep_tier:
        text = TIER_SECTION.sub("", text)
    if tilt is not None:
        body = "".join(f"{k} = {json.dumps(v)}\n" for k, v in tilt.items())
        text = text.rstrip("\n") + "\n\n[personas.tier_tilt]\n" + body
    monkeypatch.setattr(config_module, "files", lambda package: FakeResource(text))
    return load_city("ahmedabad")


def test_a_city_without_a_tilt_table_has_no_tilt(monkeypatch: pytest.MonkeyPatch) -> None:
    personas = load_with(monkeypatch, None).personas
    assert personas is not None
    assert personas.tier_tilt is None


def test_a_valid_tilt_table_loads_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    personas = load_with(monkeypatch, VALID).personas
    assert personas is not None
    assert personas.tier_tilt == TierTiltConfig(
        education_10plus=("Matric/Secondary", "Graduate & above"),
        quintile_population=(3.2, 7.2, 15.5, 28.6, 45.5),
        female_10plus=(7.5, 15.5, 23.9, 34.6, 53.4),
        male_10plus=(12.0, 23.1, 33.7, 45.2, 63.6),
        source="a source",
    )


def test_the_tilt_config_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    personas = load_with(monkeypatch, VALID).personas
    assert personas is not None and personas.tier_tilt is not None
    with pytest.raises(FrozenInstanceError):
        personas.tier_tilt.source = "x"  # type: ignore[misc]


def test_ahmedabad_loads_the_specs_values() -> None:
    personas = load_city("ahmedabad").personas
    assert personas is not None and personas.tier_tilt is not None
    tilt = personas.tier_tilt
    assert tilt.education_10plus == SPEC_EDUCATION
    assert tilt.quintile_population == (3.2, 7.2, 15.5, 28.6, 45.5)
    assert tilt.female_10plus == (7.5, 15.5, 23.9, 34.6, 53.4)
    assert tilt.male_10plus == (12.0, 23.1, 33.7, 45.2, 63.6)
    assert "NFHS-5" in tilt.source and "Table 2.24" in tilt.source and "Table 2.8" in tilt.source


@pytest.mark.parametrize("missing", sorted(VALID))
def test_a_missing_key_is_refused_naming_it(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        load_with(monkeypatch, {k: v for k, v in VALID.items() if k != missing})


def test_an_unknown_key_is_refused_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="female_college"):
        load_with(monkeypatch, {**VALID, "female_college": [1, 2, 3, 4, 5]})


@pytest.mark.parametrize("value", [[], ["Graduate & above", "Graduate & above"]])
def test_an_empty_or_repeating_education_list_is_refused(
    monkeypatch: pytest.MonkeyPatch, value: list[str]
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, "education_10plus": value})


@pytest.mark.parametrize("key", ["quintile_population", "female_10plus", "male_10plus"])
@pytest.mark.parametrize("length", [0, 4, 6])
def test_a_list_that_is_not_five_long_is_refused(
    monkeypatch: pytest.MonkeyPatch, key: str, length: int
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, key: [20.0] * length})


@pytest.mark.parametrize(
    "population", [[20.0, 20.0, 20.0, 20.0, 19.0], [21.0, 20.0, 20.0, 20.0, 20.0]]
)
def test_quintile_population_not_summing_to_one_hundred_within_half_a_point_is_refused(
    monkeypatch: pytest.MonkeyPatch, population: list[float]
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, "quintile_population": population})


@pytest.mark.parametrize(
    "population", [[20.0, 20.0, 20.0, 20.0, 19.6], [20.0, 20.0, 20.0, 20.0, 20.4]]
)
def test_quintile_population_summing_within_half_a_point_is_accepted(
    monkeypatch: pytest.MonkeyPatch, population: list[float]
) -> None:
    personas = load_with(monkeypatch, {**VALID, "quintile_population": population}).personas
    assert personas is not None and personas.tier_tilt is not None


def test_a_negative_quintile_population_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, "quintile_population": [-5.0, 25.0, 25.0, 25.0, 30.0]})


def test_a_zero_quintile_population_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    personas = load_with(
        monkeypatch, {**VALID, "quintile_population": [0.0, 25.0, 25.0, 25.0, 25.0]}
    ).personas
    assert personas is not None and personas.tier_tilt is not None


@pytest.mark.parametrize("key", ["female_10plus", "male_10plus"])
@pytest.mark.parametrize("bad", [0.0, 100.0, -1.0, 101.0])
def test_a_ten_plus_percent_not_strictly_between_zero_and_one_hundred_is_refused(
    monkeypatch: pytest.MonkeyPatch, key: str, bad: float
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, key: [10.0, 20.0, bad, 40.0, 50.0]})


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_source_is_refused(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, "source": value})


def test_a_non_text_source_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, "source": 5})


def test_a_tilt_without_a_tier_table_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="tier"):
        load_with(monkeypatch, VALID, keep_tier=False)


def test_a_city_with_a_tier_and_no_tilt_still_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_with(monkeypatch, None).tier is not None
