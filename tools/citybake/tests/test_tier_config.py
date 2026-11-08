import re
from dataclasses import FrozenInstanceError
from importlib.resources import files
from typing import Any

import pytest

from twl_citybake import config as config_module
from twl_citybake.config import CityConfig, TierConfig, load_city

AHMEDABAD = (files("twl_citybake.cities") / "ahmedabad.toml").read_text(encoding="utf-8")
TIER_SECTION = re.compile(r"(?ms)^\[tier\]\n.*?(?=^\[|\Z)")

VALID = {
    "low_share": "0.3",
    "low_share_source": '"low source"',
    "high_share": "0.2",
    "high_share_source": '"high source"',
    "attribution": '"credit"',
}


class FakeResource:
    def __init__(self, text: str) -> None:
        self.text = text

    def __truediv__(self, name: str) -> "FakeResource":
        return self

    def is_file(self) -> bool:
        return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
        return self.text


def load_with(monkeypatch: pytest.MonkeyPatch, tier: dict[str, str] | None) -> CityConfig:
    text = TIER_SECTION.sub("", AHMEDABAD)
    if tier is not None:
        body = "".join(f"{k} = {v}\n" for k, v in tier.items())
        text = text.rstrip("\n") + "\n\n[tier]\n" + body
    monkeypatch.setattr(config_module, "files", lambda package: FakeResource(text))
    return load_city("ahmedabad")


def test_a_city_without_a_tier_table_has_no_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_with(monkeypatch, None).tier is None


def test_a_valid_tier_table_loads_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_with(monkeypatch, VALID).tier == TierConfig(
        low_share=0.3,
        low_share_source="low source",
        high_share=0.2,
        high_share_source="high source",
        attribution="credit",
    )


def test_the_tier_config_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    tier = load_with(monkeypatch, VALID).tier
    assert tier is not None
    with pytest.raises(FrozenInstanceError):
        tier.low_share = 0.5  # type: ignore[misc]


def test_ahmedabad_loads_its_shares_and_sources() -> None:
    tier = load_city("ahmedabad").tier
    assert tier is not None
    assert tier.low_share == 0.38
    assert tier.high_share == 0.20
    assert "Census of India 2011" in tier.low_share_source
    assert "HL-14" in tier.low_share_source
    assert "Assumption" in tier.high_share_source
    assert tier.attribution == "Census of India 2011 (HL-14)"


@pytest.mark.parametrize("missing", sorted(VALID))
def test_a_missing_key_is_refused_naming_it(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        load_with(monkeypatch, {k: v for k, v in VALID.items() if k != missing})


def test_an_unknown_key_is_refused_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="middle_share"):
        load_with(monkeypatch, {**VALID, "middle_share": "0.5"})


@pytest.mark.parametrize("key", ["low_share", "high_share"])
@pytest.mark.parametrize("value", ["0.0", "1.0", "-0.1", "1.5", "0", "1", "true", '"0.3"'])
def test_a_share_that_is_not_a_real_strictly_inside_zero_and_one_is_refused(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, key: value})


@pytest.mark.parametrize(("low", "high"), [("0.6", "0.4"), ("0.7", "0.5")])
def test_shares_that_sum_to_one_or_more_are_refused(
    monkeypatch: pytest.MonkeyPatch, low: str, high: str
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, "low_share": low, "high_share": high})


@pytest.mark.parametrize("key", ["low_share_source", "high_share_source", "attribution"])
@pytest.mark.parametrize("value", ['""', '"   "', "5"])
def test_blank_or_non_text_sources_are_refused(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    with pytest.raises(ValueError):
        load_with(monkeypatch, {**VALID, key: value})
