import dataclasses
import hashlib
import json
from typing import Any

import pytest

from twl_minds.elicitation.instrument import ElicitationBrief, PersonaProfile
from twl_minds.elicitation.keys import (
    CacheKey,
    brief_digest,
    elicitation_scenario_hash,
    persona_id,
    profile_digest,
)

BRIEF = ElicitationBrief(
    product_name="Kesar Chai",
    category="Beverages",
    description="Saffron tea, ready in two minutes.  क्रीम",
    price_inr=120,
    pack_size="250 g",
)
PROFILE = PersonaProfile(
    age=34,
    sex="Female",
    marital="Currently Married",
    education="Graduate & above",
    occupation="Teacher",
    language="Gujarati",
    district="Ahmedabad",
    tier=1,
    household_size=4,
)
SCENARIO = "ab12cd34"


def _canonical(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(data: bytes | str) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def test_brief_digest_is_sha256_of_the_canonical_json() -> None:
    expected = _sha(
        _canonical(
            {
                "category": "Beverages",
                "description": "Saffron tea, ready in two minutes.  क्रीम",
                "pack_size": "250 g",
                "price_inr": 120,
                "product_name": "Kesar Chai",
            }
        )
    )
    assert brief_digest(BRIEF) == expected


def test_brief_digest_writes_the_price_as_an_integer_not_a_float() -> None:
    as_float = dataclasses.replace(BRIEF, price_inr=120.0)
    assert brief_digest(as_float) == brief_digest(BRIEF)


def test_scenario_hash_is_sha256_of_scenario_newline_digest() -> None:
    expected = _sha(f"{SCENARIO}\n{brief_digest(BRIEF)}")
    assert elicitation_scenario_hash(SCENARIO, BRIEF) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"product_name": "Kesar Chai 2"},
        {"category": "Tea"},
        {"description": "Saffron tea, ready in two minutes. क्रीम"},
        {"description": "Saffron tea, ready in two minutes.  क्रीम "},
        {"price_inr": 121},
        {"pack_size": "500 g"},
    ],
)
def test_any_brief_field_change_moves_the_scenario_hash(changes: dict[str, Any]) -> None:
    other = dataclasses.replace(BRIEF, **changes)
    assert brief_digest(other) != brief_digest(BRIEF)
    assert elicitation_scenario_hash(SCENARIO, other) != elicitation_scenario_hash(SCENARIO, BRIEF)


def test_a_different_scenario_hash_moves_the_elicitation_scenario_hash() -> None:
    assert elicitation_scenario_hash("ab12cd35", BRIEF) != elicitation_scenario_hash(
        SCENARIO, BRIEF
    )


def test_scenario_hash_is_deterministic() -> None:
    assert elicitation_scenario_hash(SCENARIO, BRIEF) == elicitation_scenario_hash(SCENARIO, BRIEF)


@pytest.mark.parametrize("bad", ["", "ABCD", "xyz", "ab cd", "ab\n", "a" * 129])
def test_scenario_hash_must_be_one_to_128_lowercase_hex(bad: str) -> None:
    with pytest.raises(ValueError):
        elicitation_scenario_hash(bad, BRIEF)


@pytest.mark.parametrize("good", ["a", "0", "f" * 128])
def test_scenario_hash_accepts_the_length_bounds(good: str) -> None:
    assert len(elicitation_scenario_hash(good, BRIEF)) == 64


def test_profile_digest_is_sha256_of_all_nine_fields() -> None:
    expected = _sha(_canonical(dataclasses.asdict(PROFILE)))
    assert len(dataclasses.asdict(PROFILE)) == 9
    assert profile_digest(PROFILE) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"age": 35},
        {"sex": "Male"},
        {"marital": "Never Married"},
        {"education": "Primary"},
        {"occupation": "Clerk"},
        {"language": "Hindi"},
        {"district": "Surat"},
        {"tier": 2},
        {"household_size": 5},
    ],
)
def test_any_profile_field_change_moves_the_profile_digest(changes: dict[str, Any]) -> None:
    other = dataclasses.replace(PROFILE, **changes)
    assert profile_digest(other) != profile_digest(PROFILE)


def test_persona_id_is_uuid_colon_first_16_hex_of_profile_digest() -> None:
    assert persona_id("u-123", PROFILE) == f"u-123:{profile_digest(PROFILE)[:16]}"


@pytest.mark.parametrize("changes", [{"district": "Surat"}, {"tier": 0}, {"tier": 2}])
def test_persona_id_differs_when_district_or_tier_differs(changes: dict[str, Any]) -> None:
    other = dataclasses.replace(PROFILE, **changes)
    assert persona_id("u-123", other) != persona_id("u-123", PROFILE)


def test_persona_id_differs_across_uuids_for_the_same_profile() -> None:
    assert persona_id("u-1", PROFILE) != persona_id("u-2", PROFILE)


@pytest.mark.parametrize("bad", ["", "   ", "a:b", "a b", "a\tb", "a\nb", " a"])
def test_persona_id_refuses_a_blank_or_colon_or_whitespace_uuid(bad: str) -> None:
    with pytest.raises(ValueError):
        persona_id(bad, PROFILE)


def test_cache_key_is_frozen_hashable_and_compares_by_value() -> None:
    a = CacheKey("p", "s", "elicit-v1", "m")
    b = CacheKey(persona_id="p", scenario_hash="s", prompt_version="elicit-v1", model="m")
    assert a == b
    assert len({a, b}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.model = "x"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["persona_id", "scenario_hash", "prompt_version", "model"])
@pytest.mark.parametrize("blank", ["", "  "])
def test_cache_key_refuses_blank_fields(field: str, blank: str) -> None:
    values = {"persona_id": "p", "scenario_hash": "s", "prompt_version": "v", "model": "m"}
    values[field] = blank
    with pytest.raises(ValueError):
        CacheKey(**values)
