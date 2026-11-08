import copy
import dataclasses
import hashlib
import itertools
import json
import math
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError

from twl_minds.elicitation.instrument import (
    CHANNELS,
    LADDER,
    MAX_ORDERS_PER_MONTH,
    MAX_REASON_CHARS,
    PROMPT_VERSION,
    TIE_KINDS,
    Answer,
    Brief,
    InvalidAnswer,
    PersonaProfile,
    answer_schema,
    parse_answer,
    price_ladder,
    request_params,
    system_prompt,
    user_message,
)

UNSUPPORTED = {"minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "pattern"}
SAMPLING_KEYS = {"temperature", "top_p", "top_k"}


def make_brief(**over: Any) -> Brief:
    base: dict[str, Any] = {
        "product_name": "Kesar Kulfi Bites",
        "category": "frozen desserts",
        "description": "Saffron kulfi bites made with full-cream milk.",
        "price_inr": 49.0,
        "pack_size": "200 g",
    }
    base.update(over)
    return Brief(**base)


def make_profile(**over: Any) -> PersonaProfile:
    base: dict[str, Any] = {
        "age": 34,
        "sex": "Female",
        "marital": "Married",
        "education": "Graduate",
        "occupation": "Teacher",
        "language": "Gujarati",
        "district": "Bodakdev",
        "tier": 2,
        "household_size": 4,
    }
    base.update(over)
    return PersonaProfile(**base)


def valid_payload(ladder: tuple[int, ...], yes_count: int = 3) -> dict[str, Any]:
    return {
        "purchase_intent": 4,
        "price_ladder": {f"inr_{p}": i < yes_count for i, p in enumerate(ladder)},
        "orders_per_month": 6,
        "mention": {k: 3 for k in TIE_KINDS},
        "channels": {k: 2 for k in CHANNELS},
        "reason": "It sounds nice for the family on weekends.",
    }


LADDER_49 = price_ladder(49)


def parse(payload: dict[str, Any], ladder: tuple[int, ...] = LADDER_49) -> Answer:
    return parse_answer(json.dumps(payload), ladder)


def walk(node: Any) -> Any:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def test_constants_match_spec() -> None:
    assert PROMPT_VERSION == "elicit-v1"
    assert CHANNELS == ("social", "video", "creators", "in_app", "outdoor")
    assert TIE_KINDS == ("household", "neighbour", "friend", "work")
    assert LADDER == (0.7, 0.85, 1.0, 1.15, 1.3)
    assert MAX_ORDERS_PER_MONTH == 60
    assert MAX_REASON_CHARS == 400


def test_price_ladder_worked_examples() -> None:
    assert price_ladder(49) == (34, 42, 49, 56, 64)
    assert price_ladder(100) == (70, 85, 100, 115, 130)


def test_price_ladder_rounds_half_to_even() -> None:
    assert price_ladder(15) == (10, 13, 15, 17, 20)
    assert price_ladder(35) == (24, 30, 35, 40, 46)


def test_price_ladder_accepts_float_price() -> None:
    assert price_ladder(49.0) == (34, 42, 49, 56, 64)


@pytest.mark.parametrize("price", [1, 2, 3, 5])
def test_price_ladder_refuses_price_too_low_to_ladder(price: float) -> None:
    with pytest.raises(ValueError):
        price_ladder(price)


@given(st.floats(min_value=0.01, max_value=100000, allow_nan=False, allow_infinity=False))
def test_price_ladder_is_strictly_increasing_positive_ints_or_refused(price: float) -> None:
    expected = [max(1, round(price * m)) for m in LADDER]
    try:
        ladder = price_ladder(price)
    except ValueError:
        assert any(a >= b for a, b in itertools.pairwise(expected))
        return
    assert len(ladder) == len(LADDER)
    assert all(isinstance(p, int) and p >= 1 for p in ladder)
    assert all(a < b for a, b in itertools.pairwise(ladder))
    assert ladder == tuple(max(1, round(price * m)) for m in LADDER)


@given(st.integers(min_value=20, max_value=100000))
def test_price_ladder_centre_is_the_price_for_ladderable_integers(price: int) -> None:
    assert price_ladder(price)[2] == price


def test_schema_objects_forbid_extras_and_require_every_key() -> None:
    schema = answer_schema(LADDER_49)
    objects = [n for n in walk(schema) if n.get("type") == "object"]
    assert len(objects) == 4
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])


def test_schema_uses_no_unsupported_keyword() -> None:
    def check(node: dict[str, Any]) -> None:
        assert not UNSUPPORTED & set(node)
        for child in node.get("properties", {}).values():
            check(child)

    check(answer_schema(LADDER_49))


def test_schema_uses_only_the_supported_keyword_set() -> None:
    allowed = {"type", "properties", "required", "additionalProperties", "enum", "description"}

    def check(node: dict[str, Any]) -> None:
        assert set(node) <= allowed
        for child in node.get("properties", {}).values():
            check(child)

    check(answer_schema(LADDER_49))


def test_schema_top_level_keys() -> None:
    schema = answer_schema(LADDER_49)
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {
        "purchase_intent",
        "price_ladder",
        "orders_per_month",
        "mention",
        "channels",
        "reason",
    }


def test_schema_every_property_has_a_description() -> None:
    def check(node: dict[str, Any]) -> None:
        for child in node.get("properties", {}).values():
            assert isinstance(child.get("description"), str) and child["description"].strip()
            check(child)

    check(answer_schema(LADDER_49))


def test_schema_ladder_keys_match_prices() -> None:
    props = answer_schema(LADDER_49)["properties"]["price_ladder"]
    assert set(props["properties"]) == {f"inr_{p}" for p in LADDER_49}
    assert all(v["type"] == "boolean" for v in props["properties"].values())
    other = answer_schema(price_ladder(100))["properties"]["price_ladder"]
    assert set(other["properties"]) == {"inr_70", "inr_85", "inr_100", "inr_115", "inr_130"}


def test_schema_enums_are_exactly_one_to_five() -> None:
    props = answer_schema(LADDER_49)["properties"]
    assert props["purchase_intent"]["type"] == "integer"
    assert props["purchase_intent"]["enum"] == [1, 2, 3, 4, 5]
    assert set(props["mention"]["properties"]) == set(TIE_KINDS)
    assert set(props["channels"]["properties"]) == set(CHANNELS)
    for group in ("mention", "channels"):
        for child in props[group]["properties"].values():
            assert child["type"] == "integer"
            assert child["enum"] == [1, 2, 3, 4, 5]


def test_schema_orders_is_integer_and_reason_is_string() -> None:
    props = answer_schema(LADDER_49)["properties"]
    assert props["orders_per_month"]["type"] == "integer"
    assert props["reason"]["type"] == "string"


def test_schema_is_json_serialisable_and_deterministic() -> None:
    assert json.dumps(answer_schema(LADDER_49), sort_keys=True) == json.dumps(
        answer_schema(LADDER_49), sort_keys=True
    )


def test_valid_answer_parses() -> None:
    answer = parse(valid_payload(LADDER_49))
    assert answer.purchase_intent == 4
    assert answer.orders_per_month == 6
    assert answer.price_ladder == dict(
        zip(LADDER_49, [True, True, True, False, False], strict=True)
    )
    assert answer.mention["friend"] == 3
    assert answer.channels["outdoor"] == 2
    assert answer.reason.startswith("It sounds nice")


def test_answer_is_frozen() -> None:
    answer = parse(valid_payload(LADDER_49))
    with pytest.raises(ValidationError):
        answer.purchase_intent = 1  # type: ignore[misc]


def test_ladder_all_no_and_all_yes_are_valid() -> None:
    none = parse({**valid_payload(LADDER_49, yes_count=0), "purchase_intent": 3})
    every = parse({**valid_payload(LADDER_49, yes_count=5), "purchase_intent": 3})
    assert not any(none.price_ladder.values())
    assert all(every.price_ladder.values())


def test_orders_boundaries_are_valid() -> None:
    for n in (0, MAX_ORDERS_PER_MONTH):
        payload = valid_payload(LADDER_49)
        payload["orders_per_month"] = n
        assert parse(payload).orders_per_month == n


def test_reason_at_max_length_is_valid() -> None:
    payload = valid_payload(LADDER_49)
    payload["reason"] = "x" * MAX_REASON_CHARS
    assert len(parse(payload).reason) == MAX_REASON_CHARS


@pytest.mark.parametrize("text", ["", "not json", "{", "[]", "null", "3", '"s"'])
def test_non_json_object_text_is_refused(text: str) -> None:
    with pytest.raises(InvalidAnswer):
        parse_answer(text, LADDER_49)


def test_invalid_answer_is_a_value_error() -> None:
    assert issubclass(InvalidAnswer, ValueError)


@pytest.mark.parametrize(
    "key",
    ["purchase_intent", "price_ladder", "orders_per_month", "mention", "channels", "reason"],
)
def test_missing_top_level_key_is_refused(key: str) -> None:
    payload = valid_payload(LADDER_49)
    del payload[key]
    with pytest.raises(InvalidAnswer):
        parse(payload)


def test_extra_top_level_key_is_refused() -> None:
    payload = valid_payload(LADDER_49)
    payload["confidence"] = 3
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize("group", ["mention", "channels", "price_ladder"])
def test_missing_nested_key_is_refused(group: str) -> None:
    payload = valid_payload(LADDER_49)
    del payload[group][next(iter(payload[group]))]
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize("group", ["mention", "channels", "price_ladder"])
def test_extra_nested_key_is_refused(group: str) -> None:
    payload = valid_payload(LADDER_49)
    payload[group]["extra"] = False if group == "price_ladder" else 3
    with pytest.raises(InvalidAnswer):
        parse(payload)


def test_ladder_keys_for_a_different_ladder_are_refused() -> None:
    with pytest.raises(InvalidAnswer):
        parse(valid_payload(price_ladder(100)), LADDER_49)


@pytest.mark.parametrize("value", [0, 6, -1, 3.5, "3", None, True])
def test_purchase_intent_outside_enum_is_refused(value: Any) -> None:
    payload = valid_payload(LADDER_49)
    payload["purchase_intent"] = value
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize("group,key", [("mention", "friend"), ("channels", "social")])
@pytest.mark.parametrize("value", [0, 6, 2.5, "2", None])
def test_nested_scale_outside_enum_is_refused(group: str, key: str, value: Any) -> None:
    payload = valid_payload(LADDER_49)
    payload[group][key] = value
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize("value", [-1, MAX_ORDERS_PER_MONTH + 1, 1000, 2.5, "6", None])
def test_orders_outside_range_is_refused(value: Any) -> None:
    payload = valid_payload(LADDER_49)
    payload["orders_per_month"] = value
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize("value", ["", "   ", "\n\t", "x" * (MAX_REASON_CHARS + 1), None, 5])
def test_bad_reason_is_refused(value: Any) -> None:
    payload = valid_payload(LADDER_49)
    payload["reason"] = value
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_non_boolean_ladder_value_is_refused(value: Any) -> None:
    payload = valid_payload(LADDER_49)
    payload["price_ladder"]["inr_49"] = value
    with pytest.raises(InvalidAnswer):
        parse(payload)


@pytest.mark.parametrize(
    "pattern",
    [
        (False, True, False, False, False),
        (False, False, False, False, True),
        (True, False, True, False, False),
    ],
)
def test_non_monotone_ladder_is_refused(pattern: tuple[bool, ...]) -> None:
    payload = valid_payload(LADDER_49)
    payload["price_ladder"] = {f"inr_{p}": v for p, v in zip(LADDER_49, pattern, strict=True)}
    with pytest.raises(InvalidAnswer):
        parse(payload)


def test_refusal_does_not_mutate_input() -> None:
    payload = valid_payload(LADDER_49)
    before = copy.deepcopy(payload)
    parse(payload)
    assert payload == before


@given(
    yes_count=st.integers(0, 5),
    intent=st.integers(1, 5),
    orders=st.integers(0, MAX_ORDERS_PER_MONTH),
    mention=st.lists(st.integers(1, 5), min_size=4, max_size=4),
    channels=st.lists(st.integers(1, 5), min_size=5, max_size=5),
    reason=st.text(min_size=1, max_size=MAX_REASON_CHARS).filter(lambda s: s.strip() != ""),
    price=st.integers(20, 5000),
)
def test_generated_valid_answers_round_trip(
    yes_count: int,
    intent: int,
    orders: int,
    mention: list[int],
    channels: list[int],
    reason: str,
    price: int,
) -> None:
    yes_at_stated = yes_count > LADDER.index(1.0)
    assume(intent == 3 or (intent > 3) == yes_at_stated)
    ladder = price_ladder(price)
    payload = {
        "purchase_intent": intent,
        "price_ladder": {f"inr_{p}": i < yes_count for i, p in enumerate(ladder)},
        "orders_per_month": orders,
        "mention": dict(zip(TIE_KINDS, mention, strict=True)),
        "channels": dict(zip(CHANNELS, channels, strict=True)),
        "reason": reason,
    }
    answer = parse_answer(json.dumps(payload), ladder)
    assert answer.purchase_intent == intent
    assert answer.orders_per_month == orders
    assert answer.reason == reason
    assert answer.price_ladder == {p: i < yes_count for i, p in enumerate(ladder)}
    assert dict(answer.mention) == dict(zip(TIE_KINDS, mention, strict=True))
    assert dict(answer.channels) == dict(zip(CHANNELS, channels, strict=True))


@given(
    flags=st.lists(st.booleans(), min_size=5, max_size=5),
    price=st.integers(20, 5000),
)
def test_ladder_validity_is_exactly_monotonicity(flags: list[bool], price: int) -> None:
    ladder = price_ladder(price)
    payload = {**valid_payload(ladder), "purchase_intent": 3}
    payload["price_ladder"] = {f"inr_{p}": f for p, f in zip(ladder, flags, strict=True)}
    monotone = all(not (b and not a) for a, b in itertools.pairwise(flags))
    if monotone:
        assert parse_answer(json.dumps(payload), ladder).price_ladder == dict(
            zip(ladder, flags, strict=True)
        )
    else:
        with pytest.raises(InvalidAnswer):
            parse_answer(json.dumps(payload), ladder)


def test_system_prompt_is_identical_across_profiles_for_one_brief() -> None:
    brief = make_brief()
    assert system_prompt(brief) == system_prompt(brief)
    profiles = [make_profile(), make_profile(age=61, district="Naroda", tier=0, household_size=7)]
    assert len({system_prompt(brief) for _ in profiles}) == 1


def test_system_prompt_differs_for_a_different_price() -> None:
    assert system_prompt(make_brief(price_inr=49)) != system_prompt(make_brief(price_inr=79))


def test_system_prompt_differs_for_a_different_product() -> None:
    assert system_prompt(make_brief()) != system_prompt(make_brief(product_name="Masala Chaas"))


def test_system_prompt_lists_ladder_prices() -> None:
    text = system_prompt(make_brief(price_inr=49))
    for p in LADDER_49:
        assert str(p) in text


def test_system_prompt_names_every_channel_and_tie_kind() -> None:
    text = system_prompt(make_brief()).lower()
    for name in ("social", "video", "creators", "in_app", "outdoor"):
        assert name in text
    for name in ("household", "neighbour", "friend", "work"):
        assert name in text


def test_system_prompt_contains_the_brief() -> None:
    brief = make_brief()
    text = system_prompt(brief)
    assert brief.product_name in text
    assert brief.description in text
    assert brief.pack_size in text


def test_prompts_say_group_never_third() -> None:
    texts = [system_prompt(make_brief())] + [user_message(make_profile(tier=t)) for t in (0, 1, 2)]
    for text in texts:
        assert "third" not in text.lower()
    assert "group" in user_message(make_profile()).lower()


@pytest.mark.parametrize(
    "tier,word", [(0, "lower-income"), (1, "middle-income"), (2, "higher-income")]
)
def test_user_message_uses_tier_wording(tier: int, word: str) -> None:
    text = user_message(make_profile(tier=tier))
    assert word in text
    others = {"lower-income", "middle-income", "higher-income"} - {word}
    assert not any(o in text for o in others)


def test_user_message_carries_profile_facts() -> None:
    profile = make_profile(age=34, district="Bodakdev", household_size=4)
    text = user_message(profile)
    assert "34" in text
    assert "Bodakdev" in text
    assert "4" in text
    assert profile.occupation in text
    assert profile.language in text


def test_user_message_is_deterministic_and_varies_with_profile() -> None:
    a, b = make_profile(), make_profile(age=51, district="Maninagar", household_size=2)
    assert user_message(a) == user_message(a)
    assert user_message(a) != user_message(b)
    assert "51" in user_message(b)
    assert "Maninagar" in user_message(b)


def test_persona_profile_has_no_free_text_persona_field() -> None:
    names = {f.name for f in dataclasses.fields(PersonaProfile)}
    assert names == {
        "age",
        "sex",
        "marital",
        "education",
        "occupation",
        "language",
        "district",
        "tier",
        "household_size",
    }
    assert not any("persona" in n or "text" in n or "bio" in n for n in names)


def test_request_params_shape() -> None:
    brief, profile = make_brief(), make_profile()
    params = request_params(brief, profile, "claude-test-model")
    assert params["model"] == "claude-test-model"
    assert params["max_tokens"] == 1024
    assert len(params["system"]) == 1
    block = params["system"][0]
    assert block["type"] == "text"
    assert block["text"] == system_prompt(brief)
    assert block["cache_control"] == {"type": "ephemeral"}
    assert len(params["messages"]) == 1
    message = params["messages"][0]
    assert message["role"] == "user"
    content = message["content"]
    text = content if isinstance(content, str) else "".join(b["text"] for b in content)
    assert text == user_message(profile)
    out = params["output_config"]
    assert out["effort"] == "low"
    assert out["format"] == {
        "type": "json_schema",
        "schema": answer_schema(price_ladder(brief.price_inr)),
    }


def test_request_params_has_no_sampling_or_thinking_keys() -> None:
    params = request_params(make_brief(), make_profile(), "m")
    assert not SAMPLING_KEYS & set(params)
    assert "thinking" not in params


def test_request_params_max_tokens_override() -> None:
    assert request_params(make_brief(), make_profile(), "m", max_tokens=256)["max_tokens"] == 256


def test_request_params_system_block_is_shared_across_profiles() -> None:
    brief = make_brief()
    a = request_params(brief, make_profile(), "m")
    b = request_params(brief, make_profile(age=70, tier=0), "m")
    assert a["system"] == b["system"]
    assert a["messages"] != b["messages"]
    assert a["output_config"] == b["output_config"]


def test_request_params_are_json_serialisable() -> None:
    json.dumps(request_params(make_brief(), make_profile(), "m"))


@pytest.mark.parametrize("field", ["product_name", "category", "description", "pack_size"])
@pytest.mark.parametrize("value", ["", "   "])
def test_brief_refuses_blank_text(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        make_brief(**{field: value})


@pytest.mark.parametrize("price", [0, -1, -49.0, math.nan, math.inf, -math.inf])
def test_brief_refuses_bad_price(price: float) -> None:
    with pytest.raises(ValueError):
        make_brief(price_inr=price)


def test_brief_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_brief().price_inr = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("age", [0, 17, 111, -5])
def test_profile_refuses_age_outside_range(age: int) -> None:
    with pytest.raises(ValueError):
        make_profile(age=age)


@pytest.mark.parametrize("age", [18, 110])
def test_profile_accepts_age_bounds(age: int) -> None:
    assert make_profile(age=age).age == age


@pytest.mark.parametrize("tier", [-1, 3])
def test_profile_refuses_tier_outside_range(tier: int) -> None:
    with pytest.raises(ValueError):
        make_profile(tier=tier)


@pytest.mark.parametrize("size", [0, -1])
def test_profile_refuses_household_below_one(size: int) -> None:
    with pytest.raises(ValueError):
        make_profile(household_size=size)


@pytest.mark.parametrize(
    "field", ["sex", "marital", "education", "occupation", "language", "district"]
)
@pytest.mark.parametrize("value", ["", "  "])
def test_profile_refuses_blank_text(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        make_profile(**{field: value})


def test_profile_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_profile().age = 20  # type: ignore[misc]


INSTRUMENT_DIGEST = "942f4abc3eeed0f832231ff07c06e8d4295f64b0b7e806c55a5541299a984874"


def test_instrument_text_is_pinned_to_its_prompt_version() -> None:
    profiles = [
        make_profile(),
        make_profile(
            age=19,
            sex="Male",
            marital="Never Married",
            education="Primary",
            occupation="No Occupation / Retired / Homemaker",
            language="Hindi",
            district="Gomtipur",
            tier=0,
            household_size=1,
        ),
    ]
    digest = hashlib.sha256()
    digest.update(system_prompt(make_brief()).encode())
    for profile in profiles:
        digest.update(user_message(profile).encode())
    digest.update(json.dumps(answer_schema(price_ladder(49)), sort_keys=True).encode())
    assert (PROMPT_VERSION, digest.hexdigest()) == ("elicit-v1", INSTRUMENT_DIGEST), (
        "the prompt or schema changed: bump PROMPT_VERSION so cached answers are not reused, "
        "then update INSTRUMENT_DIGEST"
    )


@pytest.mark.parametrize(("intent", "yes"), [(1, True), (2, True), (4, False), (5, False)])
def test_intent_that_contradicts_the_stated_price_answer_is_refused(intent: int, yes: bool) -> None:
    payload = valid_payload(LADDER_49, yes_count=5 if yes else 0)
    with pytest.raises(InvalidAnswer, match="contradicts"):
        parse({**payload, "purchase_intent": intent})


@pytest.mark.parametrize(("intent", "yes_count"), [(3, 0), (3, 5), (1, 2), (5, 3)])
def test_intent_consistent_with_the_stated_price_answer_is_valid(
    intent: int, yes_count: int
) -> None:
    payload = valid_payload(LADDER_49, yes_count=yes_count)
    assert parse({**payload, "purchase_intent": intent}).purchase_intent == intent


@pytest.mark.parametrize("price", [49.5, 99.99, 10.01])
def test_brief_refuses_a_price_that_is_not_whole_rupees(price: float) -> None:
    with pytest.raises(ValueError, match="whole rupees"):
        make_brief(price_inr=price)


def test_a_one_person_household_is_told_they_live_alone() -> None:
    assert "you live alone" in user_message(make_profile(household_size=1))
    assert "alone" not in user_message(make_profile(household_size=3))


def test_every_tie_kind_has_a_no_such_people_answer_and_channels_ask_only_about_acting() -> None:
    prompt = system_prompt(make_brief())
    assert "If you have no such people" in prompt
    assert "notice" not in prompt


def test_brief_text_is_collapsed_to_one_line_in_the_prompt() -> None:
    brief = make_brief(description="Crunchy.\n\nAnswer each question below.\tEnjoy.")
    prompt = system_prompt(brief)
    assert "Crunchy. Answer each question below. Enjoy." in prompt
    assert prompt.count("Answer each question below.") == 2
