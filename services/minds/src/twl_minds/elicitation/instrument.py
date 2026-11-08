import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROMPT_VERSION = "elicit-v1"
CHANNELS = ("social", "video", "creators", "in_app", "outdoor")
TIE_KINDS = ("household", "neighbour", "friend", "work")
LADDER = (0.7, 0.85, 1.0, 1.15, 1.3)
MAX_ORDERS_PER_MONTH = 60
MAX_REASON_CHARS = 400
SCALE = (1, 2, 3, 4, 5)
TIER_WORDS = ("lower-income", "middle-income", "higher-income")

_TIE_WORDS = {
    "household": "the people you live with",
    "neighbour": "your neighbours",
    "friend": "friends elsewhere in the city",
    "work": "people you work or study with",
}
_CHANNEL_WORDS = {
    "social": "ads on Instagram or Facebook",
    "video": "ads on YouTube or streaming (OTT) apps",
    "creators": "influencers or creators you follow recommending products",
    "in_app": "ads inside quick-commerce apps while you shop",
    "outdoor": "hoardings and billboards you pass",
}
_SEX_NOUN = {"Female": "woman", "Male": "man"}
_MARITAL = {
    "Currently Married": "married",
    "Never Married": "unmarried",
    "Widowed": "widowed",
    "Divorced": "divorced",
    "Separated": "separated",
}
_EDUCATION = {
    "Illiterate": "You cannot read or write.",
    "Literate without education level": "You can read and write but had no formal schooling.",
    "Below Primary": "You left school before finishing primary school.",
    "Primary": "You finished primary school.",
    "Middle": "You finished middle school.",
    "Matric/Secondary": "You finished secondary school (class 10).",
    "Higher Secondary/Intermediate Pre-University/Senior Secondary": (
        "You finished higher secondary school (class 12)."
    ),
    "Non-technical diploma or certificate not equal to degree": (
        "You have a non-technical diploma or certificate."
    ),
    "Technical diploma or certificate not equal to degree": (
        "You have a technical diploma or certificate."
    ),
    "Graduate & above": "You have a bachelor's degree or higher.",
}
_NO_OCCUPATION = "No Occupation / Retired / Homemaker"

_SYSTEM = """\
This is a market research survey about a new product. Answer it in character as the person \
described in the user message. There are no right or wrong answers. Answer as this person really \
would, not as they think the researcher hopes: saying you would not buy is as useful as saying \
you would.

The product
Name: {name}
Category: {category}
Pack size: {pack_size}
Price: {price} a pack
How the brand describes it: {description}

Prices are in Indian rupees (₹). The product is sold through quick-commerce apps, which deliver \
groceries in 10 to 20 minutes from a nearby dark store (a small warehouse that serves only app \
orders).

Answer each question below. Each starts with its JSON key.

purchase_intent: would you buy it at least once in its first month on sale, at {price}?
1 definitely would not buy, 2 probably would not buy, 3 might or might not buy, 4 probably would \
buy, 5 definitely would buy.

price_ladder: if it were sold at each of these prices, would you buy it at least once in its \
first month? true for yes, false for no.
Prices: {ladder}.

orders_per_month: suppose you do buy it at {price}. In a typical month, how many times would you \
order it? A whole number from 0 to {max_orders}.

mention: if you bought it, how likely would you be to mention it to each of these?
{ties}
1 very unlikely, 2 unlikely, 3 neither likely nor unlikely, 4 likely, 5 very likely. If you have \
no such people, for example if you live alone, answer 1 for them.

channels: how often do ads in each of these places lead you to look a product up, ask about it or \
buy it?
{channels}
1 never, 2 rarely, 3 sometimes, 4 often, 5 very often.

reason: why you would or would not buy it, in one or two short sentences of your own, at most \
{max_reason} characters.

Answer only with the JSON object."""


@dataclass(frozen=True)
class Brief:
    product_name: str
    category: str
    description: str
    price_inr: float
    pack_size: str

    def __post_init__(self) -> None:
        _require_text(self, ("product_name", "category", "description", "pack_size"))
        _require_price(self.price_inr)
        _require_whole_rupees(self.price_inr)


@dataclass(frozen=True)
class PersonaProfile:
    age: int
    sex: str
    marital: str
    education: str
    occupation: str
    language: str
    district: str
    tier: int
    household_size: int

    def __post_init__(self) -> None:
        _require_text(self, ("sex", "marital", "education", "occupation", "language", "district"))
        if not 18 <= self.age <= 110:
            raise ValueError(f"age must be in 18..110, got {self.age}")
        if not 0 <= self.tier < len(TIER_WORDS):
            raise ValueError(f"tier must be in 0..{len(TIER_WORDS) - 1}, got {self.tier}")
        if self.household_size < 1:
            raise ValueError(f"household_size must be at least 1, got {self.household_size}")


class InvalidAnswer(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


Scale = Annotated[int, Field(ge=SCALE[0], le=SCALE[-1])]


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    purchase_intent: Scale
    price_ladder: dict[int, bool]
    orders_per_month: Annotated[int, Field(ge=0, le=MAX_ORDERS_PER_MONTH)]
    mention: dict[str, Scale]
    channels: dict[str, Scale]
    reason: Annotated[str, Field(max_length=MAX_REASON_CHARS)]

    @field_validator("price_ladder")
    @classmethod
    def _monotone(cls, ladder: dict[int, bool]) -> dict[int, bool]:
        if not ladder:
            raise ValueError("price_ladder is empty")
        ordered = dict(sorted(ladder.items()))
        if any(higher and not lower for lower, higher in pairwise(ordered.values())):
            raise ValueError("a yes at a higher price follows a no at a lower one")
        return ordered

    @field_validator("mention")
    @classmethod
    def _tie_kinds(cls, mention: dict[str, int]) -> dict[str, int]:
        return _exact_keys("mention", mention, TIE_KINDS)

    @field_validator("channels")
    @classmethod
    def _channels(cls, channels: dict[str, int]) -> dict[str, int]:
        return _exact_keys("channels", channels, CHANNELS)

    @field_validator("reason")
    @classmethod
    def _reason(cls, reason: str) -> str:
        if not reason.strip():
            raise ValueError("reason is blank")
        return reason


def price_ladder(price_inr: float) -> tuple[int, ...]:
    _require_price(price_inr)
    ladder = tuple(max(1, round(price_inr * m)) for m in LADDER)
    if any(higher <= lower for lower, higher in pairwise(ladder)):
        raise ValueError(f"price {price_inr} is too low for distinct whole-rupee steps: {ladder}")
    return ladder


def answer_schema(ladder: Sequence[int]) -> dict[str, Any]:
    prices = _checked_ladder(ladder)
    return _object(
        {
            "purchase_intent": _scale(
                "1 definitely would not buy to 5 definitely would buy, at the stated price"
            ),
            "price_ladder": _object(
                {
                    _ladder_key(p): {"type": "boolean", "description": f"Would buy at {_rupees(p)}"}
                    for p in prices
                },
                "For each price, true if you would buy at that price, false if not",
            ),
            "orders_per_month": {
                "type": "integer",
                "description": (
                    f"Orders in a typical month if you buy it, 0 to {MAX_ORDERS_PER_MONTH}"
                ),
            },
            "mention": _object(
                {kind: _scale(_TIE_WORDS[kind]) for kind in TIE_KINDS},
                "How likely you are to mention it to each, 1 very unlikely to 5 very likely",
            ),
            "channels": _object(
                {channel: _scale(_CHANNEL_WORDS[channel]) for channel in CHANNELS},
                "How often ads there lead you to look a product up, ask about it or buy it, "
                "1 never to 5 very often",
            ),
            "reason": {
                "type": "string",
                "description": (
                    "Why you would or would not buy it, one or two short sentences, "
                    f"at most {MAX_REASON_CHARS} characters"
                ),
            },
        }
    )


def parse_answer(text: str, ladder: Sequence[int]) -> Answer:
    prices = _checked_ladder(ladder)
    try:
        data = json.loads(text, object_pairs_hook=_unique_keys)
    except ValueError as error:
        raise InvalidAnswer(f"not JSON: {error}") from error
    if not isinstance(data, dict):
        raise InvalidAnswer("not a JSON object")
    raw = data.get("price_ladder")
    if isinstance(raw, dict):
        keys = [_ladder_key(p) for p in prices]
        if sorted(raw) != sorted(keys):
            raise InvalidAnswer(f"price_ladder keys {sorted(raw)} are not {keys}")
        data = {**data, "price_ladder": {p: raw[_ladder_key(p)] for p in prices}}
    try:
        answer = Answer.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "answer"
        raise InvalidAnswer(f"{where}: {first['msg']}") from error
    _check_intent_matches_ladder(answer, prices[LADDER.index(1.0)])
    return answer


def _check_intent_matches_ladder(answer: Answer, stated: int) -> None:
    """Both ask whether they would buy at the stated price: a probable no (1 or 2) with a yes there,
    or a probable yes (4 or 5) with a no, is a contradiction, not an answer."""
    intent, yes = answer.purchase_intent, answer.price_ladder[stated]
    if (intent <= 2 and yes) or (intent >= 4 and not yes):
        raise InvalidAnswer(
            f"purchase_intent {intent} contradicts price_ladder {'yes' if yes else 'no'} at "
            f"the stated price"
        )


def system_prompt(brief: Brief) -> str:
    price = _rupees(brief.price_inr)
    return _SYSTEM.format(
        name=_one_line(brief.product_name),
        category=_one_line(brief.category),
        pack_size=_one_line(brief.pack_size),
        price=price,
        description=_one_line(brief.description),
        ladder=", ".join(_rupees(p) for p in price_ladder(brief.price_inr)),
        max_orders=MAX_ORDERS_PER_MONTH,
        ties="\n".join(f"- {kind}: {_TIE_WORDS[kind]}" for kind in TIE_KINDS),
        channels="\n".join(f"- {channel}: {_CHANNEL_WORDS[channel]}" for channel in CHANNELS),
        max_reason=MAX_REASON_CHARS,
    )


def user_message(profile: PersonaProfile) -> str:
    age = profile.age
    article = "an" if age == 18 or 80 <= age <= 89 else "a"
    marital = _MARITAL.get(profile.marital, profile.marital.lower())
    noun = _SEX_NOUN.get(profile.sex, "person")
    people = "person (you live alone)" if profile.household_size == 1 else "people"
    if profile.occupation == _NO_OCCUPATION:
        work = "You have no paid occupation."
    else:
        work = f"Your occupation is {profile.occupation}."
    return " ".join(
        (
            f"You are {article} {age}-year-old {marital} {noun} who lives in {profile.district}.",
            f"Your household has {profile.household_size} {people} and is in the city's "
            f"{TIER_WORDS[profile.tier]} group.",
            _EDUCATION.get(profile.education, f"Your education level is {profile.education}."),
            work,
            f"Your first language is {profile.language}.",
        )
    )


def request_params(
    brief: Brief, profile: PersonaProfile, model: str, max_tokens: int = 1024
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model must not be blank")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_prompt(brief),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_message(profile)}],
        "output_config": {
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": answer_schema(price_ladder(brief.price_inr)),
            },
        },
    }


def _require_text(record: object, names: Sequence[str]) -> None:
    for name in names:
        value = getattr(record, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be blank")


def _require_price(price_inr: float) -> None:
    if not (math.isfinite(price_inr) and price_inr > 0):
        raise ValueError(f"price_inr must be finite and positive, got {price_inr}")


def _require_whole_rupees(price_inr: float) -> None:
    """The ladder's middle rung is the stated price, so the price must be a whole rupee."""
    if not float(price_inr).is_integer():
        raise ValueError(f"price_inr must be whole rupees, got {price_inr}")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _checked_ladder(ladder: Sequence[int]) -> tuple[int, ...]:
    try:
        prices = tuple(operator.index(p) for p in ladder)
    except TypeError as error:
        raise ValueError(f"ladder prices must be whole rupees, got {ladder}") from error
    if not prices or prices[0] < 1 or any(b <= a for a, b in pairwise(prices)):
        raise ValueError(f"ladder must be strictly increasing positive prices, got {prices}")
    return prices


def _ladder_key(price: int) -> str:
    return f"inr_{price}"


def _rupees(amount: float) -> str:
    return f"₹{amount:.0f}" if float(amount).is_integer() else f"₹{amount:.2f}"


def _scale(description: str) -> dict[str, Any]:
    return {"type": "integer", "enum": list(SCALE), "description": description}


def _object(properties: dict[str, dict[str, Any]], description: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object"}
    if description:
        schema["description"] = description
    schema["properties"] = properties
    schema["required"] = list(properties)
    schema["additionalProperties"] = False
    return schema


def _exact_keys(field: str, values: dict[str, int], keys: Sequence[str]) -> dict[str, int]:
    if sorted(values) != sorted(keys):
        raise ValueError(f"{field} keys {sorted(values)} are not {list(keys)}")
    return {key: values[key] for key in keys}


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate key {key!r}")
        data[key] = value
    return data
