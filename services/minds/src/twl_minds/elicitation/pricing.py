import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass, fields

PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing, read 2026-11-11"
BATCH_FACTOR = 0.5


class UnknownModel(KeyError):
    def __init__(self, model: object) -> None:
        super().__init__(f"no price for model {model!r}")
        self.model = model

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class Rates:
    """US dollars per million tokens."""

    input: float
    cache_write_5m: float
    cache_read: float
    output: float

    def __post_init__(self) -> None:
        for field in fields(self):
            rate = getattr(self, field.name)
            if not (math.isfinite(rate) and rate >= 0):
                raise ValueError(f"{field.name} must be finite and non-negative, got {rate}")
        if self.cache_write_5m < max(self.input, self.cache_read):
            raise ValueError("cache_write_5m must be the dearest input rate")

    def at_least(self, other: "Rates") -> bool:
        return all(getattr(self, f.name) >= getattr(other, f.name) for f in fields(self))


@dataclass(frozen=True)
class ModelPrice:
    base: Rates
    long: Rates | None = None
    long_threshold: int | None = None

    def __post_init__(self) -> None:
        if (self.long is None) != (self.long_threshold is None):
            raise ValueError("long and long_threshold must be given together")
        if self.long_threshold is not None and self.long_threshold < 1:
            raise ValueError(f"long_threshold must be at least 1, got {self.long_threshold}")
        if self.long is not None and not self.long.at_least(self.base):
            raise ValueError("long-prompt rates must not be below the base rates")

    def rates(self, prompt_tokens: int) -> Rates:
        if (
            self.long is not None
            and self.long_threshold is not None
            and prompt_tokens > self.long_threshold
        ):
            return self.long
        return self.base


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(
        base=Rates(0.10, 0.125, 0.01, 0.50),
        long=Rates(0.50, 0.625, 0.05, 2.50),
        long_threshold=100_000,
    ),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        for field in fields(self):
            _count(field.name, getattr(self, field.name))

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


def model_price(model: str) -> ModelPrice:
    price = PRICES.get(model)
    if price is None:
        raise UnknownModel(model)
    return price


def cost_usd(model: str, usage: Usage, batch: bool) -> float:
    rates = model_price(model).rates(usage.prompt_tokens)
    cost = (
        usage.input_tokens * rates.input
        + usage.cache_creation_input_tokens * rates.cache_write_5m
        + usage.cache_read_input_tokens * rates.cache_read
        + usage.output_tokens * rates.output
    ) / 1e6
    return cost * BATCH_FACTOR if batch else cost


def upper_bound_usd(
    model: str, prompt_tokens: Iterable[int], max_tokens: int, batch: bool
) -> float:
    """Bills each prompt as a full cache write and the full max_tokens of output. Cache writes are
    the dearest input rate and long-prompt rates are never below base ones (both checked on
    construction), so no usage within those sizes costs more. The bound covers 5-minute cache
    writes only, which is what `request_params` asks for; a 1-hour TTL needs its own rate here."""
    model_price(model)
    output = _count("max_tokens", max_tokens)
    return sum(
        (
            cost_usd(model, Usage(0, _count("prompt_tokens", prompt), 0, output), batch)
            for prompt in prompt_tokens
        ),
        start=0.0,
    )


def _count(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}") from error
    if count < 0:
        raise ValueError(f"{name} must be a non-negative int, got {count}")
    return count
