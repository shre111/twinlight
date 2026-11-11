import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from twl_minds.elicitation.pricing import PRICES

MODEL_VAR = "TWL_MINDS_ELICIT_MODEL"
RUN_CAP_VAR = "TWL_MINDS_ELICIT_RUN_CAP_USD"
DAILY_CAP_VAR = "TWL_MINDS_ELICIT_DAILY_CAP_USD"
STORE_VAR = "TWL_MINDS_ELICIT_STORE"
MAX_TOKENS_VAR = "TWL_MINDS_ELICIT_MAX_TOKENS"

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_RUN_CAP_USD = 1.00
DEFAULT_DAILY_CAP_USD = 5.00
DEFAULT_STORE = Path("data/cache/minds/elicitations.duckdb")
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS_RANGE = (256, 16_000)


@dataclass(frozen=True)
class ElicitSettings:
    model: str = DEFAULT_MODEL
    run_cap_usd: float = DEFAULT_RUN_CAP_USD
    daily_cap_usd: float = DEFAULT_DAILY_CAP_USD
    store_path: Path = DEFAULT_STORE
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Self:
        model = environ.get(MODEL_VAR, DEFAULT_MODEL)
        if model not in PRICES:
            raise ValueError(f"{MODEL_VAR} must be a priced model {sorted(PRICES)}, got {model!r}")
        run_cap = _positive_usd(environ, RUN_CAP_VAR, DEFAULT_RUN_CAP_USD)
        daily_cap = _positive_usd(environ, DAILY_CAP_VAR, DEFAULT_DAILY_CAP_USD)
        if run_cap > daily_cap:
            raise ValueError(
                f"{RUN_CAP_VAR} ({run_cap}) must not be above {DAILY_CAP_VAR} ({daily_cap})"
            )
        store = environ.get(STORE_VAR)
        if store is not None and not store.strip():
            raise ValueError(f"{STORE_VAR} must not be blank")
        return cls(
            model=model,
            run_cap_usd=run_cap,
            daily_cap_usd=daily_cap,
            store_path=DEFAULT_STORE if store is None else Path(store),
            max_tokens=_max_tokens(environ),
        )


def _positive_usd(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number of US dollars, got {raw!r}") from error
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and positive, got {raw!r}")
    return value


def _max_tokens(environ: Mapping[str, str]) -> int:
    raw = environ.get(MAX_TOKENS_VAR)
    if raw is None:
        return DEFAULT_MAX_TOKENS
    low, high = MAX_TOKENS_RANGE
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{MAX_TOKENS_VAR} must be a whole number, got {raw!r}") from error
    if not low <= value <= high:
        raise ValueError(f"{MAX_TOKENS_VAR} must be in {low}..{high}, got {value}")
    return value
