import json
import math
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import duckdb

from twl_minds.elicitation.keys import CacheKey
from twl_minds.elicitation.pricing import Usage

STATUSES = ("valid", "invalid", "refused", "truncated", "errored", "expired", "canceled")
MEMORY = ":memory:"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS answers (
    persona_id VARCHAR NOT NULL,
    scenario_hash VARCHAR NOT NULL,
    prompt_version VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    answer_json VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (persona_id, scenario_hash, prompt_version, model)
);
CREATE TABLE IF NOT EXISTS calls (
    call_id VARCHAR PRIMARY KEY,
    persona_id VARCHAR NOT NULL,
    scenario_hash VARCHAR NOT NULL,
    prompt_version VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    status VARCHAR NOT NULL CHECK (status IN ({", ".join(f"'{s}'" for s in STATUSES)})),
    reason VARCHAR NOT NULL,
    input_tokens BIGINT NOT NULL CHECK (input_tokens >= 0),
    cache_creation_input_tokens BIGINT NOT NULL CHECK (cache_creation_input_tokens >= 0),
    cache_read_input_tokens BIGINT NOT NULL CHECK (cache_read_input_tokens >= 0),
    output_tokens BIGINT NOT NULL CHECK (output_tokens >= 0),
    cost_usd DOUBLE NOT NULL CHECK (cost_usd >= 0),
    batch BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
"""

_CACHED = """
SELECT a.persona_id, a.scenario_hash, a.prompt_version, a.model, a.answer_json
FROM (
    SELECT
        unnest(?::VARCHAR[]) AS persona_id,
        unnest(?::VARCHAR[]) AS scenario_hash,
        unnest(?::VARCHAR[]) AS prompt_version,
        unnest(?::VARCHAR[]) AS model
) AS wanted
JOIN answers AS a USING (persona_id, scenario_hash, prompt_version, model)
"""

_INSERT_CALL = """
INSERT INTO calls (
    call_id, persona_id, scenario_hash, prompt_version, model, status, reason, input_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, output_tokens, cost_usd, batch, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ANSWER = """
INSERT INTO answers (persona_id, scenario_hash, prompt_version, model, answer_json, created_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT DO NOTHING
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


class ElicitationLedger(Protocol):
    def cached(self, keys: Iterable[CacheKey]) -> dict[CacheKey, str]: ...

    def record(
        self,
        call_id: str,
        key: CacheKey,
        status: str,
        usage: Usage,
        cost_usd: float,
        batch: bool,
        reason: str = "",
        answer_json: str | None = None,
    ) -> None: ...

    def spent_since(self, since: datetime) -> float: ...

    def status_counts(self, scenario_hash: str) -> dict[str, int]: ...


class ElicitationStore:
    def __init__(self, path: Path | str, clock: Callable[[], datetime] = utc_now) -> None:
        target = str(path)
        if target != MEMORY:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._con = duckdb.connect(target)
        self._con.execute(_SCHEMA)

    def cached(self, keys: Iterable[CacheKey]) -> dict[CacheKey, str]:
        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return {}
        columns = [
            [key.persona_id for key in wanted],
            [key.scenario_hash for key in wanted],
            [key.prompt_version for key in wanted],
            [key.model for key in wanted],
        ]
        rows = self._con.execute(_CACHED, columns).fetchall()
        return {CacheKey(*row[:4]): row[4] for row in rows}

    def record(
        self,
        call_id: str,
        key: CacheKey,
        status: str,
        usage: Usage,
        cost_usd: float,
        batch: bool,
        reason: str = "",
        answer_json: str | None = None,
    ) -> None:
        """`cost_usd` must be `pricing.cost_usd(key.model, usage, batch)`: the ledger is the cost
        guard's only input, so a cost that does not come from the usage hides spend. One store per
        process and thread; DuckDB connections are not shared safely."""
        _check_call(call_id, status, cost_usd, reason, answer_json)
        created_at = _utc(self._clock(), "clock()")
        identity = [key.persona_id, key.scenario_hash, key.prompt_version, key.model]
        self._con.begin()
        try:
            if self._con.execute("SELECT 1 FROM calls WHERE call_id = ?", [call_id]).fetchone():
                raise ValueError(f"call_id {call_id!r} is already recorded")
            self._con.execute(
                _INSERT_CALL,
                [
                    call_id,
                    *identity,
                    status,
                    reason,
                    int(usage.input_tokens),
                    int(usage.cache_creation_input_tokens),
                    int(usage.cache_read_input_tokens),
                    int(usage.output_tokens),
                    float(cost_usd),
                    bool(batch),
                    created_at,
                ],
            )
            if answer_json is not None:
                self._con.execute(_INSERT_ANSWER, [*identity, answer_json, created_at])
        except BaseException:
            self._con.rollback()
            raise
        self._con.commit()

    def spent_since(self, since: datetime) -> float:
        row = self._con.execute(
            "SELECT coalesce(sum(cost_usd), 0) FROM calls WHERE created_at >= ?",
            [_utc(since, "since")],
        ).fetchone()
        return float(row[0]) if row else 0.0

    def status_counts(self, scenario_hash: str) -> dict[str, int]:
        rows = self._con.execute(
            "SELECT status, count(*) FROM calls WHERE scenario_hash = ? "
            "GROUP BY status ORDER BY status",
            [scenario_hash],
        ).fetchall()
        return Counter({status: int(count) for status, count in rows})

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _check_call(
    call_id: str, status: str, cost_usd: float, reason: str, answer_json: str | None
) -> None:
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("call_id must not be blank")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}, got {status!r}")
    if not (math.isfinite(cost_usd) and cost_usd >= 0):
        raise ValueError(f"cost_usd must be finite and non-negative, got {cost_usd}")
    if not isinstance(reason, str):
        raise ValueError(f"reason must be a string, got {reason!r}")
    if status == "valid":
        if answer_json is None:
            raise ValueError("a valid call needs its answer_json")
        _check_json_object(answer_json)
    elif answer_json is not None:
        raise ValueError(f"a {status} call must not carry an answer_json")


def _check_json_object(answer_json: str) -> None:
    try:
        data = json.loads(answer_json)
    except (TypeError, ValueError) as error:
        raise ValueError(f"answer_json is not JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("answer_json is not a JSON object")


def _utc(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {moment.isoformat()}")
    return moment.astimezone(UTC)
