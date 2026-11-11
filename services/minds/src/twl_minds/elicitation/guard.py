import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from twl_minds.elicitation.store import utc_now

Cap = Literal["run", "daily"]


class SpendLog(Protocol):
    def spent_since(self, since: datetime) -> float: ...


@dataclass(frozen=True)
class Budget:
    estimate_usd: float
    spent_today_usd: float
    run_cap_usd: float
    daily_cap_usd: float


class CostCapExceeded(ValueError):
    def __init__(self, cap: Cap, budget: Budget) -> None:
        if cap == "run":
            message = (
                f"estimate {_usd(budget.estimate_usd)} is over the run cap "
                f"{_usd(budget.run_cap_usd)}"
            )
        else:
            message = (
                f"spent today {_usd(budget.spent_today_usd)} plus estimate "
                f"{_usd(budget.estimate_usd)} is over the daily cap {_usd(budget.daily_cap_usd)}"
            )
        super().__init__(message)
        self.cap: Cap = cap
        self.budget = budget


class CostGuard:
    def __init__(
        self,
        run_cap_usd: float,
        daily_cap_usd: float,
        store: SpendLog,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        for name, cap in (("run_cap_usd", run_cap_usd), ("daily_cap_usd", daily_cap_usd)):
            if not (math.isfinite(cap) and cap > 0):
                raise ValueError(f"{name} must be finite and positive, got {cap}")
        if run_cap_usd > daily_cap_usd:
            raise ValueError(
                f"run_cap_usd {run_cap_usd} must not be above daily_cap_usd {daily_cap_usd}"
            )
        self.run_cap_usd = run_cap_usd
        self.daily_cap_usd = daily_cap_usd
        self._store = store
        self._clock = clock

    def check(self, estimate_usd: float) -> Budget:
        """Call once per run, before anything is submitted, with the run's whole upper bound
        (`pricing.upper_bound_usd`). Spend counts settled calls only, so batches still in flight
        are invisible here until the runner reserves them."""
        if not (math.isfinite(estimate_usd) and estimate_usd >= 0):
            raise ValueError(f"estimate_usd must be finite and non-negative, got {estimate_usd}")
        budget = Budget(
            estimate_usd=estimate_usd,
            spent_today_usd=self._store.spent_since(self._start_of_day()),
            run_cap_usd=self.run_cap_usd,
            daily_cap_usd=self.daily_cap_usd,
        )
        if estimate_usd > self.run_cap_usd:
            raise CostCapExceeded("run", budget)
        if budget.spent_today_usd + estimate_usd > self.daily_cap_usd:
            raise CostCapExceeded("daily", budget)
        return budget

    def _start_of_day(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(f"clock() must be timezone-aware, got naive {now.isoformat()}")
        return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _usd(amount: float) -> str:
    return f"${amount:.4f}"
