"""Each tick's local hour, day and day type (D-036, MODEL.md section 11)."""

import functools
import importlib.resources
import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from twl_engine.clock import Clock
from twl_engine.scenario import MINUTES_PER_DAY

WORKING_DAY = 0
WEEKEND = 1
DAY_TYPES = 2


@dataclass(frozen=True, eq=False)
class Calendar:
    """Per tick, at its start minute: the local hour (0 to 23), the local day counted from day 0
    and whether that day is a weekend day.

    Sim minute 0 is local midnight of the scenario's `start_date` in the city's time zone. Without
    a `start_date`, minute 0 is midnight, days are 1,440 minutes and every day is a working day.
    """

    hour: NDArray[np.int64]
    day: NDArray[np.int64]
    weekend: NDArray[np.bool_]

    @property
    def day_type(self) -> NDArray[np.int64]:
        """`WORKING_DAY` or `WEEKEND` per tick."""
        return self.weekend.astype(np.int64)

    def day_type_at(self, tick: int) -> int:
        return WEEKEND if self.weekend[tick] else WORKING_DAY

    def is_new_day(self, tick: int) -> bool:
        return tick == 0 or bool(self.day[tick] != self.day[tick - 1])


def build_calendar(
    clock: Clock, start_date: str | None, timezone: str, weekend_days: Sequence[int]
) -> Calendar:
    """The run's calendar; the zone and weekend days are checked even without a `start_date`."""
    zone = load_zone(timezone)
    weekend_days = require_weekend_days(weekend_days)
    if start_date is None:
        return plain_calendar(clock)
    first = date.fromisoformat(start_date)
    origin = datetime.combine(first, time(0), tzinfo=zone).astimezone(UTC)
    local = [(origin + timedelta(minutes=int(m))).astimezone(zone) for m in _tick_starts(clock)]
    return Calendar(
        hour=np.array([t.hour for t in local], dtype=np.int64),
        day=np.array([(t.date() - first).days for t in local], dtype=np.int64),
        weekend=np.array([t.isoweekday() in weekend_days for t in local], dtype=np.bool_),
    )


def plain_calendar(clock: Clock) -> Calendar:
    starts = _tick_starts(clock)
    return Calendar(
        hour=(starts % MINUTES_PER_DAY) // 60,
        day=starts // MINUTES_PER_DAY,
        weekend=np.zeros(clock.ticks, dtype=np.bool_),
    )


def _tick_starts(clock: Clock) -> NDArray[np.int64]:
    return np.arange(clock.ticks, dtype=np.int64) * clock.tick_minutes


@functools.cache
def load_zone(name: str) -> ZoneInfo:
    """The IANA zone `name` from the pinned `tzdata` package, never the machine's database."""
    if name not in _zone_names():
        raise ValueError(f"unknown time zone {name!r}: not an IANA zone name in tzdata")
    resource = importlib.resources.files("tzdata").joinpath("zoneinfo", *name.split("/"))
    with resource.open("rb") as file:
        return ZoneInfo.from_file(file, key=name)


@functools.cache
def _zone_names() -> tuple[str, ...]:
    return tuple(importlib.resources.files("tzdata").joinpath("zones").read_text().split())


def require_weekend_days(days: Sequence[int]) -> tuple[int, ...]:
    """ISO weekdays, 1 Monday to 7 Sunday, ascending and distinct."""
    if any(isinstance(d, bool) for d in days):
        raise TypeError(f"weekend_days must be integers, got {tuple(days)!r}")
    days = tuple(operator.index(d) for d in days)
    if not all(1 <= d <= 7 for d in days):
        raise ValueError(f"weekend_days must be ISO weekdays 1 (Monday) to 7, got {days!r}")
    if any(a >= b for a, b in itertools.pairwise(days)):
        raise ValueError(f"weekend_days must be ascending and distinct, got {days!r}")
    return days
