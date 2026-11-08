"""The city's time zone and weekend days (D-036), checked the same way by writer and reader."""

from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from numbers import Integral

from twl_pack.format import PackError

MONDAY, SUNDAY = 1, 7


@cache
def time_zones() -> frozenset[str]:
    """IANA names in the `tzdata` package, the zone files the engine loads (not the system's)."""
    listing = files("tzdata").joinpath("zones").read_text(encoding="utf-8")
    return frozenset(line.strip() for line in listing.splitlines() if line.strip())


def check_timezone(name: object) -> str:
    if not isinstance(name, str) or name not in time_zones():
        raise PackError(f"timezone must be an IANA name in tzdata, got {name!r}")
    return name


def check_weekend_days(days: object) -> tuple[int, ...]:
    """ISO weekdays, 1 Monday to 7 Sunday, strictly ascending; empty for a city with no weekend."""
    if isinstance(days, str | bytes) or not isinstance(days, Sequence):
        raise PackError(f"weekend_days must be a list of ISO weekdays, got {days!r}")
    previous = MONDAY - 1
    for day in days:
        if not isinstance(day, Integral) or isinstance(day, bool):
            raise PackError(f"weekend_days must hold whole ISO weekdays, got {day!r}")
        if not MONDAY <= day <= SUNDAY:
            raise PackError(f"weekend_days {day} is outside {MONDAY}..{SUNDAY}")
        if day <= previous:
            raise PackError(f"weekend_days must ascend strictly, got {list(days)!r}")
        previous = day
    return tuple(int(day) for day in days)
