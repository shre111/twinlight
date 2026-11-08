from dataclasses import dataclass
from numbers import Real

import numpy as np
import numpy.typing as npt

LOW, MIDDLE, HIGH = 0, 1, 2


@dataclass(frozen=True)
class Tiers:
    """A tier per building; the cuts are floor area per person in m2."""

    tier: npt.NDArray[np.uint8]
    low_cut: float
    high_cut: float


def check_shares(low_share: object, high_share: object) -> None:
    if not _share("low_share", low_share) + _share("high_share", high_share) < 1:
        raise ValueError(
            f"tier low_share plus high_share must be below 1, got {low_share!r} + {high_share!r}"
        )


def _share(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 < float(value) < 1:
        raise ValueError(f"tier {name} must be a real number strictly inside (0, 1), got {value!r}")
    return float(value)


def building_tiers(
    area_m2: npt.ArrayLike,
    floors: npt.ArrayLike,
    persons: npt.ArrayLike,
    households: npt.ArrayLike,
    low_share: float,
    high_share: float,
) -> Tiers:
    """D-035: housed buildings ranked by floor area per person, ties by index, and cut where the
    running total of households first reaches `low_share` and `1 - high_share` of all households.
    Unhoused buildings are middle."""
    check_shares(low_share, high_share)
    area, levels, people, homes = (
        np.asarray(a, np.float64) for a in (area_m2, floors, persons, households)
    )
    if any(a.ndim != 1 or len(a) != len(area) for a in (area, levels, people, homes)):
        raise ValueError("area_m2, floors, persons and households need one entry per building")
    if not (np.isfinite(area).all() and np.isfinite(levels).all()):
        raise ValueError("building areas and floors must be finite")
    if (area < 0).any() or (levels < 0).any():
        raise ValueError("building areas and floors must be non-negative")
    for name, counts in (("persons", people), ("households", homes)):
        if not (np.isfinite(counts).all() and (counts >= 0).all() and (counts % 1 == 0).all()):
            raise ValueError(f"{name} must be whole non-negative counts")
    housed = homes > 0
    if ((people > 0) != housed).any():
        raise ValueError("a building has persons exactly when it has households")
    if not housed.any():
        raise ValueError("no building houses anyone")
    index = np.flatnonzero(housed)
    space = area[index] * levels[index] / people[index]
    order = np.argsort(space, kind="stable")
    cum = np.cumsum(homes[index][order])
    total = cum[-1]
    low_cut = float(space[order][np.argmax(cum >= low_share * total)])
    high_cut = float(space[order][np.argmax(cum >= (1 - high_share) * total)])
    tier = np.full(len(area), MIDDLE, np.uint8)
    tier[index] = np.where(space < low_cut, LOW, np.where(space >= high_cut, HIGH, MIDDLE))
    return Tiers(tier=tier, low_cut=low_cut, high_cut=high_cut)
