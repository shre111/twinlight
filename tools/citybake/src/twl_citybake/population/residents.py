from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from twl_citybake.osm.tags import LEVEL_HEIGHT_M
from twl_citybake.population.capacity import storeys
from twl_citybake.population.households import Households

BOX_SHARE = 0.4
JITTER_M = 0.5


def expected_adults(
    sizes: npt.ArrayLike, counts: npt.ArrayLike, adult_share: float
) -> dict[int, float]:
    """Mean adults in a household of each census size: everyone in households of one or two,
    else two plus a share r of the rest, with r set so the census households hold `adult_share`
    adults overall (Ahmedabad: r is about 0.47)."""
    size = np.asarray(sizes, np.float64)
    n = np.asarray(counts, np.float64)
    beyond_two = n * np.maximum(size - 2, 0)
    wanted = adult_share * (n * size).sum() - (n * np.minimum(size, 2)).sum()
    r = float(np.clip(wanted / beyond_two.sum(), 0.0, 1.0)) if beyond_two.sum() else 0.0
    mean = np.where(size > 2, 2 + r * (size - 2), size)
    return {int(s): float(m) for s, m in zip(size, mean, strict=True)}


def adults(
    households: Households, adults_by_size: Mapping[int, float], rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Whole adults per household: the size's mean, its fraction rounded up with that
    probability."""
    lookup = np.zeros(int(max(adults_by_size)) + 1)
    for size, mean in adults_by_size.items():
        lookup[size] = mean
    expected = lookup[households.size.astype(np.intp)]
    whole = np.floor(expected)
    return (whole + (rng.random(len(expected)) < expected - whole)).astype(np.int64)


def sample_residents(
    households: Households,
    buildings: Mapping[str, np.ndarray],
    adults_by_size: Mapping[int, float],
    target: int,
    shopper_share: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Whole households in random order until their adults first reach `target`; every adult of
    a taken household is a resident, standing on one floor of the household's building. Each
    resident stands for an equal share of the city's shoppers. Draws, in order: adults per
    household, the permutation, the in-building offsets u and v, the floor, jitter x and y."""
    per_household = adults(households, adults_by_size, rng)
    order = rng.permutation(len(per_household))
    reached = np.cumsum(per_household[order])
    take = min(int(np.searchsorted(reached, target)) + 1, len(order)) if target > 0 else 0
    taken = np.sort(order[:take])
    b = households.building[taken].astype(np.intp)
    u = rng.uniform(-1.0, 1.0, len(taken))
    v = rng.uniform(-1.0, 1.0, len(taken))
    floor = rng.integers(0, storeys(buildings["height_m"][b]))
    members = per_household[taken]
    n = int(members.sum())
    jitter_x = rng.uniform(-JITTER_M, JITTER_M, n)
    jitter_y = rng.uniform(-JITTER_M, JITTER_M, n)

    along = BOX_SHARE * buildings["w"][b].astype(np.float64) * u
    across = BOX_SHARE * buildings["d"][b].astype(np.float64) * v
    rot = buildings["rot"][b].astype(np.float64)
    cos, sin = np.cos(rot), np.sin(rot)
    home_x = buildings["x"][b] + along * cos - across * sin
    home_y = buildings["y"][b] + along * sin + across * cos
    resident_b = np.repeat(b, members)
    weight = shopper_share * per_household.sum() / n if n else 0.0
    return {
        "household": np.repeat(taken, members).astype(np.uint32),
        "building": resident_b.astype(np.uint32),
        "district": np.repeat(households.district[taken], members).astype(np.uint16),
        "bank": np.asarray(buildings["bank"])[resident_b].astype(np.uint8),
        "x": (np.repeat(home_x, members) + jitter_x).astype(np.float32),
        "y": (np.repeat(home_y, members) + jitter_y).astype(np.float32),
        "z": np.repeat((floor + 0.5) * LEVEL_HEIGHT_M, members).astype(np.float32),
        "weight": np.full(n, weight, np.float32),
    }
