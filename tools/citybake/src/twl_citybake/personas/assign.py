from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_citybake.personas.pool import FEMALE, MALE, PersonaPool

# The spouse column holds resident rows, so `household` must be the whole residents column in order
NO_SPOUSE = 0xFFFFFFFF
# Husband age minus wife age: about 94% in 0..8 and about 6% reversed (D-026).
AGE_GAPS = np.arange(-3, 9)
AGE_GAP_WEIGHTS = np.array([1, 2, 3, 6, 10, 14, 16, 15, 12, 9, 7, 5], np.float64)
_MAX_AGE = 255
_BISECTIONS = 100
_NO_UNPAIRED_DRAWS = 1e-9


@dataclass(frozen=True)
class Assignment:
    persona_ref: NDArray[np.uint32]
    spouse: NDArray[np.uint32]


def adults_per_household(household: NDArray[np.uint32]) -> NDArray[np.int64]:
    """Run lengths of the residents' household column: one contiguous run per household."""
    if len(household) == 0:
        return np.zeros(0, np.int64)
    edges = np.flatnonzero(household[1:] != household[:-1]) + 1
    return np.diff(np.concatenate(([0], edges, [len(household)]))).astype(np.int64)


def free_draw_weights(
    household_adults: NDArray[np.int64], pool: PersonaPool, married_code: int
) -> NDArray[np.float64]:
    """The probability of each pool row on a free draw; sums to 1.

    D-026 conditions second adults on marriage, which alone would inflate the married share (and
    shrink young_single from 14.3% to about 11%). So free draws use a lower married probability s,
    chosen so the residents' married share equals the pool's. A household of k adults expects c_k
    couples, with c_0 = c_1 = 0 and c_k(s) = s (1 + c_{k-2}) + (1 - s) c_{k-1}, because a married
    adult takes the next slot as a spouse. Over H_k households and N adults that is
    C(s) = sum_k H_k c_k(s) couples and T(s) = N - C(s) free draws. Married residents number
    s T + C, which rises from 0 at s = 0 to N at s = 1, so bisection solves s T + C = N p_married.

    The sex split of the married is kept the same way. Each couple is one woman and one man, opened
    by a woman with probability r, the female share of free married draws. Married women then
    number r s T + (1 - r) C = N p_mf, so r = (N p_mf - C) / (s T - C), clipped to [0, 1].

    Within a class weights are flat: married female s r / n_mf, married male s (1 - r) / n_mm,
    everyone else (1 - s) / n_other.
    """
    rows = len(pool.uuid)
    if rows == 0:
        raise ValueError("the persona pool is empty")
    married = pool.marital == married_code
    married_female = married & (pool.sex == FEMALE)
    married_male = married & (pool.sex == MALE)
    n_mf, n_mm = int(married_female.sum()), int(married_male.sum())
    households = np.bincount(household_adults, minlength=1)
    adults = float(np.dot(households, np.arange(len(households))))
    if n_mf and n_mm and adults:
        s = _solve_married_probability(households, adults, (n_mf + n_mm) / rows)
        couples = _expected_couples(households, s)
    else:
        s, couples = (n_mf + n_mm) / rows, 0.0
    r = _female_share(adults, s, couples, n_mf, n_mm, rows)
    weights = np.full(rows, (1 - s) / max(rows - n_mf - n_mm, 1))
    weights[married_female] = s * r / max(n_mf, 1)
    weights[married_male] = s * (1 - r) / max(n_mm, 1)
    return weights / weights.sum()


def draw_free_rows(
    weights: NDArray[np.float64], count: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """The sampler: P-061b replaces it to condition free draws on household affluence."""
    return rng.choice(len(weights), size=count, p=weights)


def assign_personas(
    household: NDArray[np.uint32],
    pool: PersonaPool,
    segments: NDArray[np.uint16],
    married_code: int,
    rng: np.random.Generator,
) -> Assignment:
    """Walk each household's slots in order: a free slot draws from the pool, and a married draw
    with a next slot takes it as a spouse, married and of the other sex. The walk continues after
    the spouse, so a household of 4 can hold two couples. `segments` labels the pool rows; it does
    not steer the draws yet.

    Draws, in order and one per slot whether used or not: the free row, the spouse's uniform, the
    age gap."""
    if len(segments) != len(pool.uuid):
        raise ValueError(f"{len(segments)} segments for {len(pool.uuid)} pool rows")
    slots = len(household)
    weights = free_draw_weights(adults_per_household(household), pool, married_code)
    free = draw_free_rows(weights, slots, rng)
    uniform = rng.random(slots)
    gap = rng.choice(AGE_GAPS, size=slots, p=AGE_GAP_WEIGHTS / AGE_GAP_WEIGHTS.sum())
    mate = _spouse_candidates(free, gap, uniform, pool, married_code)
    first = _couple_openers(household, mate >= 0)
    persona_ref = free.astype(np.uint32)
    persona_ref[first + 1] = mate[first]
    spouse = np.full(slots, NO_SPOUSE, np.uint32)
    spouse[first] = first + 1
    spouse[first + 1] = first
    return Assignment(persona_ref=persona_ref, spouse=spouse)


def _expected_couples(households: NDArray[np.int64], s: float) -> float:
    per_household = [0.0, 0.0]
    for k in range(2, len(households)):
        per_household.append(s * (1 + per_household[k - 2]) + (1 - s) * per_household[k - 1])
    return float(np.dot(households, per_household[: len(households)]))


def _solve_married_probability(
    households: NDArray[np.int64], adults: float, p_married: float
) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(_BISECTIONS):
        s = (lo + hi) / 2
        couples = _expected_couples(households, s)
        if s * (adults - couples) + couples < adults * p_married:
            lo = s
        else:
            hi = s
    return (lo + hi) / 2


def _female_share(
    adults: float, s: float, couples: float, n_mf: int, n_mm: int, rows: int
) -> float:
    unpaired_draws = s * (adults - couples) - couples
    if unpaired_draws <= _NO_UNPAIRED_DRAWS * max(adults, 1.0):
        return n_mf / max(n_mf + n_mm, 1)
    return float(np.clip((adults * n_mf / rows - couples) / unpaired_draws, 0.0, 1.0))


@dataclass(frozen=True)
class _AgeLookup:
    """Personas by age; each target age holds the range of `rows` at its nearest available age,
    ties to the lower."""

    rows: NDArray[np.intp]
    start: NDArray[np.intp]
    count: NDArray[np.intp]

    def pick(self, target: NDArray[np.int64], uniform: NDArray[np.float64]) -> NDArray[np.intp]:
        age = np.clip(target, 0, _MAX_AGE)
        return self.rows[self.start[age] + (uniform * self.count[age]).astype(np.intp)]


def _age_lookup(rows: NDArray[np.intp], age: NDArray[np.int64]) -> _AgeLookup:
    by_age = rows[np.argsort(age[rows], kind="stable")]
    ages, first, count = np.unique(age[by_age], return_index=True, return_counts=True)
    target = np.arange(_MAX_AGE + 1)
    above = np.minimum(np.searchsorted(ages, target), len(ages) - 1)
    below = np.maximum(above - 1, 0)
    nearest = np.where(target - ages[below] <= ages[above] - target, below, above)
    return _AgeLookup(rows=by_age, start=first[nearest], count=count[nearest])


def _spouse_candidates(
    free: NDArray[np.int64],
    gap: NDArray[np.int64],
    uniform: NDArray[np.float64],
    pool: PersonaPool,
    married_code: int,
) -> NDArray[np.int64]:
    """The spouse each slot's free draw would take, -1 when it is not married or no married
    persona of the other sex exists. The gap is husband minus wife, so a wife's husband targets
    age + gap and a husband's wife targets age - gap."""
    age = pool.age.astype(np.int64)
    married = pool.marital == married_code
    mate = np.full(len(free), -1, np.int64)
    for sex, other, sign in ((FEMALE, MALE, 1), (MALE, FEMALE, -1)):
        candidates = np.flatnonzero(married & (pool.sex == other))
        if len(candidates) == 0:
            continue
        slots = np.flatnonzero(married[free] & (pool.sex[free] == sex))
        target = age[free[slots]] + sign * gap[slots]
        mate[slots] = _age_lookup(candidates, age).pick(target, uniform[slots])
    return mate


def _couple_openers(household: NDArray[np.uint32], has_mate: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Slots whose free draw opens a couple. Spouses are never re-drawn or chained: the walk skips
    the slot a couple takes."""
    can_open = (has_mate[:-1] & (household[1:] == household[:-1])).tolist()
    openers: list[int] = []
    slot = 0
    while slot < len(can_open):
        if can_open[slot]:
            openers.append(slot)
            slot += 2
        else:
            slot += 1
    return np.asarray(openers, np.intp)
