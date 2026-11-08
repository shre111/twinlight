from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_citybake.personas.pool import FEMALE, MALE, PersonaPool
from twl_citybake.personas.tilt import TIERS

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
    weights: NDArray[np.float64],
    slot_tier: NDArray[np.integer] | None,
    rng: np.random.Generator,
    count: int | None = None,
) -> NDArray[np.int64]:
    """One uniform per slot, mapped through the inverse CDF of its tier's row of `weights`, (rows,)
    or (tiers, rows). This is how `rng.choice(rows, size=count, p=weights)` samples, so 1-D
    weights give exactly its draws and leave the generator in the same state. `count` is the
    number of slots when `slot_tier` is None."""
    table = np.asarray(weights, np.float64)
    tiers = None if slot_tier is None else np.asarray(slot_tier)
    if tiers is not None:
        slots = len(tiers)
        if count is not None and count != slots:
            raise ValueError(f"count {count} differs from {slots} slot tiers")
    elif count is not None:
        slots = count
    else:
        raise ValueError("draw_free_rows needs slot_tier or count")
    _check_weights(table, tiers)
    uniform = rng.random(slots)
    if table.ndim == 1 or tiers is None:
        return _inverse_cdf(table, uniform)
    rows = np.empty(slots, np.int64)
    for tier, tier_weights in enumerate(table):
        at = tiers == tier
        rows[at] = _inverse_cdf(tier_weights, uniform[at])
    return rows


def _check_weights(table: NDArray[np.float64], tiers: NDArray[np.integer] | None) -> None:
    if table.ndim not in (1, 2) or table.shape[-1] == 0:
        raise ValueError(f"free-draw weights must be (rows,) or (tiers, rows), got {table.shape}")
    if not (np.isfinite(table).all() and (table >= 0).all() and (table.sum(axis=-1) > 0).all()):
        raise ValueError("free-draw weights must be finite, non-negative and not all 0")
    if table.ndim == 2:
        if tiers is None:
            raise ValueError("weights per tier need slot_tier")
        if len(tiers) and not 0 <= int(tiers.min()) <= int(tiers.max()) < len(table):
            raise ValueError(f"slot_tier must be in 0..{len(table) - 1}")


def _inverse_cdf(weights: NDArray[np.float64], uniform: NDArray[np.float64]) -> NDArray[np.int64]:
    cdf = weights.cumsum()
    cdf /= cdf[-1]
    return cdf.searchsorted(uniform, side="right").astype(np.int64, copy=False)


def assign_personas(
    household: NDArray[np.uint32],
    pool: PersonaPool,
    segments: NDArray[np.uint16],
    married_code: int,
    rng: np.random.Generator,
    slot_tier: NDArray[np.integer] | None = None,
    free_weights: NDArray[np.float64] | None = None,
    spouse_weights: NDArray[np.float64] | None = None,
) -> Assignment:
    """Walk each household's slots in order: a free slot draws from the pool, and a married draw
    with a next slot takes it as a spouse, married and of the other sex. The walk continues after
    the spouse, so a household of 4 can hold two couples. `segments` labels the pool rows; it does
    not steer the draws yet.

    `free_weights` replaces `free_draw_weights`: one row, or one per tier with `slot_tier` giving
    each slot's tier (D-035). `spouse_weights`, (tiers, rows), tilts spouses with the same per-tier
    weights within their age bucket: the spouse is still taken at the nearest available age, then
    in proportion to the opening slot's tier row of weights. Without it, age alone decides.

    Draws, in order and one per slot whether used or not: the free row, the spouse's uniform, the
    age gap."""
    if len(segments) != len(pool.uuid):
        raise ValueError(f"{len(segments)} segments for {len(pool.uuid)} pool rows")
    slots = len(household)
    weights = (
        free_draw_weights(adults_per_household(household), pool, married_code)
        if free_weights is None
        else np.asarray(free_weights, np.float64)
    )
    if weights.shape[-1] != len(pool.uuid):
        raise ValueError(f"free_weights cover {weights.shape[-1]} rows, the pool {len(pool.uuid)}")
    if weights.ndim == 2 and slot_tier is None:
        raise ValueError("free_weights per tier need slot_tier")
    if slot_tier is not None and len(slot_tier) != slots:
        raise ValueError(f"{len(slot_tier)} slot tiers for {slots} slots")
    spouse_tilt = (
        None if spouse_weights is None else _spouse_tilt(spouse_weights, slot_tier, len(pool.uuid))
    )
    free = draw_free_rows(weights, slot_tier, rng, slots)
    uniform = rng.random(slots)
    gap = rng.choice(AGE_GAPS, size=slots, p=AGE_GAP_WEIGHTS / AGE_GAP_WEIGHTS.sum())
    mate = _spouse_candidates(free, gap, uniform, pool, married_code, spouse_tilt)
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

    def pick_weighted(
        self, target: NDArray[np.int64], uniform: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.intp]:
        """`pick` within the same bucket, each row in proportion to `weights` (one per pool row):
        the inverse CDF over the bucket in `rows` order, side="right". A bucket of zero weight
        falls back to `pick`."""
        age = np.clip(target, 0, _MAX_AGE)
        start = self.start[age]
        end = start + self.count[age]
        ordered = weights[self.rows]
        cumulative = np.concatenate(([0.0], ordered.cumsum()))
        low, high = cumulative[start], cumulative[end]
        at = cumulative.searchsorted(low + uniform * (high - low), side="right") - 1
        positions = np.arange(len(ordered))
        last_weighted = np.maximum.accumulate(np.where(ordered > 0, positions, -1))[end - 1]
        weighted = np.clip(at, start, np.maximum(last_weighted, start))
        return np.where(last_weighted >= start, self.rows[weighted], self.pick(target, uniform))


def _age_lookup(rows: NDArray[np.intp], age: NDArray[np.int64]) -> _AgeLookup:
    by_age = rows[np.argsort(age[rows], kind="stable")]
    ages, first, count = np.unique(age[by_age], return_index=True, return_counts=True)
    target = np.arange(_MAX_AGE + 1)
    above = np.minimum(np.searchsorted(ages, target), len(ages) - 1)
    below = np.maximum(above - 1, 0)
    nearest = np.where(target - ages[below] <= ages[above] - target, below, above)
    return _AgeLookup(rows=by_age, start=first[nearest], count=count[nearest])


@dataclass(frozen=True)
class _SpouseTilt:
    slot_tier: NDArray[np.integer]
    weights: NDArray[np.float64]


def _spouse_tilt(
    spouse_weights: NDArray[np.float64], slot_tier: NDArray[np.integer] | None, rows: int
) -> _SpouseTilt:
    if slot_tier is None:
        raise ValueError("spouse_weights need slot_tier")
    table = np.asarray(spouse_weights, np.float64)
    if table.shape != (TIERS, rows):
        raise ValueError(f"spouse_weights must be ({TIERS}, {rows}), got {table.shape}")
    if not (np.isfinite(table).all() and (table >= 0).all()):
        raise ValueError("spouse_weights must be finite and non-negative")
    tiers = np.asarray(slot_tier)
    if len(tiers) and not 0 <= int(tiers.min()) <= int(tiers.max()) < TIERS:
        raise ValueError(f"slot_tier must be in 0..{TIERS - 1}")
    return _SpouseTilt(slot_tier=tiers, weights=table)


def _spouse_candidates(
    free: NDArray[np.int64],
    gap: NDArray[np.int64],
    uniform: NDArray[np.float64],
    pool: PersonaPool,
    married_code: int,
    tilt: _SpouseTilt | None = None,
) -> NDArray[np.int64]:
    """The spouse each slot's free draw would take, -1 when it is not married or no married
    persona of the other sex exists. The gap is husband minus wife, so a wife's husband targets
    age + gap and a husband's wife targets age - gap. With a tilt the pick within the age bucket
    follows the slot's tier row of weights."""
    age = pool.age.astype(np.int64)
    married = pool.marital == married_code
    mate = np.full(len(free), -1, np.int64)
    for sex, other, sign in ((FEMALE, MALE, 1), (MALE, FEMALE, -1)):
        candidates = np.flatnonzero(married & (pool.sex == other))
        if len(candidates) == 0:
            continue
        slots = np.flatnonzero(married[free] & (pool.sex[free] == sex))
        target = age[free[slots]] + sign * gap[slots]
        lookup = _age_lookup(candidates, age)
        if tilt is None:
            mate[slots] = lookup.pick(target, uniform[slots])
            continue
        tiers = tilt.slot_tier[slots]
        for tier, weights in enumerate(tilt.weights):
            at = tiers == tier
            mate[slots[at]] = lookup.pick_weighted(target[at], uniform[slots[at]], weights)
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
