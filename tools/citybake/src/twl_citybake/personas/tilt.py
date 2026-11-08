import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from twl_citybake.personas.pool import FEMALE, MALE, PersonaPool
from twl_citybake.population.tier import HIGH, MIDDLE, check_shares

TIERS = HIGH + 1
_BISECTIONS = 200
_LOG_ODDS_LIMIT = math.log(1e12)


def tier_education_shares(
    quintile_population: Sequence[float],
    by_quintile: Sequence[float],
    low_share: float,
    high_share: float,
) -> tuple[float, float, float]:
    """Each tier's share with 10+ years of schooling, as a fraction. The quintiles lie end to end on
    [0, 1] by population share, lowest first; the low tier covers [0, low_share), the middle
    [low_share, 1 - high_share) and the high [1 - high_share, 1], and each tier takes the
    overlap-weighted mean of `by_quintile` (percent) over its interval."""
    population = np.asarray(quintile_population, np.float64)
    values = np.asarray(by_quintile, np.float64) / 100
    if population.ndim != 1 or len(population) == 0 or population.shape != values.shape:
        raise ValueError("quintile_population and by_quintile need one value per quintile")
    if (population < 0).any() or not population.sum() > 0:
        raise ValueError("quintile_population must be non-negative with a positive total")
    check_shares(low_share, high_share)
    edges = np.concatenate(([0.0], np.cumsum(population / population.sum())))
    shares = []
    for start, end in pairwise((0.0, low_share, 1 - high_share, 1.0)):
        overlap = np.clip(np.minimum(edges[1:], end) - np.maximum(edges[:-1], start), 0.0, None)
        shares.append(float(np.dot(overlap, values) / overlap.sum()))
    low, middle, high = shares
    return low, middle, high


def odds_ratios(shares: Sequence[float]) -> tuple[float, float, float]:
    """The odds of each tier's share over the middle tier's odds, so the middle is 1."""
    if len(shares) != TIERS or not all(0 < share < 1 for share in shares):
        raise ValueError(f"odds_ratios needs {TIERS} shares strictly inside (0, 1), got {shares!r}")
    odds = [share / (1 - share) for share in shares]
    low, middle, high = (value / odds[MIDDLE] for value in odds)
    return low, middle, high


def tilted_weights(
    weights: NDArray[np.float64],
    pool: PersonaPool,
    married_code: int,
    ten_plus: NDArray[np.bool_],
    tier_share: ArrayLike,
    ratios_female: Sequence[float],
    ratios_male: Sequence[float],
) -> NDArray[np.float64]:
    """Free-draw weights per tier, shape (3, rows), from the flat-within-class `weights`.

    Draw groups are married and other, by sex. A group of mass M with 10+ share P keeps M in every
    tier, and its 10+ share in tier t is q_t = r_t b / (1 + r_t b), so the odds of 10+ in tier t
    over the middle tier are r_t. The base odds b solve sum_t tier_share_t q_t = P, so over the
    slots the group's 10+ share stays P and district marginals hold. 10+ rows are scaled by
    q_t / P and the others by (1 - q_t) / (1 - P). A group with P of 0 or 1, or with every ratio
    equal, is left as it is."""
    rows = len(pool.uuid)
    base = np.asarray(weights, np.float64)
    educated = np.asarray(ten_plus, np.bool_)
    if base.shape != (rows,) or educated.shape != (rows,):
        raise ValueError(f"weights and ten_plus need one entry per pool row ({rows})")
    share = np.asarray(tier_share, np.float64)
    if share.shape != (TIERS,) or (share < 0).any() or not abs(share.sum() - 1) <= 1e-9:
        raise ValueError(f"tier_share must be {TIERS} non-negative shares summing to 1")
    by_sex = ((FEMALE, _ratios(ratios_female)), (MALE, _ratios(ratios_male)))
    tilted = np.tile(base, (TIERS, 1))
    married = pool.marital == married_code
    for is_married in (True, False):
        for sex, r in by_sex:
            group = (married == is_married) & (pool.sex == sex)
            mass = float(base[group].sum())
            if not mass > 0 or (r == r[0]).all():
                continue
            p = float(base[group & educated].sum()) / mass
            if not 0 < p < 1:
                continue
            q = _tier_probabilities(p, r, share)
            yes, no = group & educated, group & ~educated
            for tier in range(TIERS):
                tilted[tier, yes] = base[yes] * (q[tier] / p)
                tilted[tier, no] = base[no] * ((1 - q[tier]) / (1 - p))
    return tilted


def _ratios(ratios: Sequence[float]) -> NDArray[np.float64]:
    r = np.asarray(ratios, np.float64)
    if r.shape != (TIERS,) or not (np.isfinite(r) & (r > 0)).all():
        raise ValueError(f"odds ratios must be {TIERS} positive numbers, got {ratios!r}")
    return r


def _tier_probabilities(
    p: float, ratios: NDArray[np.float64], share: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Bisection on log b: the slot-weighted mean of q_t rises with b."""
    lo, hi = -_LOG_ODDS_LIMIT, _LOG_ODDS_LIMIT
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        if float(np.dot(share, _probabilities(mid, ratios))) < p:
            lo = mid
        else:
            hi = mid
    return _probabilities((lo + hi) / 2, ratios)


def _probabilities(log_base_odds: float, ratios: NDArray[np.float64]) -> NDArray[np.float64]:
    odds = ratios * math.exp(log_base_odds)
    return odds / (1 + odds)
