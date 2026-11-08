from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import floor

import numpy as np
from numpy.typing import ArrayLike, NDArray

from twl_pack import NO_PERSONA

TIERS = 3
MAX_SEGMENTS = 256

Ints = NDArray[np.int64]
Floats = NDArray[np.float64]


@dataclass(frozen=True)
class SampleConfig:
    size: int = 300
    min_per_segment: int = 30
    by_tier: bool = False
    seed: int = 0


@dataclass(frozen=True)
class Stratum:
    segment: int
    tier: int | None
    residents: int
    weight: float
    candidates: int
    sampled: int


@dataclass(frozen=True, eq=False)
class PersonaSample:
    resident: NDArray[np.uint32]
    persona_ref: NDArray[np.uint32]
    segment: NDArray[np.uint8]
    tier: NDArray[np.uint8] | None
    stratum: NDArray[np.uint16]
    design_weight: Floats
    strata: tuple[Stratum, ...]

    def __len__(self) -> int:
        return len(self.resident)


def sample_personas(
    segment: ArrayLike,
    persona_ref: ArrayLike,
    weight: ArrayLike,
    tier: ArrayLike | None,
    segments: int,
    config: SampleConfig,
    entropy: Sequence[int] = (),
) -> PersonaSample:
    """A persona enters a stratum through its lowest-priority resident, so a persona that stands
    for more residents is likelier to be drawn: the sample follows the city, not the pool."""
    _check_config(segments, config, tier)
    seg = _integers("segment", segment, segments - 1)
    ref = _integers("persona_ref", persona_ref, NO_PERSONA)
    w = _weights("weight", weight)
    tr = None if tier is None else _integers("tier", tier, TIERS - 1)
    n = len(seg)
    if len(ref) != n or len(w) != n or (tr is not None and len(tr) != n):
        raise ValueError("segment, persona_ref, weight and tier need one entry per resident")

    rng = np.random.default_rng(np.random.SeedSequence([config.seed, *entropy]))
    priority = rng.random(n)
    eligible = (ref != NO_PERSONA) & (w > 0)
    reps = _representatives(ref, priority, eligible)
    per_segment = _segment_counts(
        np.bincount(seg[reps], minlength=segments),
        np.bincount(seg[eligible], weights=w[eligible], minlength=segments),
        config,
    )

    by_tier = config.by_tier and tr is not None
    cell = seg * TIERS + tr if config.by_tier and tr is not None else seg
    cells = segments * TIERS if by_tier else segments
    residents = np.bincount(cell[eligible], minlength=cells)
    mass = np.bincount(cell[eligible], weights=w[eligible], minlength=cells)
    candidates = np.bincount(cell[reps], minlength=cells)
    sampled = _split_tiers(per_segment, mass, candidates) if by_tier else per_segment

    chosen = _select(reps, cell[reps], priority, candidates, sampled)
    stratum = cell[chosen]
    return PersonaSample(
        resident=chosen.astype(np.uint32),
        persona_ref=ref[chosen].astype(np.uint32),
        segment=seg[chosen].astype(np.uint8),
        tier=None if tr is None else tr[chosen].astype(np.uint8),
        stratum=stratum.astype(np.uint16),
        design_weight=mass[stratum] / sampled[stratum],
        strata=_strata(residents, mass, candidates, sampled, by_tier),
    )


def apportion(total: int, weights: ArrayLike, caps: ArrayLike) -> Ints:
    """Capped largest remainder. Each round gives every open entry (count below cap) the floor of
    its share of what is left, hands the leftover by largest fractional part (ties to the lower
    index), clips at the caps and repeats until nothing is left or nothing is open. Shares are
    exact rationals of the float weights, so float rounding never decides a tie. Open entries
    whose weights all sum to zero share equally."""
    exact = [Fraction(x) for x in _weights("weights", weights).tolist()]
    cap = _integers("caps", caps, None).tolist()
    if len(cap) != len(exact):
        raise ValueError("weights and caps need the same length")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    counts = [0] * len(cap)
    left = int(total)
    while left > 0:
        open_ = [i for i, count in enumerate(counts) if count < cap[i]]
        if not open_:
            break
        mass = sum((exact[i] for i in open_), Fraction(0))
        share = [left * exact[i] / mass if mass else Fraction(left, len(open_)) for i in open_]
        grant = [floor(s) for s in share]
        spare = left - sum(grant)
        by_remainder = sorted((g - s, j) for j, (g, s) in enumerate(zip(grant, share, strict=True)))
        for _, j in by_remainder[:spare]:
            grant[j] += 1
        for i, g in zip(open_, grant, strict=True):
            given = min(g, cap[i] - counts[i])
            counts[i] += given
            left -= given
    return np.array(counts, dtype=np.int64)


def _check_config(segments: int, config: SampleConfig, tier: ArrayLike | None) -> None:
    if not 1 <= segments <= MAX_SEGMENTS:
        raise ValueError(f"segments must be in 1..{MAX_SEGMENTS}, got {segments}")
    if config.size < 1:
        raise ValueError(f"size must be at least 1, got {config.size}")
    if config.min_per_segment < 0:
        raise ValueError(f"min_per_segment must be non-negative, got {config.min_per_segment}")
    if config.by_tier and tier is None:
        raise ValueError("by_tier needs a tier per resident")


def _integers(name: str, values: ArrayLike, high: int | None) -> Ints:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        return np.zeros(0, dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must hold integers, got {array.dtype}")
    if int(array.min()) < 0 or (high is not None and int(array.max()) > high):
        raise ValueError(f"{name} must be in 0..{high if high is not None else 'any'}")
    return array.astype(np.int64)


def _weights(name: str, values: ArrayLike) -> Floats:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not (np.isfinite(array).all() and (array >= 0).all()):
        raise ValueError(f"{name} must be finite and non-negative")
    return array


def _representatives(ref: Ints, priority: Floats, eligible: NDArray[np.bool_]) -> Ints:
    pool = np.flatnonzero(eligible)
    ranked = pool[np.lexsort((pool, priority[pool], ref[pool]))]
    first = np.ones(len(ranked), dtype=bool)
    first[1:] = ref[ranked[1:]] != ref[ranked[:-1]]
    return np.sort(ranked[first])


def _segment_counts(candidates: Ints, weight: Floats, config: SampleConfig) -> Ints:
    floors = np.minimum(candidates, config.min_per_segment)
    target = min(config.size, int(candidates.sum()))
    floor_total = int(floors.sum())
    if floor_total > target:
        raise ValueError(
            f"size {config.size} is below the {floor_total} personas the segment floors need"
        )
    return floors + apportion(target - floor_total, weight, candidates - floors)


def _split_tiers(per_segment: Ints, weight: Floats, candidates: Ints) -> Ints:
    sampled = np.zeros(len(candidates), dtype=np.int64)
    for s, count in enumerate(per_segment.tolist()):
        cells = slice(s * TIERS, (s + 1) * TIERS)
        sampled[cells] = apportion(count, weight[cells], candidates[cells])
    return sampled


def _select(reps: Ints, rep_cell: Ints, priority: Floats, candidates: Ints, sampled: Ints) -> Ints:
    order = np.lexsort((reps, priority[reps], rep_cell))
    ranked_cell = rep_cell[order]
    rank = np.arange(len(order)) - (np.cumsum(candidates) - candidates)[ranked_cell]
    return np.sort(reps[order][rank < sampled[ranked_cell]])


def _strata(
    residents: Ints, mass: Floats, candidates: Ints, sampled: Ints, by_tier: bool
) -> tuple[Stratum, ...]:
    return tuple(
        Stratum(
            segment=k // TIERS if by_tier else k,
            tier=k % TIERS if by_tier else None,
            residents=int(residents[k]),
            weight=float(mass[k]),
            candidates=int(candidates[k]),
            sampled=int(sampled[k]),
        )
        for k in range(len(candidates))
    )
