from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from twl_citybake.personas.assign import adults_per_household

TIE_KINDS = ("household", "neighbour", "friend", "work")
HOUSEHOLD, NEIGHBOUR, FRIEND, WORK = 0, 1, 2, 3
SPOUSE_STRENGTH = 1.0
HOUSEMATE_STRENGTH = 0.8
NEIGHBOUR_STRENGTH_MAX = 0.5
NEIGHBOUR_STRENGTH_MIN = 0.1
NEIGHBOUR_STRENGTH_SCALE_M = 500.0
FRIEND_STRENGTH = (0.4, 0.8)

type Ties = tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.float32]]
type Floats = NDArray[np.float64]


@dataclass(frozen=True)
class Graph:
    """Directed CSR: resident `i` talks to `targets[offsets[i]:offsets[i + 1]]`, ascending. An
    undirected relation appears in both directions with one kind and one strength."""

    offsets: NDArray[np.uint32]
    targets: NDArray[np.uint32]
    kind: NDArray[np.uint8]
    strength: NDArray[np.float32]


def household_ties(household: NDArray[np.uint32], spouse: NDArray[np.uint32]) -> Ties:
    """Every pair inside a household run. Only `spouse` says who is a couple: a married resident
    need not have a spouse at home."""
    if len(household) != len(spouse):
        raise ValueError(f"{len(household)} household rows but {len(spouse)} spouse rows")
    sizes = adults_per_household(household)
    first = np.cumsum(sizes) - sizes
    us: list[NDArray[np.int64]] = []
    vs: list[NDArray[np.int64]] = []
    for size in np.unique(sizes[sizes >= 2]).tolist():
        lower, upper = np.triu_indices(size, k=1)
        starts = first[sizes == size][:, None]
        us.append((starts + lower).ravel())
        vs.append((starts + upper).ravel())
    if not us:
        return _no_ties()
    u, v = np.concatenate(us), np.concatenate(vs)
    is_couple = spouse[u] == v
    return _unique_pairs(u, v, np.where(is_couple, SPOUSE_STRENGTH, HOUSEMATE_STRENGTH))


def neighbour_ties(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    household: NDArray[np.uint32],
    degree: float,
    scale_m: float,
    candidates: int,
    rng: np.random.Generator,
) -> Ties:
    """Draws, in order: Poisson(degree / 2) per resident, then one uniform per draw to choose among
    the nearest other-household residents. Each draw adds one tie to two residents."""
    n = len(x)
    draws = rng.poisson(degree / 2, n)
    source = np.repeat(np.arange(n), draws)
    if len(source) == 0:
        return _no_ties()
    distance, nearest, allowed = _nearest_other_households(x, y, household, candidates)
    d, others, ok = distance[source], nearest[source], allowed[source]
    pick = _weighted_pick(_decay_weights(d, ok, scale_m), rng)
    rows = np.arange(len(source))
    keep = ok[rows, pick]
    strength = np.clip(
        NEIGHBOUR_STRENGTH_MAX * np.exp(-d[rows, pick] / NEIGHBOUR_STRENGTH_SCALE_M),
        NEIGHBOUR_STRENGTH_MIN,
        NEIGHBOUR_STRENGTH_MAX,
    )
    return _unique_pairs(source[keep], others[rows, pick][keep], strength[keep])


def friend_ties(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    household: NDArray[np.uint32],
    segment: NDArray[np.integer],
    degree: float,
    scale_m: float,
    homophily: float,
    candidates: int,
    rng: np.random.Generator,
) -> Ties:
    """A tournament, so the whole city is reachable without all-pairs distances. Draws, in order:
    Poisson(degree / 2) per resident, then per draw the friend's segment, `candidates` residents of
    it, the pick among them and the strength."""
    n = len(x)
    source = np.repeat(np.arange(n), rng.poisson(degree / 2, n))
    if len(source) == 0:
        return _no_ties()
    xs, ys = np.asarray(x, np.float64), np.asarray(y, np.float64)
    _, code, counts = np.unique(segment, return_inverse=True, return_counts=True)
    wanted = _friend_segments(code[source], counts, homophily, rng)
    members = np.argsort(code, kind="stable")
    starts = np.cumsum(counts) - counts
    uniform = rng.random((len(source), candidates))
    offset = (uniform * counts[wanted][:, None]).astype(np.intp)
    others = members[starts[wanted][:, None] + offset]
    distance = np.hypot(xs[others] - xs[source][:, None], ys[others] - ys[source][:, None])
    allowed = household[others] != household[source][:, None]
    pick = _weighted_pick(_decay_weights(distance, allowed, scale_m), rng)
    strength = rng.uniform(*FRIEND_STRENGTH, len(source))
    rows = np.arange(len(source))
    keep = allowed[rows, pick]
    return _unique_pairs(source[keep], others[rows, pick][keep], strength[keep])


def merge_ties(per_kind: Sequence[Ties], n_residents: int) -> Graph:
    """`per_kind` follows TIE_KINDS. A pair under several kinds keeps the lowest kind id, the
    strongest relation, with that kind's strength."""
    if len(per_kind) != len(TIE_KINDS):
        raise ValueError(f"merge_ties needs {len(TIE_KINDS)} kinds, got {len(per_kind)}")
    a = np.concatenate([np.asarray(ties[0], np.int64) for ties in per_kind])
    b = np.concatenate([np.asarray(ties[1], np.int64) for ties in per_kind])
    strength = np.concatenate([np.asarray(ties[2], np.float32) for ties in per_kind])
    lengths = [len(ties[0]) for ties in per_kind]
    kind = np.repeat(np.arange(len(TIE_KINDS), dtype=np.uint8), lengths)
    _check_ties(a, b, strength, n_residents)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    # np.unique's return_index gives first occurrences, and the rows are concatenated by kind id
    _, first = np.unique((lo << 32) | hi, return_index=True)
    source = np.concatenate((lo[first], hi[first]))
    target = np.concatenate((hi[first], lo[first]))
    order = np.lexsort((target, source))
    offsets = np.zeros(n_residents + 1, np.uint32)
    offsets[1:] = np.cumsum(np.bincount(source, minlength=n_residents))
    return Graph(
        offsets=offsets,
        targets=target[order].astype(np.uint32),
        kind=np.concatenate((kind[first], kind[first]))[order],
        strength=np.concatenate((strength[first], strength[first]))[order],
    )


def _no_ties() -> Ties:
    return np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros(0, np.float32)


def _unique_pairs(a: NDArray[np.integer], b: NDArray[np.integer], strength: np.ndarray) -> Ties:
    """One row per pair in (u, v) order, u < v; a repeated pair keeps its first strength."""
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    _, first = np.unique((lo << 32) | hi, return_index=True)
    return (
        lo[first].astype(np.uint32),
        hi[first].astype(np.uint32),
        np.asarray(strength)[first].astype(np.float32),
    )


def _check_ties(
    a: NDArray[np.int64], b: NDArray[np.int64], strength: NDArray[np.float32], n: int
) -> None:
    if (a == b).any():
        raise ValueError("a resident cannot have a tie to themselves")
    if len(a) and (min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= n):
        raise ValueError(f"tie endpoint outside the {n} residents")
    if not ((strength > 0) & (strength <= 1)).all():
        raise ValueError("tie strengths must be in (0, 1]")


def _nearest_other_households(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    household: NDArray[np.uint32],
    candidates: int,
) -> tuple[Floats, NDArray[np.intp], NDArray[np.bool_]]:
    """Per resident, distances and rows of the nearest residents, and which of them count: other
    households only, the nearest `candidates`. Asking for one household more than `candidates`
    leaves `candidates` others even for a resident of the largest household."""
    n = len(x)
    largest = int(np.unique(household, return_counts=True)[1].max())
    k = min(candidates + largest, n)
    points = np.column_stack((np.asarray(x, np.float64), np.asarray(y, np.float64)))
    found_distance, found = KDTree(points).query(points, k=k)
    distance = np.asarray(found_distance, np.float64).reshape(n, k)
    nearest = np.asarray(found, np.intp).reshape(n, k)
    allowed = household[nearest] != household[:, None]
    allowed &= np.cumsum(allowed, axis=1) <= candidates
    return distance, nearest, allowed


def _decay_weights(distance: Floats, allowed: NDArray[np.bool_], scale_m: float) -> Floats:
    """exp(-d / scale) per row, up to a factor: measuring from the row's nearest allowed column
    keeps the proportions and stops a resident far from everything underflowing to all zeros."""
    masked = np.where(allowed, distance, np.inf)
    nearest = masked.min(axis=1, keepdims=True)
    nearest[~np.isfinite(nearest)] = 0.0
    return np.exp((nearest - masked) / scale_m)


def _weighted_pick(weights: Floats, rng: np.random.Generator) -> NDArray[np.intp]:
    """Per row, a column with probability proportional to its weight; callers drop rows whose pick
    has weight zero, which happens only when the whole row is zero."""
    cumulative = np.cumsum(weights, axis=1)
    threshold = rng.random(len(weights)) * cumulative[:, -1]
    columns = (cumulative <= threshold[:, None]).sum(axis=1)
    return np.minimum(columns, weights.shape[1] - 1)


def _friend_segments(
    own: NDArray[np.intp], counts: NDArray[np.intp], homophily: float, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Per draw, segment s with probability proportional to share(s) * (homophily if own else 1)."""
    odds = np.tile(counts.astype(np.float64), (len(counts), 1))
    np.fill_diagonal(odds, counts * homophily)
    cumulative = np.cumsum(odds, axis=1)
    cumulative /= cumulative[:, -1:]
    wanted = (cumulative[own] <= rng.random(len(own))[:, None]).sum(axis=1)
    return np.minimum(wanted, len(counts) - 1)
