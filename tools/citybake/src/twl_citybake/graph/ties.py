from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from twl_citybake.osm.banks import EAST, WEST
from twl_citybake.osm.tags import BuildingKind
from twl_citybake.personas.assign import adults_per_household
from twl_citybake.population.capacity import storeys

TIE_KINDS = ("household", "neighbour", "friend", "work")
HOUSEHOLD, NEIGHBOUR, FRIEND, WORK = 0, 1, 2, 3
SPOUSE_STRENGTH = 1.0
HOUSEMATE_STRENGTH = 0.8
NEIGHBOUR_STRENGTH_MAX = 0.5
NEIGHBOUR_STRENGTH_MIN = 0.1
NEIGHBOUR_STRENGTH_SCALE_M = 500.0
FRIEND_STRENGTH = (0.4, 0.8)
WORK_STRENGTH = (0.2, 0.5)
TAGGED_WORKPLACE_KINDS = (BuildingKind.COMMERCIAL, BuildingKind.INDUSTRIAL, BuildingKind.CIVIC)

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


def work_ties(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    household: NDArray[np.uint32],
    occupied: NDArray[np.bool_],
    buildings: Mapping[str, np.ndarray],
    degree: float,
    scale_m: float,
    cell_m: float,
    candidates: int,
    min_area: float,
    max_floors: int,
    rng: np.random.Generator,
    district_names: Sequence[str] = (),
) -> tuple[Ties, dict[str, Any]]:
    """Co-workers share a workplace cell, so a tie can join homes far apart. `buildings` holds x, y,
    area_m2, kind, bank, district and floors or height_m. Draws, in order: `candidates` cells per
    worker, the pick among them, Poisson(degree / 2) per worker in cell order, then per draw the
    co-worker and the strength."""
    tagged, untagged = workplace_masks(buildings, min_area, max_floors)
    is_workplace = tagged | untagged
    workers = np.flatnonzero(occupied)
    floor_area = _floor_area(buildings)[is_workplace]
    cells = _workplace_cells(buildings, is_workplace, floor_area, cell_m)
    ties, chosen = _no_ties(), np.zeros(0, np.intp)
    if len(workers) and cells.weight.sum() > 0:
        chosen = _choose_workplaces(x[workers], y[workers], cells, scale_m, candidates, rng)
        ties = _coworker_ties(workers, chosen, household, degree, rng)
    return ties, _workplace_summary(buildings, tagged, untagged, cells.bank, chosen, district_names)


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


def workplace_masks(
    buildings: Mapping[str, np.ndarray], min_area: float, max_floors: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(tagged, untagged): OSM says what the first are. The second are big, low footprints without
    a useful tag, which is what sheds and factories look like in the industrial east."""
    kind, area = buildings["kind"], buildings["area_m2"]
    tagged = np.isin(kind, [int(k) for k in TAGGED_WORKPLACE_KINDS])
    low = _floors(buildings) <= max_floors
    untagged = (kind == BuildingKind.UNKNOWN) & (area >= min_area) & low
    return tagged, untagged


@dataclass(frozen=True)
class _Cells:
    x: Floats
    y: Floats
    weight: Floats
    bank: NDArray[np.uint8]


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


def _floors(buildings: Mapping[str, np.ndarray]) -> NDArray[np.int64]:
    if "floors" in buildings:
        return np.asarray(buildings["floors"], np.int64)
    return storeys(buildings["height_m"])


def _floor_area(buildings: Mapping[str, np.ndarray]) -> Floats:
    return np.asarray(buildings["area_m2"], np.float64) * _floors(buildings)


def _workplace_cells(
    buildings: Mapping[str, np.ndarray],
    is_workplace: NDArray[np.bool_],
    floor_area: Floats,
    cell_m: float,
) -> _Cells:
    """Grid cells holding workplaces, each at its floor-area centroid and weighted by floor area.
    A cell across the river belongs to the bank with more of its floor area."""
    x = np.asarray(buildings["x"], np.float64)[is_workplace]
    y = np.asarray(buildings["y"], np.float64)[is_workplace]
    if len(x) == 0:
        empty = np.zeros(0, np.float64)
        return _Cells(empty, empty, empty, np.zeros(0, np.uint8))
    col, row = np.floor(x / cell_m).astype(np.int64), np.floor(y / cell_m).astype(np.int64)
    key = (col - col.min()) * (int(row.max() - row.min()) + 1) + (row - row.min())
    _, cell = np.unique(key, return_inverse=True)
    weight = np.bincount(cell, weights=floor_area)
    safe = np.where(weight > 0, weight, 1.0)
    east = buildings["bank"][is_workplace] == EAST
    east_area = np.bincount(cell[east], weights=floor_area[east], minlength=len(weight))
    return _Cells(
        x=np.bincount(cell, weights=floor_area * x) / safe,
        y=np.bincount(cell, weights=floor_area * y) / safe,
        weight=weight,
        bank=np.where(east_area > weight / 2, EAST, WEST).astype(np.uint8),
    )


def _choose_workplaces(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cells: _Cells,
    scale_m: float,
    candidates: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Per worker, `candidates` cells drawn by floor area, then one by distance from home."""
    cumulative = np.cumsum(cells.weight)
    uniform = rng.random((len(x), candidates)) * cumulative[-1]
    drawn = np.minimum(np.searchsorted(cumulative, uniform, side="right"), len(cumulative) - 1)
    home_x, home_y = np.asarray(x, np.float64)[:, None], np.asarray(y, np.float64)[:, None]
    distance = np.hypot(cells.x[drawn] - home_x, cells.y[drawn] - home_y)
    pick = _weighted_pick(_decay_weights(distance, np.ones_like(distance, bool), scale_m), rng)
    return drawn[np.arange(len(x)), pick]


def _coworker_ties(
    workers: NDArray[np.intp],
    workplace: NDArray[np.intp],
    household: NDArray[np.uint32],
    degree: float,
    rng: np.random.Generator,
) -> Ties:
    """Poisson(degree / 2) draws per worker, each to a uniform other worker in the same cell. A draw
    that lands in the worker's own household is dropped, not redrawn: it is rare in a real cell."""
    order = np.lexsort((workers, workplace))
    members = workers[order]
    _, starts, counts = np.unique(workplace[order], return_index=True, return_counts=True)
    position = np.repeat(np.arange(len(members)), rng.poisson(degree / 2, len(members)))
    cell = np.repeat(np.arange(len(counts)), counts)[position]
    size, start = counts[cell], starts[cell]
    step = 1 + (rng.random(len(position)) * np.maximum(size - 1, 1)).astype(np.intp)
    other = start + (position - start + step) % size
    strength = rng.uniform(*WORK_STRENGTH, len(position))
    a, b = members[position], members[other]
    keep = household[a] != household[b]
    return _unique_pairs(a[keep], b[keep], strength[keep])


def _workplace_summary(
    buildings: Mapping[str, np.ndarray],
    tagged: NDArray[np.bool_],
    untagged: NDArray[np.bool_],
    cell_bank: NDArray[np.uint8],
    chosen: NDArray[np.intp],
    district_names: Sequence[str],
) -> dict[str, Any]:
    """Counts of workplace buildings, and of workers by the bank they work on."""
    bank, district = buildings["bank"], buildings["district"]
    is_workplace = tagged | untagged
    labels = list(district_names) or [
        str(i) for i in range(int(district.max()) + 1 if len(district) else 0)
    ]
    tagged_by = np.bincount(district[tagged], minlength=len(labels))
    untagged_by = np.bincount(district[untagged], minlength=len(labels))
    workers = np.bincount(cell_bank[chosen], minlength=2)
    return {
        "count": int(is_workplace.sum()),
        "by_source": {"osm_tagged": int(tagged.sum()), "untagged_footprint": int(untagged.sum())},
        "by_bank": _by_bank(np.bincount(bank[is_workplace], minlength=2)),
        "by_district": {
            name: {"osm_tagged": int(tagged_by[i]), "untagged_footprint": int(untagged_by[i])}
            for i, name in enumerate(labels)
        },
        "workers_by_bank": _by_bank(workers),
    }


def _by_bank(counts: NDArray[np.integer]) -> dict[str, int]:
    return {"west": int(counts[WEST]), "east": int(counts[EAST])}
