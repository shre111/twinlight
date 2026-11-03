from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from twl_citybake.population.ipf import IpfResult, integerise, ipf

AC_MIN_PEOPLE = 1000
AC_REL_ERROR = 0.01


@dataclass(frozen=True)
class Households:
    """One row per household, sorted by (building, size)."""

    building: npt.NDArray[np.uint32]
    size: npt.NDArray[np.uint8]
    district: npt.NDArray[np.uint16]
    report: dict[str, Any]


@dataclass(frozen=True)
class _DistrictFit:
    buildings: npt.NDArray[np.intp]
    counts: npt.NDArray[np.int64]
    ipf: IpfResult


def synthesise_households(
    placed: npt.ArrayLike,
    marginal: npt.ArrayLike,
    capacity: npt.ArrayLike,
    district: npt.ArrayLike,
    n_districts: int,
    sizes: npt.ArrayLike,
    counts: npt.ArrayLike,
) -> Households:
    """Whole households per building, district by district: IPF of a building x size-class
    matrix of persons to the placed people (rows) and the census persons share of each size
    class (columns), then integerised within capacity."""
    people = np.asarray(placed, np.float64)
    cap = np.asarray(capacity, np.int64)
    of = np.asarray(district, np.intp)
    size = np.asarray(sizes, np.int64)
    census = np.asarray(counts, np.float64)
    persons_share = census * size / (census * size).sum()
    fits = [
        _fit_district(members, people[members], cap[members], size, persons_share)
        for d in range(n_districts)
        if len(members := np.flatnonzero((of == d) & (people > 0) & (cap >= size.min())))
    ]
    building, hh_size = _rows(fits, size)
    order = np.lexsort((hh_size, building))
    building, hh_size = building[order], hh_size[order]
    hh_district = of[building]
    return Households(
        building=building.astype(np.uint32),
        size=hh_size.astype(np.uint8),
        district=hh_district.astype(np.uint16),
        report=_report(
            hh_size,
            hh_district,
            np.asarray(marginal, np.float64),
            np.bincount(of, weights=people, minlength=n_districts),
            census,
            size,
            fits,
        ),
    )


def _fit_district(
    members: npt.NDArray[np.intp],
    people: npt.NDArray[np.float64],
    cap: npt.NDArray[np.int64],
    size: npt.NDArray[np.int64],
    persons_share: npt.NDArray[np.float64],
) -> _DistrictFit:
    fits_in = size[None, :] <= cap[:, None]
    share = np.where(size <= cap.max(), persons_share, 0.0)
    share /= share.sum()
    seed = people[:, None] * share[None, :] * fits_in
    result = ipf(seed, people, share * people.sum())
    expected = result.matrix / size
    col_totals = np.round(expected.sum(0)).astype(np.int64)
    _match_people(col_totals, size, share, round(float(people.sum())))
    return _DistrictFit(members, integerise(expected, size, cap, col_totals), result)


def _match_people(
    col_totals: npt.NDArray[np.int64],
    size: npt.NDArray[np.int64],
    share: npt.NDArray[np.float64],
    people: int,
) -> None:
    """Rounding each column can miss the district by up to half of every size: shed the
    smallest households while the total overshoots, then top up with singles."""
    for k in range(len(size)):
        over = int(col_totals @ size) - people
        if over <= 0:
            break
        col_totals[k] -= min(int(col_totals[k]), -(-over // int(size[k])))
    if size[0] == 1 and share[0] > 0:
        col_totals[0] += max(0, people - int(col_totals @ size))


def _rows(
    fits: list[_DistrictFit], size: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    buildings, sizes = [np.zeros(0, np.intp)], [np.zeros(0, np.int64)]
    for fit in fits:
        row, k = np.nonzero(fit.counts)
        repeats = fit.counts[row, k]
        buildings.append(np.repeat(fit.buildings[row], repeats))
        sizes.append(np.repeat(size[k], repeats))
    return np.concatenate(buildings), np.concatenate(sizes)


def _report(
    hh_size: npt.NDArray[np.int64],
    hh_district: npt.NDArray[np.intp],
    marginal: npt.NDArray[np.float64],
    placed: npt.NDArray[np.float64],
    census: npt.NDArray[np.float64],
    size: npt.NDArray[np.int64],
    fits: list[_DistrictFit],
) -> dict[str, Any]:
    n_districts = len(marginal)
    persons = np.bincount(hh_district, weights=hh_size, minlength=n_districts)
    rel = np.divide(
        np.abs(persons - marginal), marginal, out=np.zeros(n_districts), where=marginal > 0
    )
    big = marginal >= AC_MIN_PEOPLE
    by_size = np.bincount(np.searchsorted(size, hh_size), minlength=len(size))
    return {
        "districts": {
            "count": n_districts,
            "marginal_people": float(marginal.sum()),
            "synthesised_people": int(persons.sum()),
            "max_rel_error": float(rel.max(initial=0.0)),
            "max_rel_error_1000_plus": float(rel[big].max(initial=0.0)),
            "outside_1pct": np.flatnonzero(big & (rel > AC_REL_ERROR)).tolist(),
            "outside_1pct_under_1000": np.flatnonzero(
                ~big & (marginal > 0) & (rel > AC_REL_ERROR)
            ).tolist(),
            "max_abs_people_vs_placed": float(np.abs(persons - placed).max(initial=0.0)),
        },
        "households": {
            "count": len(hh_size),
            "people": int(hh_size.sum()),
            "mean_size": float(hh_size.mean()) if len(hh_size) else 0.0,
            "size_shares": _shares(size, by_size),
            "census_size_shares": _shares(size, census),
        },
        "ipf": {
            "districts": len(fits),
            "max_iterations": max((f.ipf.iterations for f in fits), default=0),
            "max_rel_error": max((f.ipf.max_rel_error for f in fits), default=0.0),
            "all_converged": all(f.ipf.converged for f in fits),
        },
    }


def _shares(size: npt.NDArray[np.int64], counts: npt.NDArray[np.number]) -> dict[str, float]:
    total = counts.sum()
    return {str(s): float(n / total) if total else 0.0 for s, n in zip(size, counts, strict=True)}
