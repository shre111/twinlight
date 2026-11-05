import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from twl_citybake.config import PopulationCorrection
from twl_citybake.population.raster import PopGrid

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

M_PER_DEGREE_LON_AT_EQUATOR = 111320.0
M_PER_DEGREE_LAT = 110574.0
FOLDS = 5
# about 2 km at 3 arc seconds
FOLD_BLOCK_CELLS = 22
FOLD_MIN_FLOOR_SHARE = 0.005
FOLD_MIN_PRESENT = 3


@dataclass(frozen=True)
class Correction:
    grid: PopGrid
    report: dict[str, Any]


@dataclass(frozen=True)
class _HousedCells:
    """One entry per raster cell with housing: raveled cell, raster people, housing floor, index
    into `class_ids`, and whether the cell is in the reference."""

    cell: IntArray
    people: FloatArray
    floor: FloatArray
    class_of: IntArray
    class_ids: IntArray
    in_reference: BoolArray


@dataclass(frozen=True)
class _Rates:
    """People per housing floor area per class: what the correction uses, and which were pooled to
    the reference rate or capped."""

    rate: FloatArray
    pooled: BoolArray
    capped: BoolArray
    reference_floor: float


def correct_raster(
    grid: PopGrid,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    footprint_m2: npt.ArrayLike,
    floors: npt.ArrayLike,
    housing_floor_m2: npt.ArrayLike,
    in_reference: npt.NDArray[np.bool_],
    config: PopulationCorrection,
) -> Correction:
    """Per cell raster^(1-w) * fit^w (D-030), rescaled to the raster's total over the whole window;
    fit is a class rate times the cell's housing floor. Cells without housing end at 0, so their
    people are redistributed by the rescale."""
    reference = np.asarray(in_reference, np.bool_)
    if reference.shape != grid.values.shape:
        raise ValueError(f"in_reference has shape {reference.shape}, not {grid.values.shape}")
    cells = _housed_cells(grid, lon, lat, footprint_m2, floors, housing_floor_m2, reference, config)
    rates = _fit_rates(cells, config)
    fit = rates.rate[cells.class_of] * cells.floor
    w = float(config.weight)
    blended = cells.people ** (1 - w) * fit**w
    total = float(grid.values.sum())
    values = np.zeros(grid.values.size)
    values[cells.cell] = blended * (total / float(blended.sum()))
    values = values.reshape(grid.values.shape)
    # Floating point leaves the rescaled sum a few ulps off; the largest cell absorbs them.
    values[np.unravel_index(np.argmax(values), values.shape)] += total - float(values.sum())
    report = {
        "weight": config.weight,
        "weight_range": list(config.weight_range),
        "weight_evidence": config.weight_evidence,
        "reference_polygons": list(config.reference_polygons),
        "reference_cells": int(cells.in_reference.sum()),
        "classes": _class_report(cells, rates, config),
        "fold_variation": _fold_variation(cells, rates, grid.values.shape[1]),
        "people_moved": float(np.abs(values - grid.values).sum()) / 2,
    }
    return Correction(replace(grid, values=values), report)


def _housed_cells(
    grid: PopGrid,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    footprint_m2: npt.ArrayLike,
    floors: npt.ArrayLike,
    housing_floor_m2: npt.ArrayLike,
    reference: BoolArray,
    config: PopulationCorrection,
) -> _HousedCells:
    footprint = np.asarray(footprint_m2, np.float64)
    storeys = np.asarray(floors, np.float64)
    housing_floor = np.asarray(housing_floor_m2, np.float64)
    if not (len(np.asarray(lon)) == len(np.asarray(lat)) == len(footprint) == len(storeys)):
        raise ValueError("lon, lat, footprint_m2 and floors must have one value per building")
    if len(housing_floor) != len(footprint):
        raise ValueError("housing_floor_m2 must have one value per building")
    rows, cols = grid.values.shape
    row, col = grid.cell_of(lon, lat)
    inside = row >= 0
    cell = row[inside] * cols + col[inside]
    floor_all = (footprint * storeys)[inside]

    def per_cell(weights: FloatArray) -> FloatArray:
        return np.bincount(cell, weights=weights, minlength=rows * cols)

    housing = per_cell(housing_floor[inside])
    floor_sum = per_cell(floor_all)
    height = _safe_ratio(per_cell(floor_all * storeys[inside]), floor_sum)
    size = _safe_ratio(per_cell(floor_all * footprint[inside]), floor_sum)
    coverage = per_cell(footprint[inside]) / _cell_area_m2(grid)
    housed = np.flatnonzero(housing > 0)
    ids = _class_ids(height[housed], size[housed], coverage[housed], config)
    class_ids, class_of = np.unique(ids, return_inverse=True)
    return _HousedCells(
        cell=housed,
        people=grid.values.ravel()[housed],
        floor=housing[housed],
        class_of=class_of,
        class_ids=class_ids,
        in_reference=reference.ravel()[housed],
    )


def _safe_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _cell_area_m2(grid: PopGrid) -> float:
    mid_lat = grid.north - grid.values.shape[0] * grid.dlat / 2
    width = grid.dlon * M_PER_DEGREE_LON_AT_EQUATOR * math.cos(math.radians(mid_lat))
    return width * (grid.dlat * M_PER_DEGREE_LAT)


def _class_ids(
    height: FloatArray, size: FloatArray, coverage: FloatArray, config: PopulationCorrection
) -> IntArray:
    n_footprint = len(config.footprint_edges) + 1
    n_coverage = len(config.coverage_edges) + 1
    return (
        np.digitize(height, config.height_edges) * n_footprint * n_coverage
        + np.digitize(size, config.footprint_edges) * n_coverage
        + np.digitize(coverage, config.coverage_edges)
    ).astype(np.int64)


def _class_label(class_id: int, config: PopulationCorrection) -> str:
    n_footprint = len(config.footprint_edges) + 1
    n_coverage = len(config.coverage_edges) + 1
    height, rest = divmod(class_id, n_footprint * n_coverage)
    footprint, coverage = divmod(rest, n_coverage)
    return f"h{height}-f{footprint}-c{coverage}"


def _fit_rates(cells: _HousedCells, config: PopulationCorrection) -> _Rates:
    n = len(cells.class_ids)
    fitted = cells.in_reference
    floor_in = float(cells.floor[fitted].sum())
    people_in = float(cells.people[fitted].sum())
    if floor_in <= 0:
        raise ValueError("the population_correction reference has no housing")
    if people_in <= 0:
        raise ValueError("the population_correction reference has no people")
    floor_by_class = np.bincount(cells.class_of[fitted], weights=cells.floor[fitted], minlength=n)
    people_by_class = np.bincount(cells.class_of[fitted], weights=cells.people[fitted], minlength=n)
    pooled = (floor_by_class == 0) | (floor_by_class < config.min_class_share * floor_in)
    rate = np.where(pooled, people_in / floor_in, _safe_ratio(people_by_class, floor_by_class))
    cap = 1 / config.fit_min_m2_per_person
    return _Rates(np.minimum(rate, cap), pooled, rate > cap, floor_in)


def _class_report(
    cells: _HousedCells, rates: _Rates, config: PopulationCorrection
) -> dict[str, list[str]]:
    labels = [_class_label(int(class_id), config) for class_id in cells.class_ids]
    return {
        "used": labels,
        "pooled": [label for label, flag in zip(labels, rates.pooled, strict=True) if flag],
        "capped": [label for label, flag in zip(labels, rates.capped, strict=True) if flag],
    }


def _fold_variation(cells: _HousedCells, rates: _Rates, cols: int) -> dict[str, Any]:
    """Std over mean of each class's unpooled, uncapped rate across spatial folds of the reference.
    None for the two statistics when no class qualifies: no evidence, not zero variation."""
    fitted = cells.in_reference
    row, col = np.divmod(cells.cell[fitted], cols)
    fold = (row // FOLD_BLOCK_CELLS + col // FOLD_BLOCK_CELLS) % FOLDS
    n = len(cells.class_ids)
    key = cells.class_of[fitted] * FOLDS + fold
    floors = np.bincount(key, weights=cells.floor[fitted], minlength=n * FOLDS).reshape(n, FOLDS)
    people = np.bincount(key, weights=cells.people[fitted], minlength=n * FOLDS).reshape(n, FOLDS)
    shares = floors.sum(axis=1) / rates.reference_floor
    variations: list[float] = []
    weights: list[float] = []
    for c in np.flatnonzero(shares >= FOLD_MIN_FLOOR_SHARE):
        present = floors[c] > 0
        ratios = people[c, present] / floors[c, present]
        if int(present.sum()) >= FOLD_MIN_PRESENT and ratios.mean() > 0:
            variations.append(float(ratios.std() / ratios.mean()))
            weights.append(float(shares[c]))
    return {
        "floor_weighted": float(np.average(variations, weights=weights)) if variations else None,
        "max": max(variations) if variations else None,
        "classes": len(variations),
        "floor_share": float(sum(weights)),
    }
