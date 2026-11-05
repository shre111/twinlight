import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from twl_citybake.config import BBox, CityConfig
from twl_citybake.population.allocate import Allocation, allocate
from twl_citybake.population.capacity import residential_capacity, storeys
from twl_citybake.population.check import load_wards, population_check
from twl_citybake.population.correction import Correction, correct_raster
from twl_citybake.population.households import Households, synthesise_households
from twl_citybake.population.nonresidential import NonResidential, nonresidential_buildings
from twl_citybake.population.raster import (
    PopGrid,
    read_boundaries,
    read_boundary,
    read_raster,
    rescale_to_total,
)
from twl_citybake.population.residents import expected_adults, sample_residents
from twl_citybake.projection import LocalProjection
from twl_citybake.rng import Stream, stream
from twl_citybake.store import read_table, write_json, write_table

CENSUS_ATTRIBUTION = "Census of India 2011 (HH-01, C-14)"
# More than the rest of the report: a ratio just outside a band should not read as on its edge.
CHECK_DIGITS = 6


@dataclass(frozen=True)
class PopulationLayers:
    building_population: dict[str, np.ndarray]
    households: dict[str, np.ndarray]
    residents: dict[str, np.ndarray]
    report: dict[str, Any]
    meta: dict[str, Any]


def run_population(
    osm_layers: Path,
    raster: Path,
    city: CityConfig,
    osm_pbf: Path | None = None,
    wards: Path | None = None,
) -> PopulationLayers:
    """`osm_pbf` is read only for the boundaries of a `population_total`, a `population_correction`
    or a `population_check`. `wards` is the check's ward file, validation only."""
    if city.households is None:
        raise ValueError(f"{city.slug} has no [households] census table")
    if city.population_check is not None and wards is None:
        raise ValueError(f"{city.slug} sets population_check, so its ward file is needed")
    buildings = read_table(osm_layers / "buildings")
    districts = json.loads((osm_layers / "districts.json").read_text(encoding="utf-8"))
    if [d["id"] for d in districts] != list(range(len(districts))):
        raise ValueError("districts.json ids must run 0..n-1 in order")
    pop = city.population
    full_capacity = residential_capacity(
        buildings["area_m2"],
        buildings["height_m"],
        buildings["kind"],
        pop.share_by_kind(),
        pop.min_floor_m2_per_person,
    )
    lon, lat = LocalProjection.for_bbox(city.bbox).inverse(buildings["x"], buildings["y"])
    excluded = _nonresidential(city, lon, lat, osm_pbf)
    capacity = full_capacity if excluded is None else np.where(excluded.mask, 0, full_capacity)
    grid = read_raster(raster, city.bbox)
    total = _total_override(city, grid, osm_pbf)
    if total is not None:
        grid = total.grid
    boundaries = _boundaries(city, osm_pbf)
    correction = _correction(city, grid, buildings, capacity, lon, lat, boundaries)
    alloc = allocate(
        grid if correction is None else correction.grid,
        lon,
        lat,
        capacity,
        buildings["district"],
        len(districts),
    )
    raster_alloc = (
        alloc
        if correction is None
        else allocate(grid, lon, lat, full_capacity, buildings["district"], len(districts))
    )
    counts = np.asarray(city.households.shrunk_counts(), np.float64)
    households = synthesise_households(
        alloc.placed,
        alloc.marginal,
        capacity,
        buildings["district"],
        len(districts),
        np.asarray(city.households.sizes, np.int64),
        counts,
    )
    residents = sample_residents(
        households,
        buildings,
        expected_adults(city.households.sizes, counts, pop.adult_share),
        pop.residents,
        pop.shopper_share,
        stream(city.seed, Stream.POPULATION),
    )
    persons = np.bincount(households.building, minlength=len(capacity), weights=households.size)
    building_population = {
        "capacity": capacity,
        "target": alloc.target,
        "placed": alloc.placed,
        "persons": persons.astype(np.int64),
    }
    names = [d["name"] for d in districts]
    housed = full_capacity > 0
    edge = edge_people(grid, city.bbox, lon[housed], lat[housed])
    census = np.asarray(city.households.counts, np.float64)
    census_shares = {
        str(size): float(n / census.sum())
        for size, n in zip(city.households.sizes, census, strict=True)
    }
    report = _report(
        grid,
        raster_alloc,
        alloc,
        building_population,
        households,
        residents,
        names,
        edge,
        census_shares,
    )
    report["population"] = _rounded(
        {
            "adult_share": pop.adult_share,
            "adult_share_source": pop.adult_share_source,
            "shopper_share": pop.shopper_share,
            "household_source": city.households.source,
            "household_size_shrink": city.households.size_shrink,
            "census_mean_household_size": city.households.mean_size,
        }
    )
    report["population_total"] = None if total is None else _rounded(total.report)
    report["correction"] = (
        None
        if correction is None
        else _rounded(
            {**correction.report, "excluded": _excluded_report(excluded, full_capacity)},
            CHECK_DIGITS,
        )
    )
    report["check"] = None
    if city.population_check is not None and wards is not None:
        check = population_check(
            city.population_check,
            persons,
            raster_alloc.placed,
            lon,
            lat,
            buildings["x"],
            buildings["y"],
            load_wards(wards, city.population_check.name_property),
            boundaries,
        )
        report["check"] = _rounded(check, CHECK_DIGITS)
    source = city.population_raster
    meta = {
        "city": city.slug,
        "seed": city.seed,
        "attribution": [
            *([] if source is None else [f"{source.source}, {source.licence}"]),
            *([] if city.population_total is None else [city.population_total.source]),
            CENSUS_ATTRIBUTION,
        ],
        "shopper_share": pop.shopper_share,
        "mean_weight": report["residents"]["weight"],
        "household_size_shrink": city.households.size_shrink,
        "source": {
            "raster": None if source is None else asdict(source),
            "population_total": None
            if city.population_total is None
            else asdict(city.population_total),
            "population_correction": None
            if city.population_correction is None
            else asdict(city.population_correction),
            "population_check": None
            if city.population_check is None
            else {
                "url": city.population_check.url,
                "md5": city.population_check.md5,
                "source": city.population_check.source,
            },
            "households": city.households.source,
        },
    }
    return PopulationLayers(
        building_population=building_population,
        households={
            "building": households.building,
            "size": households.size,
            "district": households.district,
        },
        residents=residents,
        report=report,
        meta=meta,
    )


@dataclass(frozen=True)
class _TotalOverride:
    grid: PopGrid
    report: dict[str, Any]


def _total_override(city: CityConfig, grid: PopGrid, osm_pbf: Path | None) -> _TotalOverride | None:
    total = city.population_total
    if total is None:
        return None
    if osm_pbf is None:
        raise ValueError(f"{city.slug} sets population_total, so its OSM extract is needed")
    scaled, before = rescale_to_total(grid, total.people, read_boundary(osm_pbf, total.polygon))
    return _TotalOverride(
        scaled,
        {
            "people": total.people,
            "source": total.source,
            "polygon": total.polygon,
            "as_of": total.as_of,
            "raster_people_in_polygon": before,
            "factor": total.people / before,
        },
    )


def _nonresidential(
    city: CityConfig, lon: np.ndarray, lat: np.ndarray, osm_pbf: Path | None
) -> NonResidential | None:
    config = city.population_correction
    if config is None or not (config.exclude_landuse or config.exclude_amenity):
        return None
    if osm_pbf is None:
        raise ValueError(f"{city.slug} sets population_correction, so its OSM extract is needed")
    return nonresidential_buildings(
        osm_pbf, lon, lat, config.exclude_landuse, config.exclude_amenity
    )


def _excluded_report(
    excluded: NonResidential | None, full_capacity: np.ndarray
) -> dict[str, Any] | None:
    if excluded is None:
        return None
    return {
        "areas": excluded.areas,
        "skipped_areas": excluded.skipped,
        "buildings": int(excluded.mask.sum()),
        "capacity_people": int(full_capacity[excluded.mask].sum()),
        "capacity_share": float(full_capacity[excluded.mask].sum() / full_capacity.sum()),
    }


def _boundaries(city: CityConfig, osm_pbf: Path | None) -> dict[str, shapely.Geometry]:
    corrected = city.population_correction
    checked = city.population_check
    polygons = [
        *(() if corrected is None else corrected.reference_polygons),
        *(() if checked is None else (r.polygon for r in checked.regions if r.polygon)),
    ]
    if corrected is not None and osm_pbf is None:
        raise ValueError(f"{city.slug} sets population_correction, so its OSM extract is needed")
    if not polygons:
        return {}
    if osm_pbf is None:
        raise ValueError(
            f"{city.slug} sets population_check polygons, so its OSM extract is needed"
        )
    return read_boundaries(osm_pbf, polygons)


def _correction(
    city: CityConfig,
    grid: PopGrid,
    buildings: dict[str, np.ndarray],
    capacity: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    boundaries: dict[str, shapely.Geometry],
) -> Correction | None:
    config = city.population_correction
    if config is None:
        return None
    union = shapely.union_all([boundaries[polygon] for polygon in config.reference_polygons])
    shapely.prepare(union)
    return correct_raster(
        grid,
        lon,
        lat,
        buildings["area_m2"],
        buildings["floors"] if "floors" in buildings else storeys(buildings["height_m"]),
        capacity.astype(np.float64) * city.population.min_floor_m2_per_person,
        shapely.contains_xy(union, *grid.centres()),
        config,
    )


def write_population_layers(layers: PopulationLayers, out_dir: Path) -> None:
    write_table(out_dir / "building_population", layers.building_population)
    write_table(out_dir / "households", layers.households)
    write_table(out_dir / "residents", layers.residents)
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def _report(
    grid: PopGrid,
    raster_alloc: Allocation,
    alloc: Allocation,
    building_population: dict[str, np.ndarray],
    households: Households,
    residents: dict[str, np.ndarray],
    names: list[str],
    edge_people_beyond_bbox: float,
    census_shares: dict[str, float],
) -> dict[str, Any]:
    capacity, persons = building_population["capacity"], building_population["persons"]
    districts = households.report["districts"]
    weight = residents["weight"].astype(np.float64)
    report = {
        "raster": {
            "people_in_bbox": float(grid.values.sum()),
            "cells": int(grid.values.size),
            "cells_with_people": raster_alloc.cells_with_people,
            "cells_without_buildings": raster_alloc.cells_without_buildings,
            "people_without_buildings": raster_alloc.people_without_buildings,
            "edge_people_beyond_bbox": edge_people_beyond_bbox,
        },
        "capacity": {
            "people": int(capacity.sum()),
            "buildings_with_capacity": int((capacity > 0).sum()),
            "buildings_over_capacity": int((persons > capacity).sum()),
        },
        "districts": {
            **districts,
            "outside_1pct": [names[i] for i in districts["outside_1pct"]],
            "outside_1pct_under_1000": [names[i] for i in districts["outside_1pct_under_1000"]],
            "shortfall_people": float(alloc.shortfall.sum()),
        },
        "households": {
            **households.report["households"],
            "census_size_shares": census_shares,
        },
        "ipf": households.report["ipf"],
        "residents": {
            "count": len(weight),
            "households": len(np.unique(residents["household"])),
            "weight": float(weight[0]) if len(weight) else 0.0,
            "adults_represented": float(weight.sum()),
        },
    }
    return _rounded(report)


def edge_people(grid: PopGrid, bbox: BBox, lon: np.ndarray, lat: np.ndarray) -> float:
    """People that whole edge cells add from beyond the bbox: each housed cell's population
    times the share of its area outside the bbox."""
    rows, cols = grid.values.shape
    west = grid.west + np.arange(cols) * grid.dlon
    north = grid.north - np.arange(rows) * grid.dlat
    inside_x = np.clip(
        np.minimum(west + grid.dlon, bbox.east) - np.maximum(west, bbox.west), 0, None
    )
    inside_y = np.clip(
        np.minimum(north, bbox.north) - np.maximum(north - grid.dlat, bbox.south), 0, None
    )
    inside = np.outer(inside_y / grid.dlat, inside_x / grid.dlon)
    row, col = grid.cell_of(lon, lat)
    housed = np.zeros(grid.values.shape, bool)
    housed[row[row >= 0], col[row >= 0]] = True
    return float((grid.values * (1 - inside))[housed].sum())


def _rounded(value: Any, digits: int = 3) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    return round(value, digits) if isinstance(value, float) else value
