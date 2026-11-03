import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twl_citybake.config import CityConfig
from twl_citybake.population.allocate import Allocation, allocate
from twl_citybake.population.capacity import residential_capacity
from twl_citybake.population.households import Households, synthesise_households
from twl_citybake.population.residents import expected_adults, sample_residents
from twl_citybake.population.worldpop import PopGrid, read_worldpop
from twl_citybake.projection import LocalProjection
from twl_citybake.rng import Stream, stream
from twl_citybake.store import read_table, write_json, write_table

ATTRIBUTION = ["WorldPop (CC BY 4.0)", "Census of India 2011 (HH-01, C-14)"]


@dataclass(frozen=True)
class PopulationLayers:
    building_population: dict[str, np.ndarray]
    households: dict[str, np.ndarray]
    residents: dict[str, np.ndarray]
    report: dict[str, Any]
    meta: dict[str, Any]


def run_population(osm_layers: Path, worldpop: Path, city: CityConfig) -> PopulationLayers:
    if city.households is None:
        raise ValueError(f"{city.slug} has no [households] census table")
    buildings = read_table(osm_layers / "buildings")
    districts = json.loads((osm_layers / "districts.json").read_text(encoding="utf-8"))
    pop = city.population
    capacity = residential_capacity(
        buildings["area_m2"],
        buildings["height_m"],
        buildings["kind"],
        pop.share_by_kind(),
        pop.min_floor_m2_per_person,
    )
    lon, lat = LocalProjection.for_bbox(city.bbox).inverse(buildings["x"], buildings["y"])
    grid = read_worldpop(worldpop, city.bbox)
    alloc = allocate(grid, lon, lat, capacity, buildings["district"], len(districts))
    households = synthesise_households(
        alloc.placed,
        alloc.marginal,
        capacity,
        buildings["district"],
        len(districts),
        np.asarray(city.households.sizes, np.int64),
        np.asarray(city.households.counts, np.int64),
    )
    residents = sample_residents(
        households,
        buildings,
        expected_adults(city.households.sizes, city.households.counts, pop.adult_share),
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
    names = [d["name"] for d in sorted(districts, key=lambda d: d["id"])]
    report = _report(grid, alloc, building_population, households, residents, names)
    source = city.worldpop
    meta = {
        "city": city.slug,
        "seed": city.seed,
        "attribution": ATTRIBUTION,
        "source": {
            "worldpop": None if source is None else {"url": source.url, "md5": source.md5},
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


def write_population_layers(layers: PopulationLayers, out_dir: Path) -> None:
    write_table(out_dir / "building_population", layers.building_population)
    write_table(out_dir / "households", layers.households)
    write_table(out_dir / "residents", layers.residents)
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def _report(
    grid: PopGrid,
    alloc: Allocation,
    building_population: dict[str, np.ndarray],
    households: Households,
    residents: dict[str, np.ndarray],
    names: list[str],
) -> dict[str, Any]:
    capacity, persons = building_population["capacity"], building_population["persons"]
    weight = residents["weight"].astype(np.float64)
    report = {
        "worldpop": {
            "people_in_bbox": float(grid.values.sum()),
            "cells": int(grid.values.size),
            "cells_with_people": alloc.cells_with_people,
            "cells_without_buildings": alloc.cells_without_buildings,
            "people_without_buildings": alloc.people_without_buildings,
        },
        "capacity": {
            "people": int(capacity.sum()),
            "buildings_with_capacity": int((capacity > 0).sum()),
            "buildings_over_capacity": int((persons > capacity).sum()),
        },
        "districts": {
            **households.report["districts"],
            "outside_1pct": [names[i] for i in households.report["districts"]["outside_1pct"]],
            "shortfall_people": float(alloc.shortfall.sum()),
        },
        "households": households.report["households"],
        "ipf": households.report["ipf"],
        "residents": {
            "count": len(weight),
            "households": len(np.unique(residents["household"])),
            "weight": float(weight[0]) if len(weight) else 0.0,
            "adults_represented": float(weight.sum()),
        },
    }
    return _rounded(report)


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return round(value, 3) if isinstance(value, float) else value
