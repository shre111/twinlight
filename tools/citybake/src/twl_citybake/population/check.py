"""Validation only (D-030): nothing here feeds a stage table, an attribution or a pack."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely

from twl_citybake.config import (
    ALL_TARGETS,
    TOTAL_TARGET,
    CheckRegion,
    PopulationCheck,
    WardGroup,
)
from twl_citybake.population.raster import repaired_polygon

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Ward:
    name: str
    geometry: shapely.Geometry


def ward_key(name: str) -> str:
    """The file numbers its wards, `28 KHADIA`; a configured `KHADIA` compares equal."""
    return name.lstrip("0123456789 ")


def load_wards(path: Path, name_property: str) -> list[Ward]:
    """Ward polygons in file order, 2D, invalid ones repaired. Names stay as the file has them."""
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features") if isinstance(data, dict) else None
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection with features")
    return [_ward(feature, name_property, path) for feature in features]


def _ward(feature: Any, name_property: str, path: Path) -> Ward:
    properties = feature.get("properties") if isinstance(feature, dict) else None
    name = properties.get(name_property) if isinstance(properties, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path} has a feature without a {name_property!r} name")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"{path}: ward {name!r} has no geometry")
    return Ward(
        name, repaired_polygon(shapely.force_2d(shapely.from_geojson(json.dumps(geometry))))
    )


def assign_wards(
    wards: Sequence[Ward], lon: npt.ArrayLike, lat: npt.ArrayLike
) -> npt.NDArray[np.int32]:
    """Ward index of each point, -1 outside every ward; on an overlap the lowest index wins."""
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    owner = np.full(lon.shape, -1, np.int32)
    for index, ward in enumerate(wards):
        west, south, east, north = shapely.bounds(ward.geometry)
        near = np.flatnonzero(
            (owner < 0) & (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        )
        shapely.prepare(ward.geometry)
        owner[near[shapely.contains_xy(ward.geometry, lon[near], lat[near])]] = index
    return owner


def population_check(
    config: PopulationCheck,
    persons: npt.ArrayLike,
    raster_people: npt.ArrayLike,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    wards: Sequence[Ward],
    polygons: Mapping[str, shapely.Geometry],
) -> dict[str, Any]:
    """`persons` per building from the corrected bake against the ward and region bands;
    `raster_people` per building gives the before figures. All values are unrounded."""
    if not wards:
        raise ValueError("population_check needs at least one ward")
    corrected = np.asarray(persons, np.float64)
    raster = np.asarray(raster_people, np.float64)
    if len(corrected) != len(raster):
        raise ValueError("persons and raster_people must have one value per building")
    owner = assign_wards(wards, lon, lat)
    inside = owner >= 0
    people = np.bincount(owner[inside], weights=corrected[inside], minlength=len(wards))
    before = np.bincount(owner[inside], weights=raster[inside], minlength=len(wards))
    ratio = people / config.ward_people_2011
    raster_ratio = before / config.ward_people_2011
    index_of = _ward_index(wards)
    groups = {
        group.name: _group(group, index_of, ratio, raster_ratio) for group in config.ward_groups
    }
    regions = {
        region.name: _region(region, corrected, raster, lon, lat, x, y, polygons)
        for region in config.regions
    }
    total_people = float(people.sum())
    total_ratio = total_people / config.total_people_2011
    return {
        "source": config.source,
        "wards": [
            {
                "name": ward.name,
                "people": float(people[i]),
                "ratio": float(ratio[i]),
                "raster_ratio": float(raster_ratio[i]),
            }
            for i, ward in enumerate(wards)
        ],
        "ward_summary": _ward_summary(ratio, raster_ratio),
        "outside_wards_people": float(corrected[~inside].sum()),
        "total": {
            "people": total_people,
            "ratio_to_2011": total_ratio,
            "raster_ratio_to_2011": float(before.sum()) / config.total_people_2011,
            "band": list(config.total_growth),
        },
        "groups": groups,
        "regions": regions,
        "within_targets": _within_targets(config, groups, regions, total_ratio),
    }


def _ward_index(wards: Sequence[Ward]) -> dict[str, int]:
    index_of: dict[str, int] = {}
    for i, ward in enumerate(wards):
        index_of.setdefault(ward_key(ward.name), i)
    return index_of


def _group(
    group: WardGroup, index_of: Mapping[str, int], ratio: FloatArray, raster_ratio: FloatArray
) -> dict[str, Any]:
    if missing := [ward for ward in group.wards if ward_key(ward) not in index_of]:
        raise ValueError(f"population_check ward {missing[0]!r} is not in the ward file")
    members = [index_of[ward_key(ward)] for ward in group.wards]
    if len(np.unique(members)) != len(members):
        raise ValueError(f"population_check group {group.name} names one ward twice")
    values = ratio[members]
    low, high = group.ward_ratio if group.ward_ratio is not None else (-np.inf, np.inf)
    below = [w for w, v in zip(group.wards, values, strict=True) if v < low]
    above = [w for w, v in zip(group.wards, values, strict=True) if v > high]
    missed = [w for w in group.wards if w in below or w in above]
    return {
        "mean_ratio": float(values.mean()),
        "raster_mean_ratio": float(raster_ratio[members].mean()),
        "band": None if group.mean_ratio is None else list(group.mean_ratio),
        "wards": {w: float(v) for w, v in zip(group.wards, values, strict=True)},
        "below_ward_band": below,
        "above_ward_band": above,
        "known_misses": [w for w in missed if w in group.known_misses],
        "unexpected_misses": [w for w in missed if w not in group.known_misses],
    }


def _region(
    region: CheckRegion,
    corrected: FloatArray,
    raster: FloatArray,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    polygons: Mapping[str, shapely.Geometry],
) -> dict[str, Any]:
    mask = _region_mask(region, lon, lat, x, y, polygons)
    return {
        "people": float(corrected[mask].sum()),
        "raster_people": float(raster[mask].sum()),
        "band": list(region.people),
    }


def _region_mask(
    region: CheckRegion,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    polygons: Mapping[str, shapely.Geometry],
) -> npt.NDArray[np.bool_]:
    if region.box is not None:
        x0, y0, x1, y1 = region.box
        xs, ys = np.asarray(x, np.float64), np.asarray(y, np.float64)
        return (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    if region.polygon not in polygons:
        raise ValueError(f"population_check region {region.name}: no polygon {region.polygon}")
    boundary = polygons[region.polygon]
    shapely.prepare(boundary)
    return shapely.contains_xy(boundary, np.asarray(lon, np.float64), np.asarray(lat, np.float64))


def _ward_summary(ratio: FloatArray, raster_ratio: FloatArray) -> dict[str, Any]:
    p10, median, p90 = np.percentile(ratio, [10, 50, 90])
    return {
        "count": len(ratio),
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
        "log_std": _log_std(ratio),
        "raster_log_std": _log_std(raster_ratio),
    }


def _log_std(ratio: FloatArray) -> float | None:
    """Information only; nothing selects on it."""
    positive = ratio[ratio > 0]
    return float(np.std(np.log(positive))) if len(positive) else None


def _within_targets(
    config: PopulationCheck,
    groups: Mapping[str, dict[str, Any]],
    regions: Mapping[str, dict[str, Any]],
    total_ratio: float,
) -> dict[str, bool]:
    targets: dict[str, bool] = {}
    for group in config.ward_groups:
        if group.mean_ratio is not None:
            low, high = group.mean_ratio
            targets[group.name] = bool(low <= groups[group.name]["mean_ratio"] <= high)
    low, high = config.total_growth
    targets[TOTAL_TARGET] = bool(low <= total_ratio <= high)
    for region in config.regions:
        low, high = region.people
        targets[region.name] = bool(low <= regions[region.name]["people"] <= high)
    targets[ALL_TARGETS] = all(targets.values())
    return targets
