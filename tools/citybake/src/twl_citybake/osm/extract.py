from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import osmium
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area, Way
from shapely import Polygon

from twl_citybake.config import CityConfig
from twl_citybake.osm.buildings import BuildingCollector, Buildings
from twl_citybake.osm.geometry import clip_polygon
from twl_citybake.osm.roads import RoadCollector, RoadGraph, length_km_by_class
from twl_citybake.osm.tags import LEVEL_HEIGHT_M, BuildingKind, HeightSource
from twl_citybake.osm.water import Water, WaterCollector
from twl_citybake.projection import LocalProjection
from twl_citybake.store import write_json, write_ragged, write_table

ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"
COVERAGE_CELL_M = 1000
DEFAULT_HEIGHT_M = 2 * LEVEL_HEIGHT_M


@dataclass(frozen=True)
class OsmLayers:
    buildings: dict[str, np.ndarray]
    rings: dict[str, np.ndarray]
    road_nodes: dict[str, np.ndarray]
    road_edges: dict[str, np.ndarray]
    road_geom: dict[str, np.ndarray]
    water: dict[str, np.ndarray]
    districts: list[dict[str, Any]]
    report: dict[str, Any]
    meta: dict[str, Any]


def extract_osm(pbf: Path, city: CityConfig) -> OsmLayers:
    proj = LocalProjection.for_bbox(city.bbox)
    clip = clip_polygon(city.bbox, proj)
    buildings, roads, water = BuildingCollector(), RoadCollector(), WaterCollector()
    factory = WKBFactory()
    for obj in osmium.FileProcessor(str(pbf)).with_areas():
        if isinstance(obj, Area):
            buildings.add(obj, factory)
            water.add(obj, factory)
        elif isinstance(obj, Way):
            roads.add(obj)

    footprints = buildings.finish(proj, clip)
    graph = roads.finish(proj, clip)
    waters = water.finish(proj, clip)
    n = len(footprints)
    untagged = footprints.height_source == HeightSource.DEFAULT
    height_m = np.where(untagged, DEFAULT_HEIGHT_M, footprints.tagged_height_m).astype(np.float32)
    table = footprints.table(height_m, np.zeros(n, np.uint16), np.zeros(n, np.uint8))
    districts = [_city_district(clip, n)]
    report = {
        "buildings": _building_report(footprints, height_m),
        "roads": _road_report(graph),
        "water": _water_report(waters),
        "banks_split": False,
        "bank_share_west": 1.0 if n else 0.0,
        "districts": {
            "count": 1,
            "buildings_min": n,
            "buildings_median": float(n),
            "buildings_max": n,
        },
        "coverage": _coverage(clip, footprints.x, footprints.y),
    }
    meta = {
        "city": city.slug,
        "bbox": [city.bbox.west, city.bbox.south, city.bbox.east, city.bbox.north],
        "proj4": proj.proj4,
        "attribution": ATTRIBUTION,
        "source": {"url": city.osm.url, "md5": city.osm.md5},
    }
    return OsmLayers(
        buildings=table,
        rings=footprints.rings,
        road_nodes=graph.nodes,
        road_edges=graph.edges,
        road_geom=graph.geom,
        water=waters.columns,
        districts=districts,
        report=report,
        meta=meta,
    )


def write_osm_layers(layers: OsmLayers, out_dir: Path) -> None:
    write_table(out_dir / "buildings", layers.buildings)
    write_ragged(out_dir / "building_rings", layers.rings, len(layers.rings["offsets"]) - 1)
    write_table(out_dir / "road_nodes", layers.road_nodes)
    write_table(out_dir / "road_edges", layers.road_edges)
    write_ragged(out_dir / "road_geom", layers.road_geom, len(layers.road_geom["offsets"]) - 1)
    write_ragged(out_dir / "water", layers.water, len(layers.water["poly_kind"]))
    write_json(out_dir / "districts.json", layers.districts)
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def _city_district(clip: Polygon, buildings: int) -> dict[str, Any]:
    centre = shapely.centroid(clip)
    return {
        "id": 0,
        "name": "City",
        "seed": np.round([centre.x, centre.y], 1).tolist(),
        "bank": 0,
        "buildings": buildings,
        "polygon": [np.round(np.asarray(clip.exterior.coords), 1).tolist()],
    }


def _round(value: Any) -> float:
    return round(float(value), 3)


def _quantiles(values: np.ndarray, names: dict[str, float]) -> dict[str, float | None]:
    if not len(values):
        return dict.fromkeys(names)
    points = np.percentile(values.astype(np.float64), list(names.values()))
    return {name: _round(p) for name, p in zip(names, points, strict=True)}


def _building_report(b: Buildings, height_m: np.ndarray) -> dict[str, Any]:
    kinds = np.bincount(b.kind, minlength=len(BuildingKind))
    sources = np.bincount(b.height_source, minlength=len(HeightSource))
    levels, counts = np.unique(b.levels[~np.isnan(b.levels)], return_counts=True)
    return {
        "count": len(b),
        "skipped_invalid": b.skipped_invalid,
        "by_kind": {kind.name: int(kinds[kind]) for kind in BuildingKind},
        "by_height_source": {source.name: int(sources[source]) for source in HeightSource},
        "height_m": _quantiles(
            height_m, {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95, "max": 100}
        ),
        "levels_histogram": {f"{v:g}": int(n) for v, n in zip(levels, counts, strict=True)},
        "footprint_m2": _quantiles(b.area_m2, {"p5": 5, "p50": 50, "p95": 95}),
    }


def _road_report(graph: RoadGraph) -> dict[str, Any]:
    n = len(graph.nodes["osm_id"])
    labels = graph.components() if n else np.zeros(0, np.int32)
    sizes = np.bincount(labels)
    return {
        "nodes": n,
        "edges": len(graph.edges["u"]),
        "length_km_by_class": length_km_by_class(graph.edges),
        "components": len(sizes),
        "largest_component_share": _round(sizes.max() / n) if n else 0.0,
    }


def _water_report(water: Water) -> dict[str, Any]:
    return {
        "river_area_km2": _round(water.river_area_m2 / 1e6),
        "lakes": water.lakes,
        "lake_area_km2": _round(water.lake_area_m2 / 1e6),
    }


def _coverage(clip: Polygon, x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """1 km cells aligned to the origin whose centres fall inside the clip polygon."""
    cell = COVERAGE_CELL_M
    min_x, min_y, max_x, max_y = clip.bounds
    i0, j0 = int(np.floor(min_x / cell)), int(np.floor(min_y / cell))
    i1, j1 = int(np.ceil(max_x / cell)), int(np.ceil(max_y / cell))
    gx, gy = np.meshgrid((np.arange(i0, i1) + 0.5) * cell, (np.arange(j0, j1) + 0.5) * cell)
    in_clip = shapely.contains_xy(clip, gx, gy)
    occupied = np.zeros_like(in_clip)
    cols = np.clip(np.floor(x / cell).astype(np.int64) - i0, 0, i1 - i0 - 1)
    rows = np.clip(np.floor(y / cell).astype(np.int64) - j0, 0, j1 - j0 - 1)
    occupied[rows, cols] = True
    return {
        "cell_m": cell,
        "cells_in_bbox": int(in_clip.sum()),
        "cells_without_buildings": int((in_clip & ~occupied).sum()),
    }
