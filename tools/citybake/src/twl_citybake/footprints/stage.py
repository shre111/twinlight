import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from shapely import Polygon

from twl_citybake.config import CityConfig, FootprintsConfig
from twl_citybake.footprints.dedupe import dedupe_tiers, valid_footprints
from twl_citybake.footprints.raster import HeightRaster, mosaic_tiles
from twl_citybake.footprints.vida import SOURCE_CODES, VidaRows, read_vida
from twl_citybake.osm.districts import assign_districts
from twl_citybake.osm.extract import coverage, district_seeds
from twl_citybake.osm.geometry import (
    GeometryArray,
    clip_polygon,
    from_wkb,
    oriented_boxes,
    outer_rings,
    ring_arrays,
    ring_polygons,
)
from twl_citybake.osm.heights import final_heights
from twl_citybake.osm.tags import LEVEL_HEIGHT_M, BuildingKind, HeightSource
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table, table_digest, write_json, write_ragged, write_table


def attribution(config: FootprintsConfig) -> list[str]:
    """Licence lines for every source; Microsoft's rows carry ODbL as of VIDA's snapshot."""
    snapshot = f"VIDA's {config.vida_snapshot} snapshot" if config.vida_snapshot else "VIDA"
    return [
        "© OpenStreetMap contributors, ODbL 1.0",
        "Google-Microsoft-OSM Open Buildings combined by VIDA, ODbL 1.0",
        "Google Open Buildings v3, CC BY 4.0 or ODbL 1.0",
        f"Microsoft Global ML Building Footprints (ODbL 1.0, via {snapshot})",
        "Google Open Buildings 2.5D Temporal (Sirko et al. 2023, arXiv:2310.11622), CC BY 4.0 or "
        "ODbL 1.0, contains modified Copernicus Sentinel data",
    ]


OSM_SOURCE = 0
SOURCE_NAMES = {"osm": OSM_SOURCE, **SOURCE_CODES}
NON_OSM_TYPE = 255
TAGGED = (HeightSource.HEIGHT, HeightSource.LEVELS)


@dataclass(frozen=True)
class BuildingLayers:
    buildings: dict[str, np.ndarray]
    rings: dict[str, np.ndarray]
    districts: list[dict[str, Any]]
    report: dict[str, Any]
    meta: dict[str, Any]


@dataclass(frozen=True)
class NewFootprints:
    """Kept Google then kept Microsoft footprints, each in cache order, in local metres."""

    geoms: GeometryArray
    source: npt.NDArray[np.uint8]
    confidence: npt.NDArray[np.float32]
    counts: dict[str, dict[str, int]]


def build_buildings(
    osm_layers: Path, osm_pbf: Path, vida: Path, heights: Path, city: CityConfig
) -> BuildingLayers:
    """P-020's buildings followed by the VIDA footprints that survive the D-021 dedupe, with
    heights from OSM tags, else the 2.5D raster, else the P-020 default chain."""
    if city.footprints is None:
        raise ValueError(f"{city.slug} has no [footprints] sources")
    config = city.footprints
    proj = LocalProjection.for_bbox(city.bbox)
    clip = clip_polygon(city.bbox, proj)
    osm = read_table(osm_layers / "buildings")
    osm_rings = read_table(osm_layers / "building_rings")
    districts = json.loads((osm_layers / "districts.json").read_text(encoding="utf-8"))
    osm_geoms = ring_polygons(osm_rings["offsets"], osm_rings["xy"])
    new = select_footprints(read_vida(vida), proj, clip, osm_geoms, config.overlap_drop)
    columns, rings = _new_columns(new, osm_pbf, city, districts)
    table = {
        name: np.concatenate([osm[name], columns[name].astype(osm[name].dtype)]) for name in osm
    }
    table["source"] = np.concatenate([np.full(len(osm["x"]), OSM_SOURCE, np.uint8), new.source])
    table["confidence"] = np.concatenate([np.ones(len(osm["x"]), np.float32), new.confidence])
    raster = HeightRaster.read(heights)
    medians = raster.footprint_medians(
        np.concatenate([osm_geoms, new.geoms]), proj, config.raster_min_height_m
    )
    table["height_m"], table["height_source"] = _heights(table, medians)
    table["raster_height_m"] = medians.astype(np.float32)
    counts = np.bincount(table["district"], minlength=len(districts))
    report = _report(table, new, medians, clip, len(osm["x"]), [d["name"] for d in districts])
    return BuildingLayers(
        buildings=table,
        rings={
            "offsets": np.concatenate(
                [osm_rings["offsets"], rings["offsets"][1:] + osm_rings["offsets"][-1]]
            ).astype(np.uint32),
            "xy": np.concatenate([osm_rings["xy"], rings["xy"]]).astype(np.float32),
        },
        districts=[{**d, "buildings": int(counts[d["id"]])} for d in districts],
        report=report,
        meta=_meta(city, config, proj, heights, osm_layers),
    )


def write_building_layers(layers: BuildingLayers, out_dir: Path) -> None:
    write_table(out_dir / "buildings", layers.buildings)
    write_ragged(out_dir / "building_rings", layers.rings, len(layers.rings["offsets"]) - 1)
    write_json(out_dir / "districts.json", layers.districts)
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def select_footprints(
    rows: VidaRows,
    proj: LocalProjection,
    clip: Polygon,
    osm_geoms: GeometryArray,
    threshold: float,
) -> NewFootprints:
    """Valid footprints whose centroid lies in the clip, as P-020 keeps OSM buildings, deduped
    Google against OSM, then Microsoft against OSM and the kept Google."""
    geoms = from_wkb(rows.wkb, proj)
    valid = valid_footprints(geoms)
    inside = np.zeros(len(geoms), np.bool_)
    centroids = shapely.centroid(geoms[valid])
    inside[valid] = shapely.contains_xy(clip, shapely.get_x(centroids), shapely.get_y(centroids))
    tiers = [np.flatnonzero(inside & (rows.source == code)) for code in SOURCE_CODES.values()]
    masks = dedupe_tiers(osm_geoms, [geoms[tier] for tier in tiers], threshold)
    kept = np.concatenate([tier[mask] for tier, mask in zip(tiers, masks, strict=True)])
    counts = {}
    for (name, code), mask in zip(SOURCE_CODES.items(), masks, strict=True):
        of_source = rows.source == code
        counts[name] = {
            "fetched": int(of_source.sum()),
            "invalid": int((of_source & ~valid).sum()),
            "outside_bbox": int((of_source & valid & ~inside).sum()),
            "dropped_by_dedupe": int((~mask).sum()),
            "kept": int(mask.sum()),
        }
    return NewFootprints(
        geoms=geoms[kept].astype(object),
        source=rows.source[kept],
        confidence=rows.confidence[kept],
        counts=counts,
    )


def _new_columns(
    new: NewFootprints, osm_pbf: Path, city: CityConfig, districts: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """P-020 columns for the new footprints; district and bank from P-020's seeds and banks."""
    n = len(new.geoms)
    centroids = shapely.centroid(new.geoms)
    x, y = shapely.get_x(centroids), shapely.get_y(centroids)
    seeds, banks = district_seeds(osm_pbf, city)
    bank = banks.of(x, y)
    district, assigned = assign_districts(seeds, banks, x, y, bank)
    if [d["name"] for d in assigned] != [d["name"] for d in districts]:
        raise ValueError(f"{osm_pbf} seeds other districts than the OSM layers; rerun osm")
    w, d, rot = oriented_boxes(new.geoms)
    offsets, xy = ring_arrays(outer_rings(new.geoms))
    columns = {
        "osm_type": np.full(n, NON_OSM_TYPE),
        "osm_id": np.full(n, -1),
        "x": x,
        "y": y,
        "w": w,
        "d": d,
        "rot": rot,
        "area_m2": shapely.area(new.geoms),
        "height_m": np.full(n, np.nan),
        "height_source": np.full(n, HeightSource.DEFAULT),
        "kind": np.full(n, BuildingKind.UNKNOWN),
        "district": district,
        "bank": bank,
    }
    return columns, {"offsets": offsets, "xy": xy}


def _heights(
    table: dict[str, np.ndarray], medians: np.ndarray
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Tagged OSM heights stay; the rest take the raster median, else the P-020 default chain
    sampled from the same LEVELS-tagged buildings."""
    source = table["height_source"]
    tagged = np.isin(source, TAGGED)
    from_raster = ~tagged & ~np.isnan(medians)
    height_source = np.where(
        tagged, source, np.where(from_raster, HeightSource.RASTER, HeightSource.DEFAULT)
    ).astype(np.uint8)
    height = np.where(tagged, table["height_m"], np.where(from_raster, medians, np.nan))
    # P-020 stores float32(levels x 3.2); tagged levels have at most two decimals
    levels = np.where(
        source == HeightSource.LEVELS,
        np.round(table["height_m"].astype(np.float64) / LEVEL_HEIGHT_M, 3),
        np.nan,
    )
    final = final_heights(
        height, height_source, table["district"], table["kind"], table["area_m2"], levels
    )
    return final, height_source


def _report(
    table: dict[str, np.ndarray],
    new: NewFootprints,
    medians: np.ndarray,
    clip: Polygon,
    n_osm: int,
    names: list[str],
) -> dict[str, Any]:
    sources = np.bincount(table["height_source"], minlength=len(HeightSource))
    mix = np.bincount(
        table["district"].astype(np.int64) * len(SOURCE_NAMES) + table["source"],
        minlength=len(names) * len(SOURCE_NAMES),
    ).reshape(len(names), len(SOURCE_NAMES))
    cover = coverage(clip, table["x"], table["y"])
    before = coverage(clip, table["x"][:n_osm], table["y"][:n_osm])
    osm = {"fetched": n_osm, "invalid": 0, "outside_bbox": 0, "dropped_by_dedupe": 0}
    return {
        "count": len(table["x"]),
        "sources": {"osm": {**osm, "kept": n_osm}, **new.counts},
        "by_height_source": {s.name: int(sources[s]) for s in HeightSource},
        "raster_check": _raster_check(table, medians, n_osm),
        "districts": {
            name: {source: int(mix[i, code]) for source, code in SOURCE_NAMES.items()}
            for i, name in enumerate(names)
        },
        "coverage": {
            **cover,
            "cells_without_osm_buildings": before["cells_without_buildings"],
        },
    }


def _raster_check(table: dict[str, np.ndarray], medians: np.ndarray, n_osm: int) -> dict[str, Any]:
    """The raster median against LEVELS-tagged OSM heights (levels x 3.2 m)."""
    levels = table["height_source"][:n_osm] == HeightSource.LEVELS
    raster = medians[:n_osm][levels].astype(np.float64)
    tagged = table["height_m"][:n_osm][levels].astype(np.float64)
    seen = ~np.isnan(raster)
    if not seen.any():
        return {"n": 0, "median_ratio": None, "median_abs_error_m": None}
    raster, tagged = raster[seen], tagged[seen]
    return {
        "n": int(seen.sum()),
        "median_ratio": round(float(np.median(raster / tagged)), 3),
        "median_abs_error_m": round(float(np.median(np.abs(raster - tagged))), 3),
    }


def _meta(
    city: CityConfig,
    config: FootprintsConfig,
    proj: LocalProjection,
    heights: Path,
    osm_layers: Path,
) -> dict[str, Any]:
    return {
        "city": city.slug,
        "bbox": [city.bbox.west, city.bbox.south, city.bbox.east, city.bbox.north],
        "proj4": proj.proj4,
        "attribution": attribution(config),
        "osm_buildings_sha256": table_digest(osm_layers / "buildings"),
        "source": {
            "osm": {"url": city.osm.url, "md5": city.osm.md5},
            "vida": {
                "url": config.vida_url,
                "etag": config.vida_etag,
                "snapshot": config.vida_snapshot,
            },
            "heights": {"manifest": config.heights_manifest, "tiles": mosaic_tiles(heights)},
        },
        "overlap_drop": config.overlap_drop,
        "raster_min_height_m": config.raster_min_height_m,
        "raster_cell_m": config.raster_cell_m,
    }
