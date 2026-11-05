import argparse
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from twl_citybake.config import CityConfig, DataDirs, PopulationRaster, load_city
from twl_citybake.footprints.raster import fetch_height_mosaic
from twl_citybake.footprints.stage import build_buildings, write_building_layers
from twl_citybake.footprints.vida import fetch_vida
from twl_citybake.osm.clip import clip_extract
from twl_citybake.osm.extract import extract_osm, write_osm_layers
from twl_citybake.personas.pool import fetch_pool
from twl_citybake.personas.stage import run_personas, write_persona_layers
from twl_citybake.population.stage import run_population, write_population_layers
from twl_citybake.sources import extract_member, fetch
from twl_citybake.store import table_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citybake",
        description="Bake a Twinlight city pack from OpenStreetMap, population and persona data.",
    )
    parser.add_argument("--version", action="version", version=version("twl-citybake"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="raw downloads and stage outputs (default $TWL_DATA_DIR or ./data)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser(
        "fetch", help="download the pinned OSM extract and clip it, and the population raster"
    )
    fetch_cmd.add_argument("city")
    osm_cmd = commands.add_parser("osm", help="extract buildings, roads, water and districts")
    osm_cmd.add_argument("city")
    footprints_cmd = commands.add_parser(
        "footprints", help="fill OSM building gaps from VIDA footprints with 2.5D heights"
    )
    footprints_cmd.add_argument("city")
    population_cmd = commands.add_parser(
        "population", help="place raster people in households per building, sample residents"
    )
    population_cmd.add_argument("city")
    personas_cmd = commands.add_parser(
        "personas", help="assign Nemotron personas and D-026 segments to residents"
    )
    personas_cmd.add_argument("city")
    return parser


def run_fetch(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    region = fetch(config.osm.url, config.osm.md5, dirs.raw / "osm" / config.osm.filename)
    clipped = dirs.stage(city, "osm") / f"{city}.osm.pbf"
    clip_extract(region, clipped, config.bbox)
    print(f"{clipped} ({clipped.stat().st_size / 1e6:.1f} MB)")
    if config.population_raster is not None:
        raster = fetch_population_raster(config.population_raster, dirs)
        print(f"{raster} ({raster.stat().st_size / 1e6:.1f} MB)")


def fetch_population_raster(source: PopulationRaster, dirs: DataDirs) -> Path:
    folder = dirs.raw / "population"
    download = fetch(source.url, source.md5, folder / source.filename)
    if not source.member:
        return download
    return extract_member(download, source.member, folder / source.member)


def population_raster_path(config: CityConfig, dirs: DataDirs) -> Path:
    if config.population_raster is None:
        raise SystemExit(f"{config.slug} has no [population_raster] source")
    return dirs.raw / "population" / config.population_raster.raster_name


def run_osm(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    stage = dirs.stage(city, "osm")
    pbf = stage / f"{city}.osm.pbf"
    if not pbf.exists():
        raise SystemExit(f"{pbf} is missing; run `citybake fetch {city}` first")
    layers = extract_osm(pbf, config)
    write_osm_layers(layers, stage / "layers")
    print(summary(layers.report))
    print(stage / "layers")


def run_footprints(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    footprints = config.footprints
    if footprints is None:
        raise SystemExit(f"{city} has no [footprints] sources")
    osm = dirs.stage(city, "osm")
    pbf, osm_layers = osm / f"{city}.osm.pbf", osm / "layers"
    if not pbf.exists() or not (osm_layers / "buildings").exists():
        raise SystemExit(f"{osm} is incomplete; run `citybake fetch {city}` and `citybake osm`")
    cache = dirs.stage(city, "footprints")
    vida = fetch_vida(
        footprints.vida_url, footprints.vida_etag, config.bbox, cache / "vida.parquet"
    )
    heights = fetch_height_mosaic(
        footprints.heights_manifest, config.bbox, footprints.raster_cell_m, cache / "heights.tif"
    )
    layers = build_buildings(osm_layers, pbf, vida, heights, config)
    out = dirs.stage(city, "buildings") / "layers"
    write_building_layers(layers, out)
    print(footprints_summary(layers.report))
    print(out)


def building_layers(city: str, dirs: DataDirs) -> Path:
    """The combined footprints layers once `citybake footprints` has run, else the OSM ones;
    combined layers baked from older OSM layers are refused."""
    combined = dirs.stage(city, "buildings") / "layers"
    osm = dirs.stage(city, "osm") / "layers"
    if not (combined / "buildings").exists():
        return osm
    meta = json.loads((combined / "layer.json").read_text(encoding="utf-8"))
    if meta.get("osm_buildings_sha256") != table_digest(osm / "buildings"):
        raise SystemExit(f"{combined} predates the OSM layers; rerun `citybake footprints {city}`")
    return combined


def footprints_summary(report: dict[str, Any]) -> str:
    lines = [
        f"{name} {c['fetched']:,} fetched: {c['invalid']:,} invalid, {c['outside_bbox']:,} "
        f"outside the bbox, {c['dropped_by_dedupe']:,} overlapping, {c['kept']:,} kept"
        for name, c in report["sources"].items()
        if name != "osm"
    ]
    heights = ", ".join(f"{k.lower()} {v:,}" for k, v in report["by_height_source"].items())
    check, cover = report["raster_check"], report["coverage"]
    validation = report["height_correction"]["validation"]
    bands = ", ".join(
        f"{band} {v['before']} to {v['after']} (signed {v['signed_after']}, n {v['n']:,})"
        for band, v in validation["bands"].items()
    )
    return "\n".join(
        [
            *lines,
            f"buildings {report['count']:,}; heights {heights}",
            f"raster vs levels on {check['n']:,} buildings: median ratio "
            f"{check['median_ratio']} ({check['median_ratio_corrected']} corrected), "
            f"median abs error {check['median_abs_error_m']} m",
            f"held-out median abs % error by storeys, raw to corrected: {bands}",
            f"coverage {cover['cells_without_buildings']} of {cover['cells_in_bbox']} "
            f"{cover['cell_m']} m cells without buildings "
            f"(OSM alone {cover['cells_without_osm_buildings']})",
        ]
    )


def run_population_command(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    if config.households is None:
        raise SystemExit(f"{city} has no [households] census table")
    layers = building_layers(city, dirs)
    if not (layers / "buildings").exists():
        raise SystemExit(f"{layers} is missing; run `citybake osm {city}` first")
    raster = population_raster_path(config, dirs)
    if not raster.exists():
        raise SystemExit(f"{raster} is missing; run `citybake fetch {city}` first")
    pbf = dirs.stage(city, "osm") / f"{city}.osm.pbf"
    population = run_population(layers, raster, config, pbf if config.population_total else None)
    print(f"buildings from {layers}")
    out = dirs.stage(city, "population")
    write_population_layers(population, out)
    print(population_summary(population.report, config.households.mean_size))
    print(out)


def population_summary(report: dict, census_mean_size: float) -> str:
    raster, cap, d = report["raster"], report["capacity"], report["districts"]
    hh, fit, res = report["households"], report["ipf"], report["residents"]
    outside = ", ".join(d["outside_1pct"]) or "none"
    return "\n".join(
        [
            f"raster {raster['people_in_bbox']:,.0f} people in {raster['cells']} cells; "
            f"{raster['people_without_buildings']:,.0f} in {raster['cells_without_buildings']} "
            "cells without buildings are unplaced",
            f"capacity {cap['people']:,} people in {cap['buildings_with_capacity']} buildings, "
            f"{cap['buildings_over_capacity']} over capacity",
            f"districts {d['count']}: marginal {d['marginal_people']:,.0f}, synthesised "
            f"{d['synthesised_people']:,}, shortfall {d['shortfall_people']:,.0f}, max error "
            f"{d['max_rel_error_1000_plus']:.2%} at 1,000+ people, outside 1% {outside}",
            f"households {hh['count']:,}, {hh['people']:,} people, mean size "
            f"{hh['mean_size']:.2f} (census {census_mean_size:.2f})",
            f"ipf {fit['districts']} districts, max {fit['max_iterations']} iterations, "
            f"max error {fit['max_rel_error']:.1e}, all converged {fit['all_converged']}",
            f"residents {res['count']:,} from {res['households']:,} households, weight "
            f"{res['weight']:.2f}, {res['adults_represented']:,.0f} adults represented",
        ]
    )


def run_personas_command(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    if config.personas is None:
        raise SystemExit(f"{city} has no [personas] section")
    population = dirs.stage(city, "population")
    if not (population / "residents").exists():
        raise SystemExit(f"{population} is incomplete; run `citybake population {city}` first")
    out = dirs.stage(city, "personas")
    pool = fetch_pool(config.personas, out / "pool.parquet")
    layers = run_personas(population, pool, config)
    write_persona_layers(layers, out)
    print(personas_summary(layers.report))
    print(out)


def personas_summary(report: dict) -> str:
    pool, married, couples = report["pool"], report["married"], report["couples"]
    segments = [
        f"  {name} {s['pool_share']:.2%} in the pool, {s['resident_share']:.2%} of residents"
        for name, s in report["segments"].items()
    ]
    distances = ", ".join(
        f"{name} {marginal['total_variation']}" for name, marginal in report["marginals"].items()
    )
    return "\n".join(
        [
            f"pool {pool['rows']:,} personas, {pool['district']}, {pool['state']}",
            *segments,
            f"max segment difference {report['max_segment_difference']:.2%}, "
            f"within tolerance {report['within_tolerance']}",
            f"total variation {distances}",
            f"married {married['pool_share']:.2%} in the pool, {married['resident_share']:.2%} of "
            f"residents, {married['free_draw_married_probability']:.2%} on a free draw",
            f"couples {couples['count']:,} holding {couples['residents_in_couples']:,} residents, "
            f"gap p10 {couples['gap_p10']} p50 {couples['gap_p50']} p90 {couples['gap_p90']}, "
            f"husband older or equal {couples['husband_older_or_equal_share']:.2%}",
        ]
    )


def summary(report: dict) -> str:
    b, r, w, d = report["buildings"], report["roads"], report["water"], report["districts"]
    sources = ", ".join(f"{k.lower()} {v}" for k, v in b["by_height_source"].items())
    heights = b["height_m"]
    cover = report["coverage"]
    return "\n".join(
        [
            f"buildings {b['count']} ({b['skipped_invalid']} invalid skipped); heights {sources}",
            f"height m p5 {heights['p5']} p50 {heights['p50']} p95 {heights['p95']}",
            f"roads {r['nodes']} nodes, {r['edges']} edges, {r['components']} components "
            f"(largest {r['largest_component_share']:.1%})",
            f"river {w['river_area_km2']} km2, {w['lakes']} lakes {w['lake_area_km2']} km2, "
            f"banks split {report['banks_split']}",
            f"districts {d['count']} (buildings min {d['buildings_min']}, "
            f"median {d['buildings_median']}, max {d['buildings_max']}); "
            f"{d['extra_seeds']} extra seeds, {d['merged']} merged, {d['renamed']} renamed",
            f"coverage {cover['cells_without_buildings']} of {cover['cells_in_bbox']} "
            f"{cover['cell_m']} m cells without buildings",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dirs = DataDirs.from_env(args.data_dir)
    if args.command == "fetch":
        run_fetch(args.city, dirs)
    elif args.command == "osm":
        run_osm(args.city, dirs)
    elif args.command == "footprints":
        run_footprints(args.city, dirs)
    elif args.command == "population":
        run_population_command(args.city, dirs)
    elif args.command == "personas":
        run_personas_command(args.city, dirs)
    return 0
