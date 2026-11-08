import argparse
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from twl_citybake.config import (
    ALL_TARGETS,
    CityConfig,
    DataDirs,
    PopulationCheck,
    PopulationRaster,
    load_city,
)
from twl_citybake.footprints.raster import fetch_height_mosaic
from twl_citybake.footprints.stage import build_buildings, write_building_layers
from twl_citybake.footprints.vida import fetch_vida
from twl_citybake.graph.stage import StalePersonasError, run_graph, write_graph_layers
from twl_citybake.osm.clip import clip_extract
from twl_citybake.osm.extract import extract_osm, write_osm_layers
from twl_citybake.pack.rules import PackStageError
from twl_citybake.pack.stage import PackBudgetError, run_pack, write_pack_stage
from twl_citybake.personas.pool import fetch_pool
from twl_citybake.personas.stage import run_personas, write_persona_layers
from twl_citybake.population.stage import run_population, write_population_layers
from twl_citybake.sources import extract_member, fetch
from twl_citybake.store import META, table_digest, write_json
from twl_citybake.stores.stage import StoresError, run_stores, write_store_layers
from twl_pack import PackError


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
        "fetch",
        help="download the pinned OSM extract and clip it, the population raster and the ward file",
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
    graph_cmd = commands.add_parser(
        "graph", help="build the social graph: household, neighbour, friend and work ties"
    )
    graph_cmd.add_argument("city")
    stores_cmd = commands.add_parser(
        "stores", help="place synthetic dark stores and compute travel minutes to every hex"
    )
    stores_cmd.add_argument("city")
    pack_cmd = commands.add_parser(
        "pack", help="apply the pack-time rules and write the city pack from the baked stages"
    )
    pack_cmd.add_argument("city")
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
    if config.population_check is not None:
        check = config.population_check
        wards = fetch(check.url, check.md5, ward_file_path(city, check, dirs))
        print(f"{wards} ({wards.stat().st_size / 1e6:.1f} MB)")


def ward_file_path(city: str, check: PopulationCheck, dirs: DataDirs) -> Path:
    return dirs.raw / "population_check" / city / check.filename


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
    if not (combined / "layer.json").exists():
        raise SystemExit(f"{combined} is incomplete; rerun `citybake footprints {city}`")
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
    uses_pbf = config.population_total or config.population_correction or config.population_check
    extra: dict[str, Path] = {}
    if config.population_check is not None:
        extra["wards"] = ward_file_path(city, config.population_check, dirs)
        if not extra["wards"].exists():
            raise SystemExit(f"{extra['wards']} is missing; run `citybake fetch {city}` first")
    population = run_population(layers, raster, config, pbf if uses_pbf else None, **extra)
    print(f"buildings from {layers}")
    out = dirs.stage(city, "population")
    write_population_layers(population, out)
    print(population_summary(population.report, config.households.mean_size))
    print(out)
    check = population.report["check"]
    if check is not None and not check["within_targets"][ALL_TARGETS]:
        failed = [k for k, ok in check["within_targets"].items() if k != ALL_TARGETS and not ok]
        raise SystemExit(
            f"population check failed: {', '.join(failed)}; the layers and report were written "
            f"to {out}"
        )


def population_summary(report: dict, census_mean_size: float) -> str:
    raster, cap, d = report["raster"], report["capacity"], report["districts"]
    hh, fit, res = report["households"], report["ipf"], report["residents"]
    outside = ", ".join(d["outside_1pct"]) or "none"
    correction, check = report.get("correction"), report.get("check")
    fate = "are redistributed by the correction" if correction else "are unplaced"
    return "\n".join(
        [
            f"raster {raster['people_in_bbox']:,.0f} people in {raster['cells']} cells; "
            f"{raster['people_without_buildings']:,.0f} in {raster['cells_without_buildings']} "
            f"cells without buildings {fate}",
            *([] if correction is None else [correction_summary(correction)]),
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
            *([] if check is None else check_summary(check)),
        ]
    )


def correction_summary(correction: dict) -> str:
    classes, folds = correction["classes"], correction["fold_variation"]
    low, high = correction["weight_range"]
    variation = (
        f"{folds['floor_weighted']:.3f} floor-weighted, max {folds['max']:.3f} over "
        f"{folds['classes']} classes holding {folds['floor_share']:.1%} of reference floor"
        if folds["classes"]
        else "not measurable, no class qualifies"
    )
    excluded = correction["excluded"]
    removed = (
        ""
        if excluded is None
        else f"{excluded['buildings']:,} non-residential buildings excluded, "
        f"{excluded['capacity_share']:.1%} of capacity; "
    )
    return (
        f"correction weight {correction['weight']} (range {low} to {high}): {removed}"
        f"{len(classes['used'])} classes, {len(classes['pooled'])} pooled, "
        f"{len(classes['capped'])} capped on {correction['reference_cells']:,} reference cells; "
        f"fold variation {variation}; {correction['people_moved']:,.0f} people moved"
    )


def check_summary(check: dict) -> list[str]:
    total = check["total"]
    low, high = total["band"]
    return [
        *(_group_line(name, group) for name, group in check["groups"].items()),
        f"check 48-ward total {total['people']:,.0f}, {total['ratio_to_2011']:.3f} x 2011 "
        f"(band {low} to {high}), raster {total['raster_ratio_to_2011']:.3f}; "
        f"{check['outside_wards_people']:,.0f} people outside the wards",
        *(
            f"check {name} {region['people']:,.0f} people "
            f"(band {region['band'][0]:,} to {region['band'][1]:,}), "
            f"raster {region['raster_people']:,.0f}"
            for name, region in check["regions"].items()
        ),
    ]


def _group_line(name: str, group: dict) -> str:
    band = "no band" if group["band"] is None else f"band {group['band'][0]} to {group['band'][1]}"
    misses = [
        f"{label} misses {', '.join(group[key])}"
        for label, key in (("known", "known_misses"), ("unexpected", "unexpected_misses"))
        if group[key]
    ]
    return "; ".join(
        [
            f"check {name} mean {group['mean_ratio']:.3f} of a 2011 ward ({band}), "
            f"raster {group['raster_mean_ratio']:.3f}",
            *misses,
        ]
    )


def refuse_failed_population(population: Path, city: str) -> None:
    """A failed check still writes its layers, for diagnosis; nothing may build on them."""
    report = population / "report.json"
    if not report.exists():
        return
    check = json.loads(report.read_text(encoding="utf-8")).get("check")
    if check is None or check["within_targets"][ALL_TARGETS]:
        return
    failed = [k for k, ok in check["within_targets"].items() if k != ALL_TARGETS and not ok]
    raise SystemExit(
        f"{population} failed its checks: {', '.join(failed)}; "
        f"resolve them and rerun `citybake population {city}` first"
    )


def run_personas_command(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    if config.personas is None:
        raise SystemExit(f"{city} has no [personas] section")
    population = dirs.stage(city, "population")
    refuse_failed_population(population, city)
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


def run_graph_command(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    population = dirs.stage(city, "population")
    refuse_failed_population(population, city)
    if not (population / "residents").exists():
        raise SystemExit(f"{population} is incomplete; run `citybake population {city}` first")
    personas = dirs.stage(city, "personas")
    needed = ("assignment", "pool", "vocab.json", "layer.json")
    if not all((personas / name).exists() for name in needed):
        raise SystemExit(f"{personas} is incomplete; run `citybake personas {city}` first")
    layers = building_layers(city, dirs)
    if not (layers / "buildings").exists():
        raise SystemExit(f"{layers} is missing; run `citybake osm {city}` first")
    try:
        graph = run_graph(population, personas, layers, config)
    except StalePersonasError as stale:
        raise SystemExit(str(stale)) from stale
    out = dirs.stage(city, "graph")
    write_graph_layers(graph, out)
    print(graph_summary(graph.report))
    print(out)


def graph_summary(report: dict) -> str:
    degree, clustering = report["degree"], report["clustering"]
    river, components = report["cross_river"], report["components"]
    by_kind = ", ".join(f"{name} {v}" for name, v in degree["mean_by_kind"].items())
    missed = [name for name, ok in report["within_targets"].items() if not ok]
    return "\n".join(
        [
            f"mean degree {degree['mean']} ({by_kind}), p10 {degree['p10']} p50 {degree['p50']} "
            f"p90 {degree['p90']} p99 {degree['p99']} max {degree['max']}, isolated "
            f"{degree['isolated_share']:.2%}",
            f"clustering {clustering['average_local']}, transitivity {clustering['transitivity']}",
            f"cross-river ties {river['all']:.2%}, non-household {river['non_household']:.2%}",
            f"components {components['count']:,}, largest {components['largest_share']:.2%}",
            f"targets {'all met' if not missed else 'missed: ' + ', '.join(missed)}",
        ]
    )


def run_stores_command(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    population = dirs.stage(city, "population")
    refuse_failed_population(population, city)
    if not _has_table(population / "residents"):
        raise SystemExit(f"{population} is incomplete; run `citybake population {city}` first")
    osm_layers = dirs.stage(city, "osm") / "layers"
    needed = ("road_nodes", "road_edges", "buildings")
    if not all(_has_table(osm_layers / name) for name in needed):
        raise SystemExit(f"{osm_layers} is incomplete; run `citybake osm {city}` first")
    layers = building_layers(city, dirs)
    if not _has_table(layers / "buildings") or not (layers / "districts.json").exists():
        rerun = "footprints" if layers.parent.name == "buildings" else "osm"
        raise SystemExit(f"{layers} is incomplete; run `citybake {rerun} {city}` first")
    try:
        stores = run_stores(population, osm_layers, layers, config)
    except StoresError as failure:
        raise SystemExit(str(failure)) from failure
    out = dirs.stage(city, "stores")
    write_store_layers(stores, out)
    print(stores_summary(stores.report))
    print(out)


def _has_table(path: Path) -> bool:
    if not (path / META).exists():
        return False
    columns = json.loads((path / META).read_text(encoding="utf-8"))["columns"]
    return all((path / f"{column['name']}.npy").exists() for column in columns)


def stores_summary(report: dict) -> str:
    stores, hexes, travel = report["stores"], report["hexes"], report["travel"]
    distance = hexes["node_distance_m"]
    bands = [
        f"  {name} p50 {b['p50']:g} p90 {b['p90']:g} max {b['max']:g}, within radius "
        f"p50 {b['within_radius_p50']:g} p90 {b['within_radius_p90']:g}"
        for name, b in travel["bands"].items()
    ]
    missed = [name for name, ok in report["within_targets"].items() if not ok]
    return "\n".join(
        [
            f"stores {stores['count']} cover {stores['coverage']:.1%} of resident weight within "
            f"{stores['radius_m']:g} m, west {stores['by_bank']['west']}, "
            f"east {stores['by_bank']['east']}",
            f"hexes {hexes['count']:,}, median {hexes['residents_per_hex_p50']:g} residents, road "
            f"node p50 {distance['p50']:g} m p90 {distance['p90']:g} m max {distance['max']:g} m",
            f"travel minutes by {travel['backend']}:",
            *bands,
            f"table {travel['table_bytes']:,} bytes, zstd {travel['table_zstd_bytes']:,}, in "
            f"radius {travel['table_in_radius_share']:.1%}, complete {travel['complete']}",
            f"road nodes {report['graph']['nodes']:,}, main component "
            f"{report['graph']['main_component_share']:.1%}",
            f"targets {'all met' if not missed else 'missed: ' + ', '.join(missed)}",
        ]
    )


def run_pack_command(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    population = dirs.stage(city, "population")
    refuse_failed_population(population, city)
    _require_stage(population, f"population {city}", ("residents",), ("layer.json",))
    osm_layers = dirs.stage(city, "osm") / "layers"
    _require_stage(
        osm_layers,
        f"osm {city}",
        ("road_geom", "road_edges", "road_nodes", "water"),
        ("layer.json",),
    )
    layers = building_layers(city, dirs)
    rerun = "footprints" if layers.parent.name == "buildings" else "osm"
    _require_stage(
        layers,
        f"{rerun} {city}",
        ("buildings", "building_rings"),
        ("districts.json", "layer.json"),
    )
    personas = dirs.stage(city, "personas")
    _require_stage(personas, f"personas {city}", ("assignment",), ("layer.json",))
    graph = dirs.stage(city, "graph")
    _require_stage(graph, f"graph {city}", ("graph",), ("layer.json",))
    stores = dirs.stage(city, "stores")
    _require_stage(
        stores,
        f"stores {city}",
        ("stores", "hexes", "resident_hex", "travel"),
        ("bands.json", "layer.json"),
    )
    wards = None
    if config.population_check is not None:
        wards = ward_file_path(city, config.population_check, dirs)
        if not wards.exists():
            raise SystemExit(f"{wards} is missing; run `citybake fetch {city}` first")
    out = dirs.stage(city, "pack")
    try:
        stage = run_pack(layers, osm_layers, population, personas, graph, stores, config, wards)
    except PackBudgetError as failure:
        write_json(out / "report.json", failure.report)
        raise SystemExit(f"{failure}; see {out / 'report.json'}") from failure
    except (PackStageError, PackError) as failure:
        raise SystemExit(str(failure)) from failure
    pack_dir = dirs.pack(city)
    write_pack_stage(stage, pack_dir, out)
    print(pack_summary(stage.report))
    print(pack_dir)


def _require_stage(folder: Path, command: str, tables: Sequence[str], files: Sequence[str]) -> None:
    complete = all(_has_table(folder / name) for name in tables) and all(
        (folder / name).exists() for name in files
    )
    if not complete:
        raise SystemExit(f"{folder} is incomplete; run `citybake {command}` first")


def pack_summary(report: dict) -> str:
    first_load, engine = report["sizes"]["first_load"], report["sizes"]["engine"]
    b, residents = report["buildings"], report["residents"]
    distance = residents["repoint_distance_m"]
    clipped = [f"{name} {count:,}" for name, count in report["clipped"].items() if count]
    return "\n".join(
        [
            f"first load {first_load['bytes']:,} bytes of the {first_load['budget']:,} budget",
            f"engine {engine['bytes']:,} bytes of the {engine['budget']:,} soft budget",
            f"buildings {b['baked']:,} baked: {b['first_load']:,} first load, {b['detail']:,} "
            f"detail, {b['in_water']:,} in water ({b['in_water_would_be_first_load']:,} would be "
            f"first load, {b['in_water_would_be_detail']:,} detail)",
            f"floor kept {report['floor_kept']:.1%} of baked floor area",
            *(
                f"{name} floor kept {group['floor_kept']:.1%}, {group['buildings_kept']:,} of "
                f"{group['buildings']:,} buildings"
                for name, group in report["ward_groups"].items()
            ),
            f"residents {residents['count']:,}: {residents['moved_out_of_water']:,} moved out of "
            f"water, {residents['floors_lowered']:,} on a lower floor, "
            f"{residents['repointed']:,} repointed (p50 {distance['p50']:g} m, "
            f"max {distance['max']:g} m)",
            *([f"clipped {', '.join(clipped)}"] if clipped else []),
            f"pack_hash {report['pack_hash']}",
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
    elif args.command == "graph":
        run_graph_command(args.city, dirs)
    elif args.command == "stores":
        run_stores_command(args.city, dirs)
    elif args.command == "pack":
        run_pack_command(args.city, dirs)
    return 0
