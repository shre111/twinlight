import argparse
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from twl_citybake.config import DataDirs, load_city
from twl_citybake.osm.clip import clip_extract
from twl_citybake.osm.extract import extract_osm, write_osm_layers
from twl_citybake.sources import fetch


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
    fetch_cmd = commands.add_parser("fetch", help="download the pinned OSM extract and clip it")
    fetch_cmd.add_argument("city")
    osm_cmd = commands.add_parser("osm", help="extract buildings, roads, water and districts")
    osm_cmd.add_argument("city")
    return parser


def run_fetch(city: str, dirs: DataDirs) -> None:
    config = load_city(city)
    region = fetch(config.osm.url, config.osm.md5, dirs.raw / "osm" / config.osm.filename)
    clipped = dirs.stage(city, "osm") / f"{city}.osm.pbf"
    clip_extract(region, clipped, config.bbox)
    print(f"{clipped} ({clipped.stat().st_size / 1e6:.1f} MB)")


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
            f"median {d['buildings_median']}, max {d['buildings_max']})",
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
    return 0
