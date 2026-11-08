import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import CityConfig, PackConfig
from twl_citybake.pack.rules import (
    ARTERIAL,
    BRIDGE,
    STREET,
    PackStageError,
    Rehomed,
    attributions,
    floors_from_z,
    footprint_polygons,
    in_water,
    isolated_small,
    pack_districts,
    pack_road_class,
    rehome_residents,
    river_columns,
    water_geometry,
)
from twl_citybake.population.check import assign_wards, load_wards, ward_key
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table, table_digest, write_json
from twl_citybake.stores.report import rounded
from twl_pack import (
    Band,
    BuildingColumns,
    EncodedPack,
    GraphColumns,
    PackContents,
    ResidentColumns,
    ResidentEngineColumns,
    RoadColumns,
    StoreColumns,
    encode_pack,
    write_pack,
)
from twl_pack.format import DETAIL_FILE

Table = Mapping[str, np.ndarray]


class StaleInputError(PackStageError):
    """A stage was baked from other tables than the current ones."""


class PackBudgetError(PackStageError):
    """The first-load download is over its budget. `report` is the stage report, for tuning."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, eq=False)
class PackStage:
    encoded: EncodedPack
    report: dict[str, Any]
    meta: dict[str, Any]


@dataclass(frozen=True, eq=False)
class _Applied:
    """Masks over all baked buildings, and the residents after the rule."""

    isolated: NDArray[np.bool_]
    wet: NDArray[np.bool_]
    kept: NDArray[np.bool_]
    detail: NDArray[np.bool_]
    rehomed: Rehomed


def run_pack(
    buildings_layers: Path,
    osm_layers: Path,
    population: Path,
    personas: Path,
    graph: Path,
    stores: Path,
    city: CityConfig,
    wards: Path | None = None,
) -> PackStage:
    """The folders are stage outputs. `wards` is the ward file behind the report's ward groups;
    ward data goes in `report.json` only. Nothing here draws randomness."""
    if city.timezone is None or city.weekend_days is None:
        raise PackStageError(f"{city.slug} needs timezone and weekend_days in its city config")
    digests = _input_digests(buildings_layers, osm_layers, population, personas, graph, stores)
    _check_current("personas", personas, digests, ("population_residents_sha256",))
    _check_current(
        "graph",
        graph,
        digests,
        ("population_residents_sha256", "persona_assignment_sha256", "buildings_sha256"),
    )
    _check_current(
        "stores",
        stores,
        digests,
        (
            "population_residents_sha256",
            "buildings_sha256",
            "osm_road_nodes_sha256",
            "osm_road_edges_sha256",
        ),
    )

    residents = read_table(population / "residents")
    if "tier" not in residents:
        raise StaleInputError(f"{population} has no tier column; rerun `citybake population`")
    assignment = read_table(personas / "assignment")
    resident_hex = read_table(stores / "resident_hex")
    _check_row_counts(residents, assignment, resident_hex)
    buildings = read_table(buildings_layers / "buildings")
    rings = read_table(buildings_layers / "building_rings")
    water = read_table(osm_layers / "water")
    applied = _apply_rules(buildings, rings, residents, water, city.pack)

    edges = read_table(osm_layers / "road_edges")
    road_class = pack_road_class(edges["road_class"], edges["bridge"], city.pack.arterial_classes)
    persona_meta = _layer_meta(personas)
    credits = attributions(
        _layer_meta(osm_layers),
        _layer_meta(buildings_layers),
        _layer_meta(population),
        persona_meta,
        _layer_meta(stores),
    )
    encoded = encode_pack(
        PackContents(
            kind="city",
            name=city.slug,
            crs=LocalProjection.for_bbox(city.bbox).proj4,
            timezone=city.timezone,
            weekend_days=city.weekend_days,
            segments=list(persona_meta["segments"]),
            bands=_bands(stores),
            attributions=credits,
            districts=pack_districts(_read_json(buildings_layers / "districts.json")),
            buildings=_building_columns(buildings, applied.kept),
            detail=_building_columns(buildings, applied.detail),
            roads=_road_columns(read_table(osm_layers / "road_geom"), road_class),
            residents=_resident_columns(applied, assignment),
            residents_engine=ResidentEngineColumns(
                hex=resident_hex["hex"],
                persona_ref=assignment["persona_ref"],
                weight=residents["weight"],
                tier=residents["tier"],
            ),
            stores=_store_columns(read_table(stores / "stores")),
            river=river_columns(water),
            graph=_graph_columns(read_table(graph / "graph")),
            hexes=read_table(stores / "hexes")["h3"],
            travel=read_table(stores / "travel")["minutes"],
            tile_level=city.pack.tile_level,
        )
    )
    sizes = json.loads(encoded.manifest)["sizes"]
    report = {
        "pack_hash": encoded.pack_hash,
        "sizes": sizes,
        "within_budget": {name: _within(sizes[name]) for name in ("first_load", "engine")},
        **_rule_report(buildings, applied, city.pack, road_class),
        "ward_groups": _ward_groups(buildings, applied.kept, city, wards),
        "clipped": encoded.clipped,
    }
    _check_budgets(sizes, rounded(report))
    return PackStage(
        encoded=encoded,
        report=rounded(report),
        meta={
            "city": city.slug,
            "seed": city.seed,
            "pack_hash": encoded.pack_hash,
            "pack": asdict(city.pack),
            "buildings_stage": buildings_layers.parent.name,
            "attribution": sorted(set(credits)),
            **digests,
        },
    )


def write_pack_stage(stage: PackStage, pack_dir: Path, out_dir: Path) -> None:
    write_pack(stage.encoded, pack_dir)
    if DETAIL_FILE not in stage.encoded.objects:
        (pack_dir / DETAIL_FILE).unlink(missing_ok=True)
    write_json(out_dir / "report.json", stage.report)
    write_json(out_dir / "layer.json", stage.meta)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_meta(folder: Path) -> dict[str, Any]:
    return _read_json(folder / "layer.json")


def _input_digests(
    buildings_layers: Path,
    osm_layers: Path,
    population: Path,
    personas: Path,
    graph: Path,
    stores: Path,
) -> dict[str, str]:
    return {
        "buildings_sha256": table_digest(buildings_layers / "buildings"),
        "building_rings_sha256": table_digest(buildings_layers / "building_rings"),
        "osm_road_geom_sha256": table_digest(osm_layers / "road_geom"),
        "osm_road_edges_sha256": table_digest(osm_layers / "road_edges"),
        "osm_water_sha256": table_digest(osm_layers / "water"),
        "population_residents_sha256": table_digest(population / "residents"),
        "persona_assignment_sha256": table_digest(personas / "assignment"),
        "graph_sha256": table_digest(graph / "graph"),
        "stores_sha256": table_digest(stores / "stores"),
        "hexes_sha256": table_digest(stores / "hexes"),
        "resident_hex_sha256": table_digest(stores / "resident_hex"),
        "travel_sha256": table_digest(stores / "travel"),
        # Road nodes are not a pack input, but stores stand on them.
        "osm_road_nodes_sha256": table_digest(osm_layers / "road_nodes"),
    }


def _check_current(
    stage: str, folder: Path, current: Mapping[str, str], required: Sequence[str]
) -> None:
    meta = _layer_meta(folder)
    for key in required:
        if meta.get(key) != current.get(key):
            raise StaleInputError(
                f"{folder} is stale: its {key} does not match the current table; "
                f"rerun `citybake {stage}`"
            )


def _check_row_counts(residents: Table, assignment: Table, resident_hex: Table) -> None:
    counts = {
        "residents": len(residents["x"]),
        "assignment": len(assignment["segment"]),
        "resident_hex": len(resident_hex["hex"]),
    }
    if len(set(counts.values())) > 1:
        raise PackStageError(f"the resident tables disagree on their row counts: {counts}")


def _apply_rules(
    buildings: Table, rings: Table, residents: Table, water: Table, config: PackConfig
) -> _Applied:
    if len(rings["offsets"]) - 1 != len(buildings["x"]):
        raise PackStageError(
            f"{len(rings['offsets']) - 1} building rings for {len(buildings['x'])} buildings"
        )
    isolated = isolated_small(
        footprint_polygons(rings["offsets"], rings["xy"]),
        buildings["area_m2"],
        config.min_footprint_m2,
        config.isolated_m,
    )
    water_shape = water_geometry(water)
    wet = in_water(water_shape, buildings["x"], buildings["y"])
    kept = ~isolated & ~wet
    rehomed = rehome_residents(
        residents["building"],
        floors_from_z(residents["z"]),
        residents["x"],
        residents["y"],
        in_water(water_shape, residents["x"], residents["y"]),
        kept,
        buildings["x"],
        buildings["y"],
        buildings["height_m"],
    )
    return _Applied(isolated, wet, kept, isolated & ~wet, rehomed)


def _bands(stores: Path) -> list[Band]:
    return [Band(band["name"], band["start_minute"]) for band in _read_json(stores / "bands.json")]


def _building_columns(buildings: Table, mask: NDArray[np.bool_]) -> BuildingColumns:
    return BuildingColumns(
        x=buildings["x"][mask],
        y=buildings["y"][mask],
        width=buildings["w"][mask],
        depth=buildings["d"][mask],
        height=buildings["height_m"][mask],
        rotation=buildings["rot"][mask],
        district=buildings["district"][mask],
    )


def _road_columns(road_geom: Table, road_class: NDArray[np.uint8]) -> RoadColumns:
    return RoadColumns(
        points=np.diff(road_geom["offsets"].astype(np.int64)),
        road_class=road_class,
        x=road_geom["xy"][:, 0],
        y=road_geom["xy"][:, 1],
    )


def _resident_columns(applied: _Applied, assignment: Table) -> ResidentColumns:
    kept_row = np.cumsum(applied.kept, dtype=np.int64) - 1
    return ResidentColumns(
        x=applied.rehomed.x,
        y=applied.rehomed.y,
        floor=applied.rehomed.floor,
        building=kept_row[applied.rehomed.building],
        segment=assignment["segment"],
    )


def _store_columns(stores: Table) -> StoreColumns:
    return StoreColumns(
        x=stores["x"],
        y=stores["y"],
        hex=stores["hex"],
        radius_m=stores["radius_m"],
        district=stores["district"],
    )


def _graph_columns(graph: Table) -> GraphColumns:
    return GraphColumns(
        offsets=graph["offsets"],
        targets=graph["targets"],
        kind=graph["kind"],
        strength=graph["strength"],
    )


def _check_budgets(sizes: Mapping[str, Mapping[str, Any]], report: dict[str, Any]) -> None:
    """The engine budget is soft: reported, never raised."""
    first_load = sizes["first_load"]
    if not _within(first_load):
        used, budget = first_load["bytes"], first_load["budget"]
        raise PackBudgetError(f"first load is {used:,} bytes, over its {budget:,} budget", report)


def _within(size: Mapping[str, Any]) -> bool:
    return bool(size["bytes"] <= size["budget"])


def _rule_report(
    buildings: Table, applied: _Applied, config: PackConfig, road_class: NDArray[np.uint8]
) -> dict[str, Any]:
    isolated, wet, kept = applied.isolated, applied.wet, applied.kept
    rehomed = applied.rehomed
    area = buildings["area_m2"].astype(np.float64)
    floor_area = _floor_area(buildings)
    return {
        "buildings": {
            "baked": len(kept),
            "first_load": _count(kept),
            "detail": _count(applied.detail),
            "in_water": _count(wet),
            "in_water_would_be_first_load": _count(wet & ~isolated),
            "in_water_would_be_detail": _count(wet & isolated),
            "isolated_small": _count(isolated),
            "small": _count(area < config.min_footprint_m2),
        },
        "floor_kept": _share(floor_area[kept], floor_area),
        "residents": {
            "count": len(rehomed.building),
            "moved_out_of_water": rehomed.moved,
            "floors_lowered": rehomed.floors_lowered,
            "repointed": rehomed.repointed,
            "repoint_distance_m": _distance_summary(rehomed.repoint_distance_m),
        },
        "roads": {
            "count": len(road_class),
            "arterial": _count(road_class == ARTERIAL),
            "bridge": _count(road_class == BRIDGE),
            "street": _count(road_class == STREET),
        },
    }


def _floor_area(buildings: Table) -> NDArray[np.float64]:
    return buildings["area_m2"].astype(np.float64) * buildings["floors"]


def _count(mask: NDArray[np.bool_]) -> int:
    return int(np.count_nonzero(mask))


def _share(part: NDArray[np.float64], whole: NDArray[np.float64]) -> float:
    """0.0 over nothing, never NaN."""
    total = float(whole.sum())
    return float(part.sum()) / total if total > 0 else 0.0


def _distance_summary(distance_m: NDArray[np.float64]) -> dict[str, float]:
    if not distance_m.size:
        return {"p50": 0.0, "max": 0.0}
    return {"p50": float(np.percentile(distance_m, 50)), "max": float(distance_m.max())}


def _ward_groups(
    buildings: Table, kept: NDArray[np.bool_], city: CityConfig, wards_path: Path | None
) -> dict[str, dict[str, Any]]:
    check = city.population_check
    if wards_path is None or check is None:
        return {}
    wards = load_wards(wards_path, check.name_property)
    lon, lat = LocalProjection.for_bbox(city.bbox).inverse(buildings["x"], buildings["y"])
    owner = assign_wards(wards, lon, lat)
    index_of: dict[str, int] = {}
    for index, ward in enumerate(wards):
        index_of.setdefault(ward_key(ward.name), index)
    floor_area = _floor_area(buildings)
    groups: dict[str, dict[str, Any]] = {}
    for group in check.ward_groups:
        if missing := [ward for ward in group.wards if ward_key(ward) not in index_of]:
            raise PackStageError(f"ward group {group.name}: {missing[0]!r} is not in {wards_path}")
        members = np.isin(owner, [index_of[ward_key(ward)] for ward in group.wards])
        groups[group.name] = {
            "buildings": _count(members),
            "buildings_kept": _count(members & kept),
            "floor_kept": _share(floor_area[members & kept], floor_area[members]),
        }
    return groups
