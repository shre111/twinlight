import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from twl_citybake.config import CityConfig, StoresConfig, TravelBand
from twl_citybake.projection import LocalProjection
from twl_citybake.rng import Stream, stream
from twl_citybake.store import read_table, table_digest, write_json, write_table
from twl_citybake.stores.check_pairs import Places, draw_check_pairs
from twl_citybake.stores.hexes import cells_at, hex_centres, hex_nodes, resident_hexes
from twl_citybake.stores.placement import (
    Placement,
    candidate_nodes,
    main_component,
    place_stores,
)
from twl_citybake.stores.report import (
    graph_section,
    hexes_section,
    rounded,
    stores_section,
    travel_section,
    within_targets,
)
from twl_citybake.stores.travel import GraphTravelBackend, TravelBackend

ATTRIBUTION = "Road network © OpenStreetMap contributors, ODbL 1.0"
STORES_SOURCE = "greedy max coverage, synthetic"
MAX_TABLE_MINUTES = int(np.iinfo(np.uint16).max)
# Dijkstra sums floats, so a route of exactly 5 minutes can arrive as 5.000000000000001.
ROUND_UP_TOLERANCE_MIN = 1e-9


class StoresError(ValueError):
    """A known condition that stops the stores stage."""


class UnreachablePairError(StoresError):
    """A store has no route to a hex; the travel table has no sentinel to hold that."""


@dataclass(frozen=True)
class StoreLayers:
    stores: dict[str, np.ndarray]
    hexes: dict[str, np.ndarray]
    resident_hex: dict[str, np.ndarray]
    travel: dict[str, np.ndarray]
    bands: list[dict[str, Any]]
    check_pairs: list[dict[str, Any]]
    report: dict[str, Any]
    meta: dict[str, Any]


@dataclass(frozen=True)
class HexGrid:
    """Hexes holding residents or stores, sorted by H3 id, with their dense ids per row."""

    h3: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    node: NDArray[np.uint32]
    node_distance_m: NDArray[np.float64]
    resident_hex: NDArray[np.uint32]
    store_hex: NDArray[np.uint32]


def run_stores(
    population: Path,
    osm_layers: Path,
    buildings_layers: Path,
    city: CityConfig,
    backend: TravelBackend | None = None,
) -> StoreLayers:
    """`population` is a stage folder; `osm_layers` holds the road tables, `buildings_layers` the
    `buildings` table and `districts.json`. Only the check pairs draw randomness; placement is a
    pure function of its inputs (D-013)."""
    residents = read_table(population / "residents")
    nodes = read_table(osm_layers / "road_nodes")
    edges = read_table(osm_layers / "road_edges")
    buildings = read_table(buildings_layers / "buildings")
    districts = json.loads((buildings_layers / "districts.json").read_text(encoding="utf-8"))
    in_component = main_component(nodes, edges)
    if not in_component.any():
        raise StoresError(f"{osm_layers / 'road_nodes'} holds no road nodes")

    config, travel = city.stores, city.travel
    placement = _place(residents, nodes, in_component, config)
    store_node = placement.node.astype(np.int64)
    store_x, store_y = nodes["x"][store_node], nodes["y"][store_node]
    proj = LocalProjection.for_bbox(city.bbox)
    grid = _hex_grid(residents, store_x, store_y, nodes, in_component, proj, travel.h3_resolution)

    backend = backend or GraphTravelBackend(len(nodes["x"]), edges, travel.speed_kmh)
    bands = travel.bands_by_start()
    table = travel_table(backend, store_node, grid.node.astype(np.int64), bands)

    distance_m = np.hypot(store_x[:, None] - grid.x[None, :], store_y[:, None] - grid.y[None, :])
    in_radius = distance_m <= config.radius_m
    residents_per_hex = np.bincount(grid.resident_hex, minlength=len(grid.h3))
    holds_residents = residents_per_hex[None, :] > 0
    store_lon, store_lat = proj.inverse(store_x, store_y)
    hex_lon, hex_lat = proj.inverse(grid.x, grid.y)
    nearest_building = _nearest(buildings["x"], buildings["y"], store_x, store_y)
    district = buildings["district"][nearest_building].astype(np.uint16)
    report = {
        "stores": stores_section(
            placement,
            config,
            district,
            buildings["bank"][nearest_building],
            [d["name"] for d in districts],
        ),
        "hexes": hexes_section(residents_per_hex, grid.node_distance_m),
        "travel": travel_section(
            table, bands, in_radius, holds_residents, backend.name, travel.speed_kmh
        ),
        "graph": graph_section(in_component),
        "within_targets": within_targets(placement, config.coverage_target, table),
    }
    return StoreLayers(
        stores={
            "node": placement.node,
            "x": placement.x,
            "y": placement.y,
            "lon": store_lon,
            "lat": store_lat,
            "hex": grid.store_hex,
            "radius_m": np.full(len(store_node), config.radius_m, np.float32),
            "district": district,
        },
        hexes={
            "h3": grid.h3,
            "x": grid.x.astype(np.float32),
            "y": grid.y.astype(np.float32),
            "node": grid.node,
            "node_distance_m": grid.node_distance_m.astype(np.float32),
        },
        resident_hex={"hex": grid.resident_hex},
        travel={"minutes": table.ravel()},
        bands=[asdict(band) for band in bands],
        check_pairs=draw_check_pairs(
            distance_m,
            holds_residents,
            table,
            bands,
            Places(store_lat, store_lon, hex_lat, hex_lon),
            stream(city.seed, Stream.STORES),
        ),
        report=rounded(report),
        meta={
            "city": city.slug,
            "seed": city.seed,
            "backend": backend.name,
            "h3_resolution": travel.h3_resolution,
            "population_residents_sha256": table_digest(population / "residents"),
            "osm_road_nodes_sha256": table_digest(osm_layers / "road_nodes"),
            "osm_road_edges_sha256": table_digest(osm_layers / "road_edges"),
            "buildings_stage": buildings_layers.parent.name,
            "buildings_sha256": table_digest(buildings_layers / "buildings"),
            "attribution": ATTRIBUTION,
            "stores_source": STORES_SOURCE,
            "travel_shape": list(table.shape),
            "travel_order": ["store", "hex", "band"],
            "stores": asdict(config),
            "travel": asdict(travel),
        },
    )


def write_store_layers(layers: StoreLayers, out_dir: Path) -> None:
    write_table(out_dir / "stores", layers.stores)
    write_table(out_dir / "hexes", layers.hexes)
    write_table(out_dir / "resident_hex", layers.resident_hex)
    write_table(out_dir / "travel", layers.travel)
    write_json(out_dir / "bands.json", layers.bands)
    write_json(out_dir / "check_pairs.json", layers.check_pairs)
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def travel_table(
    backend: TravelBackend,
    sources: NDArray[np.int64],
    targets: NDArray[np.int64],
    bands: Sequence[TravelBand],
) -> NDArray[np.uint16]:
    """Whole minutes shaped (sources, targets, bands), rounded up, at least 1."""
    table = np.zeros((len(sources), len(targets), len(bands)), np.uint16)
    if not table.size:
        return table
    for b, band in enumerate(bands):
        minutes = backend.minutes(sources, targets, band)
        unreachable = np.argwhere(~np.isfinite(minutes))
        if len(unreachable):
            s, h = unreachable[0]
            raise UnreachablePairError(f"store {s} cannot reach hex {h} in band {band.name}")
        whole = np.maximum(np.ceil(minutes - ROUND_UP_TOLERANCE_MIN), 1.0)
        too_long = np.argwhere(whole > MAX_TABLE_MINUTES)
        if len(too_long):
            s, h = too_long[0]
            raise StoresError(
                f"store {s} to hex {h} takes {whole[s, h]:.0f} minutes in band {band.name}, "
                f"more than the table's {MAX_TABLE_MINUTES}"
            )
        table[:, :, b] = whole
    return table


def _place(
    residents: Mapping[str, np.ndarray],
    nodes: Mapping[str, np.ndarray],
    in_component: NDArray[np.bool_],
    config: StoresConfig,
) -> Placement:
    candidates = candidate_nodes(nodes["x"], nodes["y"], in_component, config.candidate_cell_m)
    return place_stores(
        residents["x"],
        residents["y"],
        residents["weight"],
        nodes["x"][candidates],
        nodes["y"][candidates],
        candidates,
        config.radius_m,
        config.coverage_target,
        config.max_stores,
    )


def _hex_grid(
    residents: Mapping[str, np.ndarray],
    store_x: NDArray[np.floating],
    store_y: NDArray[np.floating],
    nodes: Mapping[str, np.ndarray],
    in_component: NDArray[np.bool_],
    proj: LocalProjection,
    resolution: int,
) -> HexGrid:
    resident_index, resident_cells = resident_hexes(
        residents["x"], residents["y"], proj, resolution
    )
    store_cells = cells_at(store_x, store_y, proj, resolution)
    h3_ids = np.union1d(resident_cells, store_cells)
    x, y = hex_centres(h3_ids, proj)
    node, distance = hex_nodes(h3_ids, proj, nodes["x"], nodes["y"], in_component)
    return HexGrid(
        h3=h3_ids,
        x=x,
        y=y,
        node=node,
        node_distance_m=distance,
        resident_hex=np.searchsorted(h3_ids, resident_cells)[resident_index].astype(np.uint32),
        store_hex=np.searchsorted(h3_ids, store_cells).astype(np.uint32),
    )


def _nearest(
    from_x: np.ndarray, from_y: np.ndarray, to_x: np.ndarray, to_y: np.ndarray
) -> NDArray[np.intp]:
    tree = KDTree(np.column_stack([from_x, from_y]))
    return np.asarray(tree.query(np.column_stack([to_x, to_y]))[1], np.intp)
