from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import CityConfig, DataDirs, load_city
from twl_citybake.osm.tags import BuildingKind
from twl_citybake.store import table_digest, write_json, write_table

KIND_NAMES = ("household", "neighbour", "friend", "work")
HOUSEHOLD, NEIGHBOUR, FRIEND, WORK = range(4)
NONE = 0xFFFFFFFF

Triple = tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.float32]]


def triple(pairs: Sequence[tuple[int, int, float]]) -> Triple:
    return (
        np.array([p[0] for p in pairs], np.uint32),
        np.array([p[1] for p in pairs], np.uint32),
        np.array([p[2] for p in pairs], np.float32),
    )


def merge(per_kind: Mapping[int, Triple], n: int) -> Any:
    from twl_citybake.graph.ties import merge_ties

    return merge_ties([per_kind.get(k, triple([])) for k in range(4)], n)


def build_graph(ties: Sequence[tuple[int, int, int, float]], n: int) -> Any:
    """Ties as (u, v, kind, strength)."""
    return merge({k: triple([(u, v, s) for u, v, kk, s in ties if kk == k]) for k in range(4)}, n)


def edges(graph: Any) -> dict[tuple[int, int], tuple[int, float]]:
    offsets, targets = graph.offsets.astype(np.int64), graph.targets
    out: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(len(offsets) - 1):
        for j in range(offsets[i], offsets[i + 1]):
            out[(i, int(targets[j]))] = (int(graph.kind[j]), float(graph.strength[j]))
    return out


def assert_csr_invariants(graph: Any, n: int) -> None:
    offsets, targets = graph.offsets, graph.targets
    assert offsets.dtype == np.uint32 and targets.dtype == np.uint32
    assert graph.kind.dtype == np.uint8 and graph.strength.dtype == np.float32
    assert len(offsets) == n + 1 and offsets[0] == 0 and offsets[-1] == len(targets)
    assert (np.diff(offsets.astype(np.int64)) >= 0).all()
    assert len(graph.kind) == len(targets) == len(graph.strength)
    if len(targets):
        assert int(targets.max()) < n
        assert (graph.strength > 0).all() and (graph.strength <= 1).all()
        assert int(graph.kind.max()) <= WORK
    source = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets.astype(np.int64)))
    assert (source != targets.astype(np.int64)).all()
    key = source * n + targets.astype(np.int64)
    assert (np.diff(key) > 0).all()
    table = edges(graph)
    for (a, b), (kind, strength) in table.items():
        assert table[(b, a)] == (kind, strength)


def pair_distance(
    x: NDArray[np.float64], y: NDArray[np.float64], u: NDArray[np.uint32], v: NDArray[np.uint32]
) -> NDArray[np.float64]:
    return np.hypot(x[u] - x[v], y[u] - y[v])


@dataclass(frozen=True)
class World:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    household: NDArray[np.uint32]
    segment: NDArray[np.uint16]

    @property
    def n(self) -> int:
        return len(self.x)


def uniform_world(
    n: int,
    seed: int = 7,
    extent_m: float = 3000.0,
    household_size: int = 1,
    shares: Sequence[float] = (0.5, 0.3, 0.2),
) -> World:
    rng = np.random.default_rng(seed)
    households = -(-n // household_size)
    hx = rng.uniform(0, extent_m, households)
    hy = rng.uniform(0, extent_m, households)
    household = (np.arange(n) // household_size).astype(np.uint32)
    segment = rng.choice(len(shares), n, p=list(shares)).astype(np.uint16)
    return World(hx[household], hy[household], household, segment)


def building_table(
    rows: Sequence[tuple[float, float, int, float, int, int, int]],
) -> dict[str, NDArray[Any]]:
    """Rows as (x, y, kind, area_m2, floors, district, bank)."""
    cols = list(zip(*rows, strict=True)) if rows else [[] for _ in range(7)]
    return {
        "x": np.array(cols[0], np.float64),
        "y": np.array(cols[1], np.float64),
        "kind": np.array(cols[2], np.uint8),
        "area_m2": np.array(cols[3], np.float32),
        "floors": np.array(cols[4], np.uint8),
        "height_m": np.array(cols[4], np.float32) * np.float32(3.2),
        "district": np.array(cols[5], np.uint16),
        "bank": np.array(cols[6], np.uint8),
    }


def run_work(
    world: World,
    occupied: NDArray[np.bool_],
    buildings: Mapping[str, NDArray[Any]],
    seed: int = 1,
    degree: float = 1.0,
    scale_m: float = 5000.0,
    cell_m: float = 500.0,
    candidates: int = 16,
    min_area_m2: float = 1000.0,
    max_floors: int = 3,
    district_names: Sequence[str] = ("Alpha", "Beta"),
) -> tuple[Triple, dict[str, Any]]:
    from twl_citybake.graph.ties import work_ties

    return work_ties(
        world.x,
        world.y,
        world.household,
        occupied,
        buildings,
        degree,
        scale_m,
        cell_m,
        candidates,
        min_area_m2,
        max_floors,
        np.random.default_rng(seed),
        district_names=list(district_names),
    )


def graph_report(graph: Any, bank: Any, segment: Any, x: Any, y: Any) -> dict[str, Any]:
    from twl_citybake.graph.metrics import graph_report as report

    n = len(graph.offsets) - 1
    return report(
        graph,
        np.asarray(bank, np.uint8),
        np.asarray(segment, np.uint16),
        np.asarray(x, np.float64),
        np.asarray(y, np.float64),
        n,
    )


@dataclass(frozen=True)
class Built:
    city: CityConfig
    dirs: DataDirs
    population: Path
    personas: Path
    buildings: Path
    residents: int


def small_city(seed: int = 20260101) -> CityConfig:
    return replace(load_city("ahmedabad"), seed=seed)


def build_inputs(root: Path, residents: int = 3000, seed: int = 4) -> Built:
    """Population, personas and OSM building stages for a small two-bank world, written as the
    earlier stages write them."""
    city = small_city()
    assert city.personas is not None
    dirs = DataDirs(root)
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 2, 3, 4], residents // 2, p=[0.15, 0.35, 0.3, 0.2])
    sizes = sizes[: int(np.searchsorted(np.cumsum(sizes), residents)) + 1]
    n = int(sizes.sum())
    household = np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes)
    hx = rng.uniform(-3000, 3000, len(sizes))
    hy = rng.uniform(-3000, 3000, len(sizes))
    x, y = hx[household], hy[household]
    bank = (x > 0).astype(np.uint8)
    district = np.clip((x // 1500 + 2).astype(np.int64), 0, 3).astype(np.uint16)

    population = dirs.stage(city.slug, "population")
    write_table(
        population / "residents",
        {
            "household": household,
            "building": np.zeros(n, np.uint32),
            "district": district,
            "bank": bank,
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "z": np.zeros(n, np.float32),
            "weight": np.ones(n, np.float32),
        },
    )

    pool_n = 60
    vocab = {
        "occupation": sorted(["Driver", "Teacher", "Clerk", city.personas.no_occupation]),
        "segment": ["a", "b", "c"],
    }
    occupation = rng.integers(0, 4, pool_n).astype(np.uint16)
    pool_segment = rng.integers(0, 3, pool_n).astype(np.uint16)
    persona_ref = rng.integers(0, pool_n, n).astype(np.uint32)
    spouse = np.full(n, NONE, np.uint32)
    first = np.flatnonzero(np.r_[True, household[1:] != household[:-1]])
    for start, size in zip(first, sizes, strict=True):
        if size >= 2:
            spouse[start], spouse[start + 1] = start + 1, start
    personas = dirs.stage(city.slug, "personas")
    write_table(
        personas / "pool",
        {
            "uuid": np.array([f"p{i}" for i in range(pool_n)]),
            "occupation": occupation,
            "segment": pool_segment,
        },
    )
    write_table(
        personas / "assignment",
        {"persona_ref": persona_ref, "segment": pool_segment[persona_ref], "spouse": spouse},
    )
    write_json(personas / "vocab.json", vocab)
    write_json(
        personas / "layer.json",
        {
            "city": city.slug,
            "seed": city.seed,
            "population_residents_sha256": table_digest(population / "residents"),
        },
    )

    nb = 250
    bx = rng.uniform(-3000, 3000, nb)
    kinds = rng.choice(
        [
            BuildingKind.RESIDENTIAL,
            BuildingKind.COMMERCIAL,
            BuildingKind.INDUSTRIAL,
            BuildingKind.CIVIC,
            BuildingKind.UNKNOWN,
        ],
        nb,
    )
    floors = rng.integers(1, 6, nb).astype(np.uint8)
    buildings = dirs.stage(city.slug, "osm") / "layers"
    write_table(
        buildings / "buildings",
        {
            "x": bx,
            "y": rng.uniform(-3000, 3000, nb),
            "area_m2": rng.uniform(100, 3000, nb).astype(np.float32),
            "height_m": floors.astype(np.float32) * np.float32(3.2),
            "floors": floors,
            "kind": kinds.astype(np.uint8),
            "district": np.clip((bx // 1500 + 2).astype(np.int64), 0, 3).astype(np.uint16),
            "bank": (bx > 0).astype(np.uint8),
        },
    )
    write_json(
        buildings / "districts.json",
        [{"id": i, "name": f"District {i}"} for i in range(4)],
    )
    return Built(city, dirs, population, personas, buildings, n)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }
