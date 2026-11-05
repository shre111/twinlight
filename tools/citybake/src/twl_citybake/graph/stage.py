import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import CityConfig, GraphConfig
from twl_citybake.graph.metrics import graph_report, within_targets
from twl_citybake.graph.ties import (
    TIE_KINDS,
    Graph,
    Ties,
    friend_ties,
    household_ties,
    merge_ties,
    neighbour_ties,
    work_ties,
)
from twl_citybake.rng import Stream, stream
from twl_citybake.store import read_table, table_digest, write_json, write_ragged

REPORT_DECIMALS = 4


class StalePersonasError(ValueError):
    """The persona assignment was made for other residents than the current ones."""


@dataclass(frozen=True)
class GraphLayers:
    graph: Graph
    report: dict[str, Any]
    meta: dict[str, Any]


def run_graph(population: Path, personas: Path, buildings: Path, city: CityConfig) -> GraphLayers:
    """`population` and `personas` are stage folders, `buildings` a layers folder holding the
    `buildings` table and `districts.json`. One GRAPH stream spawns one child per drawn kind, so a
    kind's parameters cannot disturb another kind's draws."""
    residents_digest = _check_personas_current(population, personas)
    residents = read_table(population / "residents")
    assignment = read_table(personas / "assignment")
    n = len(residents["household"])
    if len(assignment["spouse"]) != n:
        raise ValueError(f"{len(assignment['spouse'])} persona rows for {n} residents")
    config = city.graph
    x, y, household = residents["x"], residents["y"], residents["household"]
    neighbour_rng, friend_rng, work_rng = stream(city.seed, Stream.GRAPH).spawn(3)
    neighbour = neighbour_ties(
        x,
        y,
        household,
        config.neighbour_degree,
        config.neighbour_scale_m,
        config.neighbour_candidates,
        neighbour_rng,
    )
    friend = friend_ties(
        x,
        y,
        household,
        assignment["segment"],
        config.friend_degree,
        config.friend_scale_m,
        config.friend_homophily,
        config.friend_candidates,
        friend_rng,
    )
    work, workplaces = _work_ties(
        residents, _occupied(personas, assignment, city), buildings, config, work_rng
    )
    graph = merge_ties(
        (household_ties(household, assignment["spouse"]), neighbour, friend, work), n
    )
    report = graph_report(graph, residents["bank"], assignment["segment"], x, y, n)
    report["workplaces"] = workplaces
    report["within_targets"] = within_targets(report, config.targets)
    return GraphLayers(
        graph=graph,
        report=_rounded(report),
        meta={
            "city": city.slug,
            "seed": city.seed,
            "tie_kinds": list(TIE_KINDS),
            "population_residents_sha256": residents_digest,
            "persona_assignment_sha256": table_digest(personas / "assignment"),
            "buildings_stage": buildings.parent.name,
            "buildings_sha256": table_digest(buildings / "buildings"),
            "graph": asdict(config),
        },
    )


def write_graph_layers(layers: GraphLayers, out_dir: Path) -> None:
    graph = layers.graph
    write_ragged(
        out_dir / "graph",
        {
            "offsets": graph.offsets,
            "targets": graph.targets,
            "kind": graph.kind,
            "strength": graph.strength,
        },
        rows=len(graph.offsets) - 1,
    )
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def _check_personas_current(population: Path, personas: Path) -> str:
    current = table_digest(population / "residents")
    meta = json.loads((personas / "layer.json").read_text(encoding="utf-8"))
    if meta.get("population_residents_sha256") != current:
        raise StalePersonasError(
            f"{personas} is stale: its assignment predates {population / 'residents'}; "
            "rerun `citybake personas`"
        )
    return current


def _occupied(
    personas: Path, assignment: dict[str, np.ndarray], city: CityConfig
) -> NDArray[np.bool_]:
    if city.personas is None:
        raise ValueError(f"{city.slug} has no [personas] section")
    vocabulary = json.loads((personas / "vocab.json").read_text(encoding="utf-8"))["occupation"]
    if city.personas.no_occupation not in vocabulary:
        raise ValueError(f"the pool has no occupation {city.personas.no_occupation!r}")
    occupation = read_table(personas / "pool")["occupation"]
    return occupation[assignment["persona_ref"]] != vocabulary.index(city.personas.no_occupation)


def _work_ties(
    residents: dict[str, np.ndarray],
    occupied: NDArray[np.bool_],
    buildings: Path,
    config: GraphConfig,
    rng: np.random.Generator,
) -> tuple[Ties, dict[str, Any]]:
    districts = json.loads((buildings / "districts.json").read_text(encoding="utf-8"))
    return work_ties(
        residents["x"],
        residents["y"],
        residents["household"],
        occupied,
        read_table(buildings / "buildings"),
        config.work_degree,
        config.work_scale_m,
        config.work_cell_m,
        config.workplace_candidates,
        config.workplace_min_area_m2,
        config.workplace_max_floors,
        rng,
        district_names=[d["name"] for d in districts],
    )


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return round(value, REPORT_DECIMALS) if isinstance(value, float) else value
