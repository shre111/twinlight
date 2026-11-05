import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from graph_fixture import (
    FRIEND,
    HOUSEHOLD,
    KIND_NAMES,
    NEIGHBOUR,
    WORK,
    Built,
    assert_csr_invariants,
    build_inputs,
    snapshot,
)

from twl_citybake.cli import main
from twl_citybake.config import CityConfig, GraphTargets
from twl_citybake.graph.stage import (
    GraphLayers,
    StalePersonasError,
    run_graph,
    write_graph_layers,
)
from twl_citybake.store import read_table, table_digest, write_table

TARGET_NAMES = {
    "mean_degree",
    "clustering",
    "cross_river_non_household",
    "largest_component_min",
    "isolated_max",
}
PERMISSIVE = GraphTargets(
    mean_degree=(0.0, 1000.0),
    clustering=(0.0, 1.0),
    cross_river_non_household=(0.0, 1.0),
    largest_component_min=0.0,
    isolated_max=1.0,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_inputs(tmp_path_factory.mktemp("graph-stage"))


def with_graph(city: CityConfig, **changes: object) -> CityConfig:
    return replace(city, graph=replace(city.graph, **changes))


def bake(built: Built, city: CityConfig | None = None) -> GraphLayers:
    return run_graph(built.population, built.personas, built.buildings, city or built.city)


def written(layers: GraphLayers, out: Path) -> Path:
    write_graph_layers(layers, out)
    return out


@pytest.fixture(scope="module")
def out(built: Built, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return written(bake(built), tmp_path_factory.mktemp("graph-out") / "graph")


def table_of(out: Path) -> dict[str, np.ndarray]:
    return read_table(out / "graph")


def kind_rows(table: dict[str, np.ndarray], kind: int) -> list[tuple[int, int, float]]:
    offsets = table["offsets"].astype(np.int64)
    source = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    pick = table["kind"] == kind
    return sorted(
        zip(
            source[pick].tolist(),
            table["targets"][pick].tolist(),
            table["strength"][pick].tolist(),
            strict=True,
        )
    )


def test_written_graph_is_a_csr_with_one_row_per_resident(built: Built, out: Path) -> None:
    table = table_of(out)
    assert set(table) == {"offsets", "targets", "kind", "strength"}
    meta = json.loads((out / "graph" / "_table.json").read_text())
    assert meta["rows"] == built.residents

    class Csr:
        offsets, targets, kind, strength = (
            table[k] for k in ("offsets", "targets", "kind", "strength")
        )

    assert_csr_invariants(Csr, built.residents)
    assert len(table["targets"]) > 0
    assert set(np.unique(table["kind"]).tolist()) == {HOUSEHOLD, NEIGHBOUR, FRIEND, WORK}


def test_household_members_are_tied_as_a_clique_in_the_written_graph(
    built: Built, out: Path
) -> None:
    household = read_table(built.population / "residents")["household"]
    table = table_of(out)
    rows = kind_rows(table, HOUSEHOLD)
    sizes = np.bincount(household).astype(np.int64)
    assert len(rows) == int((sizes * (sizes - 1)).sum())
    assert all(household[a] == household[b] for a, b, _ in rows)


def test_spouses_are_tied_at_full_strength_and_other_household_members_below(
    built: Built, out: Path
) -> None:
    spouse = read_table(built.personas / "assignment")["spouse"]
    rows = kind_rows(table_of(out), HOUSEHOLD)
    for a, b, w in rows:
        assert w == (1.0 if spouse[a] == b else pytest.approx(0.8, abs=1e-6))


def test_work_ties_only_join_residents_with_an_occupation(built: Built, out: Path) -> None:
    pool = read_table(built.personas / "pool")
    assignment = read_table(built.personas / "assignment")
    vocab = json.loads((built.personas / "vocab.json").read_text())
    none = vocab["occupation"].index(built.city.personas.no_occupation)  # type: ignore[union-attr]
    occupied = pool["occupation"][assignment["persona_ref"]] != none
    assert 0.2 < occupied.mean() < 0.9
    rows = kind_rows(table_of(out), WORK)
    assert rows
    assert all(occupied[a] and occupied[b] for a, b, _ in rows)


def test_nobody_occupied_means_no_work_ties(built: Built, tmp_path: Path) -> None:
    assert built.city.personas is not None
    vocab = json.loads((built.personas / "vocab.json").read_text())
    none = vocab["occupation"].index(built.city.personas.no_occupation)
    pool = read_table(built.personas / "pool")
    other = build_inputs(tmp_path)
    write_table(
        other.personas / "pool",
        {**pool, "occupation": np.full(len(pool["occupation"]), none, np.uint16)},
    )
    table = table_of(written(bake(other), tmp_path / "out"))
    assert not (table["kind"] == WORK).any()


def test_same_seed_writes_byte_identical_layers(built: Built, out: Path, tmp_path: Path) -> None:
    assert snapshot(written(bake(built), tmp_path / "again")) == snapshot(out)


def test_a_different_seed_writes_different_ties(built: Built, out: Path, tmp_path: Path) -> None:
    other = written(bake(built, replace(built.city, seed=7)), tmp_path / "other")
    a, b = table_of(out), table_of(other)
    assert not (
        np.array_equal(a["targets"], b["targets"]) and np.array_equal(a["strength"], b["strength"])
    )


def test_changing_friend_degree_leaves_household_and_neighbour_ties_unchanged(
    built: Built, out: Path, tmp_path: Path
) -> None:
    changed = with_graph(built.city, friend_degree=built.city.graph.friend_degree * 3)
    other = table_of(written(bake(built, changed), tmp_path / "friends"))
    base = table_of(out)
    assert kind_rows(other, FRIEND) != kind_rows(base, FRIEND)
    assert len(kind_rows(other, FRIEND)) > len(kind_rows(base, FRIEND))
    for kind in (HOUSEHOLD, NEIGHBOUR):
        assert kind_rows(other, kind) == kind_rows(base, kind)


def test_changing_work_degree_leaves_neighbour_and_friend_ties_unchanged(
    built: Built, out: Path, tmp_path: Path
) -> None:
    changed = with_graph(built.city, work_degree=built.city.graph.work_degree * 3)
    other = table_of(written(bake(built, changed), tmp_path / "work"))
    base = table_of(out)
    assert len(kind_rows(other, WORK)) > len(kind_rows(base, WORK))
    for kind in (HOUSEHOLD, NEIGHBOUR, FRIEND):
        assert kind_rows(other, kind) == kind_rows(base, kind)


def assert_draws_unmoved(
    base: dict[str, np.ndarray], other: dict[str, np.ndarray], kind: int, stronger: tuple[int, ...]
) -> None:
    """Rows of `kind` agree, except where a stronger kind took the pair in one of the two bakes:
    merge_ties keeps only the lowest kind, so those rows differ without any draw having moved."""
    taken = {(a, b) for table in (base, other) for k in stronger for a, b, _ in kind_rows(table, k)}
    base_rows, other_rows = set(kind_rows(base, kind)), set(kind_rows(other, kind))
    assert base_rows & other_rows
    assert all((a, b) in taken for a, b, _ in base_rows ^ other_rows)


def test_changing_neighbour_degree_moves_no_friend_or_work_draw(
    built: Built, out: Path, tmp_path: Path
) -> None:
    changed = with_graph(built.city, neighbour_degree=built.city.graph.neighbour_degree * 3)
    other = table_of(written(bake(built, changed), tmp_path / "neighbours"))
    base = table_of(out)
    assert len(kind_rows(other, NEIGHBOUR)) > len(kind_rows(base, NEIGHBOUR))
    assert_draws_unmoved(base, other, FRIEND, (HOUSEHOLD, NEIGHBOUR))
    assert_draws_unmoved(base, other, WORK, (HOUSEHOLD, NEIGHBOUR, FRIEND))


def test_changing_friend_degree_moves_no_work_draw(built: Built, out: Path, tmp_path: Path) -> None:
    changed = with_graph(built.city, friend_degree=built.city.graph.friend_degree * 3)
    other = table_of(written(bake(built, changed), tmp_path / "friends"))
    assert_draws_unmoved(table_of(out), other, WORK, (HOUSEHOLD, NEIGHBOUR, FRIEND))


def test_layer_json_records_provenance_and_the_graph_config(built: Built, out: Path) -> None:
    meta = json.loads((out / "layer.json").read_text())
    assert meta["city"] == "ahmedabad"
    assert meta["seed"] == built.city.seed
    assert meta["tie_kinds"] == list(KIND_NAMES)
    assert meta["population_residents_sha256"] == table_digest(built.population / "residents")
    assert len(meta["persona_assignment_sha256"]) == 64
    assert meta["persona_assignment_sha256"] == table_digest(built.personas / "assignment")
    assert meta["buildings_stage"] == "osm"
    assert meta["buildings_sha256"] == table_digest(built.buildings / "buildings")
    assert meta["graph"]["friend_degree"] == built.city.graph.friend_degree
    assert meta["graph"]["neighbour_candidates"] == built.city.graph.neighbour_candidates


def test_report_has_the_documented_sections_and_workplaces(out: Path) -> None:
    report = json.loads((out / "report.json").read_text())
    assert {
        "degree",
        "clustering",
        "cross_river",
        "homophily",
        "distance_m",
        "components",
        "strength",
        "workplaces",
        "within_targets",
    } <= set(report)
    assert set(report["workplaces"]) >= {
        "count",
        "by_source",
        "by_bank",
        "by_district",
        "workers_by_bank",
    }
    assert set(report["workplaces"]["by_source"]) == {"osm_tagged", "untagged_footprint"}
    assert set(report["workplaces"]["by_bank"]) == {"west", "east"}
    assert set(report["degree"]["mean_by_kind"]) == set(KIND_NAMES)


def test_within_targets_names_every_target_and_all(out: Path) -> None:
    flags = json.loads((out / "report.json").read_text())["within_targets"]
    assert set(flags) == TARGET_NAMES | {"all"}
    assert all(isinstance(v, bool) for v in flags.values())
    assert flags["all"] == all(v for k, v in flags.items() if k != "all")


def test_permissive_targets_pass_and_a_stricter_config_flips_a_target(
    built: Built, tmp_path: Path
) -> None:
    easy = with_graph(built.city, targets=PERMISSIVE)
    flags = json.loads((written(bake(built, easy), tmp_path / "easy") / "report.json").read_text())[
        "within_targets"
    ]
    assert all(flags.values())

    for name, strict in {
        "mean_degree": (500.0, 600.0),
        "clustering": (0.999, 1.0),
        "cross_river_non_household": (0.999, 1.0),
    }.items():
        city = with_graph(built.city, targets=replace(PERMISSIVE, **{name: strict}))
        flags = json.loads(
            (written(bake(built, city), tmp_path / name) / "report.json").read_text()
        )["within_targets"]
        assert flags[name] is False, name
        assert flags["all"] is False, name
        assert all(v for k, v in flags.items() if k not in (name, "all")), name


def test_share_targets_compare_against_the_reported_values(built: Built, tmp_path: Path) -> None:
    city = with_graph(
        built.city, targets=replace(PERMISSIVE, largest_component_min=1.0, isolated_max=0.0)
    )
    report = json.loads(
        (written(bake(built, city), tmp_path / "shares") / "report.json").read_text()
    )
    flags = report["within_targets"]
    assert flags["isolated_max"] == (report["degree"]["isolated_share"] <= 0.0)
    assert flags["largest_component_min"] == (report["components"]["largest_share"] >= 1.0)


def make_personas_stale(built: Built) -> None:
    residents = read_table(built.population / "residents")
    write_table(
        built.population / "residents",
        {**residents, "x": residents["x"] + np.float32(1.0)},
    )


def test_stale_personas_are_refused(tmp_path: Path) -> None:
    other = build_inputs(tmp_path)
    make_personas_stale(other)
    with pytest.raises(StalePersonasError, match="citybake personas"):
        bake(other)


def test_graph_command_on_stale_personas_exits_with_the_rerun_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = build_inputs(tmp_path)
    make_personas_stale(built)
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: built.city)
    with pytest.raises(SystemExit, match="citybake personas"):
        main(["--data-dir", str(tmp_path), "graph", built.city.slug])


@pytest.mark.parametrize("missing", ["layer.json", "vocab.json"])
def test_graph_command_on_a_partial_personas_bake_exits_with_a_message(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = build_inputs(tmp_path)
    (built.personas / missing).unlink()
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: built.city)
    with pytest.raises(SystemExit, match="citybake personas"):
        main(["--data-dir", str(tmp_path), "graph", built.city.slug])


def test_graph_command_writes_the_stage_and_prints_a_summary(
    built: Built, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: built.city)
    assert main(["--data-dir", str(built.dirs.root), "graph", built.city.slug]) == 0
    stage = built.dirs.stage(built.city.slug, "graph")
    assert {"graph", "report.json", "layer.json"} <= {p.name for p in stage.iterdir()}
    table = read_table(stage / "graph")
    assert len(table["offsets"]) == built.residents + 1
    assert capsys.readouterr().out.strip()


def test_graph_command_without_personas_exits_with_a_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = build_inputs(tmp_path)
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: built.city)
    for child in sorted(built.personas.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    built.personas.rmdir()
    with pytest.raises(SystemExit, match="citybake personas"):
        main(["--data-dir", str(tmp_path), "graph", built.city.slug])
