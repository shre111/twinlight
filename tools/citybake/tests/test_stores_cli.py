from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stores_fixture import Inputs, city_with, line_inputs, snapshot

from twl_citybake.cli import main
from twl_citybake.store import read_table, write_json, write_table
from twl_citybake.stores.stage import run_stores, write_store_layers
from twl_citybake.stores.travel import GraphTravelBackend


def bake(inputs: Inputs, out: Path) -> Path:
    write_store_layers(
        run_stores(inputs.population, inputs.osm_layers, inputs.buildings_layers, inputs.city), out
    )
    return out


@pytest.fixture
def cli_city(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Inputs:
    inputs = line_inputs(tmp_path, city_with(1500.0))
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: inputs.city)
    return inputs


def run_cli(inputs: Inputs) -> int:
    return main(["--data-dir", str(inputs.dirs.root), "stores", inputs.city.slug])


def test_stores_command_writes_the_stage_and_prints_a_summary(
    cli_city: Inputs, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(cli_city) == 0
    stage = cli_city.dirs.stage(cli_city.city.slug, "stores")
    assert {"stores", "hexes", "resident_hex", "travel", "report.json", "layer.json"} <= {
        p.name for p in stage.iterdir()
    }
    assert len(read_table(stage / "stores")["node"]) == 1
    assert capsys.readouterr().out.strip()


def test_stores_command_output_equals_the_library_output(cli_city: Inputs, tmp_path: Path) -> None:
    run_cli(cli_city)
    stage = cli_city.dirs.stage(cli_city.city.slug, "stores")
    assert snapshot(stage) == snapshot(bake(cli_city, tmp_path / "lib"))


def delete(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    path.rmdir()


@pytest.mark.parametrize("missing", ["road_nodes", "road_edges", "buildings"])
def test_stores_command_on_a_missing_osm_table_exits_naming_the_osm_command(
    cli_city: Inputs, missing: str
) -> None:
    delete(cli_city.osm_layers / missing)
    with pytest.raises(SystemExit, match="citybake osm"):
        run_cli(cli_city)


def test_stores_command_on_a_partly_written_table_exits_naming_the_osm_command(
    cli_city: Inputs,
) -> None:
    (cli_city.osm_layers / "road_nodes" / "x.npy").unlink()
    with pytest.raises(SystemExit, match="citybake osm"):
        run_cli(cli_city)


def test_stores_command_without_district_names_exits_naming_the_osm_command(
    cli_city: Inputs,
) -> None:
    (cli_city.osm_layers / "districts.json").unlink()
    with pytest.raises(SystemExit, match="citybake osm"):
        run_cli(cli_city)


def test_stores_command_on_combined_buildings_without_a_layer_file_names_footprints(
    cli_city: Inputs,
) -> None:
    combined = cli_city.dirs.stage(cli_city.city.slug, "buildings") / "layers"
    write_table(combined / "buildings", read_table(cli_city.osm_layers / "buildings"))
    with pytest.raises(SystemExit, match="citybake footprints"):
        run_cli(cli_city)


def test_stores_command_without_a_population_exits_naming_the_population_command(
    cli_city: Inputs,
) -> None:
    delete(cli_city.population / "residents")
    with pytest.raises(SystemExit, match="citybake population"):
        run_cli(cli_city)


def test_stores_command_refuses_combined_buildings_older_than_the_osm_layers(
    cli_city: Inputs,
) -> None:
    combined = cli_city.dirs.stage(cli_city.city.slug, "buildings") / "layers"
    table = read_table(cli_city.osm_layers / "buildings")
    write_table(combined / "buildings", table)
    write_json(combined / "layer.json", {"osm_buildings_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="citybake footprints"):
        run_cli(cli_city)


def test_stores_command_turns_an_unreachable_pair_into_a_clean_exit(
    cli_city: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    def nothing_reaches(
        self: GraphTravelBackend, sources: np.ndarray, targets: np.ndarray, band: Any
    ) -> np.ndarray:
        return np.full((len(sources), len(targets)), np.inf)

    monkeypatch.setattr(GraphTravelBackend, "minutes", nothing_reaches)
    with pytest.raises(SystemExit, match=r"(?i)reach") as caught:
        run_cli(cli_city)
    assert caught.value.code not in (0, None)
    stage = cli_city.dirs.stage(cli_city.city.slug, "stores")
    assert not (stage / "report.json").exists()
