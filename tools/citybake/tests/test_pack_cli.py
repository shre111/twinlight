import json
from pathlib import Path

import pytest
from pack_fixture import (
    SLUG,
    PackInputs,
    build_pack_inputs,
    delete,
    patch_budget,
    rewrite_meta,
    snapshot,
)

from twl_citybake.cli import main
from twl_citybake.pack.stage import run_pack, write_pack_stage
from twl_citybake.store import write_json
from twl_pack import PackReader


@pytest.fixture
def cli_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PackInputs:
    inputs = build_pack_inputs(tmp_path / "data")
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: inputs.city)
    return inputs


def run_cli(inputs: PackInputs) -> int:
    return main(["--data-dir", str(inputs.dirs.root), "pack", SLUG])


def test_pack_command_writes_the_pack_and_the_stage_folders(cli_inputs: PackInputs) -> None:
    assert run_cli(cli_inputs) == 0
    pack_dir = cli_inputs.dirs.pack(SLUG)
    stage = cli_inputs.dirs.stage(SLUG, "pack")
    assert pack_dir == cli_inputs.dirs.root / "cache" / "packs" / SLUG
    assert (pack_dir / "manifest.json").exists()
    PackReader(pack_dir).verify()
    assert {p.name for p in stage.iterdir()} == {"report.json", "layer.json"}


def test_pack_command_output_equals_the_library_output(
    cli_inputs: PackInputs, tmp_path: Path
) -> None:
    run_cli(cli_inputs)
    write_pack_stage(run_pack(*cli_inputs.args(wards=True)), tmp_path / "pack", tmp_path / "stage")
    assert snapshot(cli_inputs.dirs.pack(SLUG)) == snapshot(tmp_path / "pack")
    assert snapshot(cli_inputs.dirs.stage(SLUG, "pack")) == snapshot(tmp_path / "stage")


def test_pack_command_summary_names_first_load_bytes_hash_and_pack_path(
    cli_inputs: PackInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(cli_inputs)
    out = capsys.readouterr().out
    report = json.loads(
        (cli_inputs.dirs.stage(SLUG, "pack") / "report.json").read_text(encoding="utf-8")
    )
    used = report["sizes"]["first_load"]["bytes"]
    assert any(text in out for text in (str(used), f"{used:,}", f"{used / 1e6:.2f}"))
    assert report["pack_hash"] in out
    assert str(cli_inputs.dirs.pack(SLUG)) in out


def test_pack_command_summary_reports_moved_repointed_and_walled_city_floor(
    cli_inputs: PackInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(cli_inputs)
    out = capsys.readouterr().out
    assert "moved" in out and "repointed" in out
    assert "70.5" in out or "0.705" in out


@pytest.mark.parametrize(
    ("missing", "command"),
    [
        ("population/residents", "citybake population"),
        ("osm/layers/road_geom", "citybake osm"),
        ("osm/layers/road_edges", "citybake osm"),
        ("osm/layers/water", "citybake osm"),
        ("personas/assignment", "citybake personas"),
        ("graph/graph", "citybake graph"),
        ("stores/stores", "citybake stores"),
        ("stores/travel", "citybake stores"),
        ("stores/resident_hex", "citybake stores"),
    ],
)
def test_pack_command_on_an_incomplete_stage_exits_naming_the_command(
    cli_inputs: PackInputs, missing: str, command: str
) -> None:
    delete(cli_inputs.dirs.stage(SLUG, missing.split("/")[0]).parent / missing)
    with pytest.raises(SystemExit, match=command):
        run_cli(cli_inputs)
    assert not cli_inputs.dirs.pack(SLUG).exists()


def test_pack_command_without_the_ward_file_exits_naming_fetch(cli_inputs: PackInputs) -> None:
    cli_inputs.wards.unlink()
    with pytest.raises(SystemExit, match="citybake fetch"):
        run_cli(cli_inputs)
    assert not cli_inputs.dirs.pack(SLUG).exists()


def test_pack_command_refuses_a_failed_population_check(cli_inputs: PackInputs) -> None:
    write_json(
        cli_inputs.population / "report.json",
        {"check": {"within_targets": {"all": False, "walled city": False}}},
    )
    with pytest.raises(SystemExit, match="citybake population"):
        run_cli(cli_inputs)
    assert not cli_inputs.dirs.pack(SLUG).exists()


def test_pack_command_on_a_stale_input_exits_naming_the_stage(cli_inputs: PackInputs) -> None:
    rewrite_meta(cli_inputs.personas, population_residents_sha256="0" * 64)
    with pytest.raises(SystemExit, match="personas"):
        run_cli(cli_inputs)
    assert not cli_inputs.dirs.pack(SLUG).exists()


def test_pack_command_turns_a_budget_failure_into_a_clean_exit(
    cli_inputs: PackInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_budget(monkeypatch, "FIRST_LOAD_BUDGET", 1)
    with pytest.raises(SystemExit) as caught:
        run_cli(cli_inputs)
    assert caught.value.code not in (0, None)
    assert not (cli_inputs.dirs.pack(SLUG) / "manifest.json").exists()
    report = json.loads(
        (cli_inputs.dirs.stage(SLUG, "pack") / "report.json").read_text(encoding="utf-8")
    )
    assert report["within_budget"]["first_load"] is False
