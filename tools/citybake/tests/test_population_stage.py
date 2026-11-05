import json
from pathlib import Path

import numpy as np
import pytest
from osm_fixture import BBOX
from population_fixture import (
    CENSUS_COUNTS,
    CENSUS_SIZES,
    RESIDENTIAL_SHARE,
    write_world,
)

from twl_citybake.cli import main
from twl_citybake.config import (
    CityConfig,
    DataDirs,
    HouseholdConfig,
    OsmSource,
    PopulationConfig,
    PopulationRaster,
)
from twl_citybake.population.stage import run_population, write_population_layers
from twl_citybake.store import read_table

RESIDENTS = 1500


def tiny_config() -> CityConfig:
    return CityConfig(
        slug="tinyville",
        name="Tinyville",
        bbox=BBOX,
        osm=OsmSource(url="https://example.invalid/tiny.osm.pbf", md5="0" * 32),
        seed=20260101,
        population_raster=PopulationRaster(
            url="https://example.invalid/pop.tif",
            md5="1" * 32,
            source="Test raster",
            licence="CC BY 4.0",
        ),
        population=PopulationConfig(
            min_floor_m2_per_person=4.0,
            adult_share=0.69,
            residents=RESIDENTS,
            shopper_share=0.27,
            residential_share=dict(RESIDENTIAL_SHARE),
        ),
        households=HouseholdConfig(source="test fixture", sizes=CENSUS_SIZES, counts=CENSUS_COUNTS),
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("world")
    layers = root / "layers"
    tif = root / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    return layers, tif


@pytest.fixture(scope="module")
def report(world: tuple[Path, Path]) -> dict:
    layers, tif = world
    return run_population(layers, tif, tiny_config()).report


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_report_meets_the_district_and_capacity_acceptance_criteria(report: dict) -> None:
    assert report["districts"]["count"] == 2
    assert report["districts"]["outside_1pct"] == []
    assert report["districts"]["max_rel_error_1000_plus"] <= 0.01
    assert report["capacity"]["buildings_over_capacity"] == 0
    assert report["capacity"]["buildings_with_capacity"] > 0


def test_ring_cells_without_buildings_are_reported_as_unplaced_people(report: dict) -> None:
    wp = report["raster"]
    assert wp["cells_without_buildings"] > 0
    assert wp["people_without_buildings"] > 0
    placed = report["districts"]["marginal_people"]
    assert placed + wp["people_without_buildings"] == pytest.approx(wp["people_in_bbox"], rel=1e-3)


def test_report_counts_at_least_the_target_residents_with_positive_weight(report: dict) -> None:
    res = report["residents"]
    assert res["count"] >= RESIDENTS
    assert res["households"] > 0
    assert res["weight"] > 0
    assert report["ipf"]["all_converged"] is True


def test_written_layers_hold_the_documented_files(world: tuple[Path, Path], tmp_path: Path) -> None:
    layers, tif = world
    out = tmp_path / "population"
    write_population_layers(run_population(layers, tif, tiny_config()), out)
    names = {p.name for p in out.iterdir()}
    assert {"building_population", "households", "residents", "report.json", "layer.json"} <= names

    meta = json.loads((out / "layer.json").read_text())
    assert meta["city"] == "tinyville"
    assert meta["seed"] == 20260101
    assert "Test raster, CC BY 4.0" in meta["attribution"]
    assert "Census of India 2011 (HH-01, C-14)" in meta["attribution"]
    assert meta["source"]["raster"] == {
        "source": "Test raster",
        "licence": "CC BY 4.0",
        "member": "",
        "url": "https://example.invalid/pop.tif",
        "md5": "1" * 32,
    }
    assert meta["source"]["households"] == "test fixture"


def test_written_building_population_never_exceeds_capacity(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    layers, tif = world
    out = tmp_path / "population"
    write_population_layers(run_population(layers, tif, tiny_config()), out)
    table = read_table(out / "building_population")
    assert table["capacity"].dtype == np.int64
    assert table["persons"].dtype == np.int64
    assert (table["persons"] <= table["capacity"]).all()
    assert (table["placed"] <= table["capacity"]).all()
    assert table["target"].dtype == np.float64


def test_written_tables_agree_with_each_other_and_the_report(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    layers, tif = world
    out = tmp_path / "population"
    write_population_layers(run_population(layers, tif, tiny_config()), out)
    report = json.loads((out / "report.json").read_text())
    hh = read_table(out / "households")
    residents = read_table(out / "residents")
    bp = read_table(out / "building_population")
    assert len(hh["building"]) == report["households"]["count"]
    assert int(hh["size"].sum()) == report["households"]["people"]
    assert int(bp["persons"].sum()) == report["households"]["people"]
    assert len(residents["household"]) == report["residents"]["count"]
    assert residents["household"].max() < len(hh["building"])


def test_report_json_has_sorted_keys(world: tuple[Path, Path], tmp_path: Path) -> None:
    layers, tif = world
    out = tmp_path / "population"
    write_population_layers(run_population(layers, tif, tiny_config()), out)
    text = (out / "report.json").read_text()
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def test_running_twice_writes_byte_identical_files(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    layers, tif = world
    first, second = tmp_path / "a", tmp_path / "b"
    write_population_layers(run_population(layers, tif, tiny_config()), first)
    write_population_layers(run_population(layers, tif, tiny_config()), second)
    assert snapshot(first) == snapshot(second)


def test_a_different_seed_changes_the_residents_but_not_the_households(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    from dataclasses import replace

    layers, tif = world
    first, second = tmp_path / "a", tmp_path / "b"
    write_population_layers(run_population(layers, tif, tiny_config()), first)
    write_population_layers(run_population(layers, tif, replace(tiny_config(), seed=7)), second)
    a, b = snapshot(first), snapshot(second)
    assert any(a[k] != b[k] for k in a if k.startswith("residents/") and k.endswith(".npy"))
    assert all(a[k] == b[k] for k in a if k.startswith("building_population/"))


def run_command_on_tiny_city(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[CityConfig, Path, str]:
    city = tiny_config()
    assert city.population_raster is not None
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    dirs = DataDirs(tmp_path)
    write_world(
        city.bbox,
        dirs.raw / "population" / city.population_raster.raster_name,
        dirs.stage(city.slug, "osm") / "layers",
        block=18,
    )
    assert main(["--data-dir", str(tmp_path), "population", city.slug]) == 0
    return city, dirs.stage(city.slug, "population"), capsys.readouterr().out


def test_population_command_writes_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    city, out, _ = run_command_on_tiny_city(tmp_path, monkeypatch, capsys)
    report = json.loads((out / "report.json").read_text())
    assert report["districts"]["outside_1pct"] == []
    assert report["capacity"]["buildings_over_capacity"] == 0
    assert report["residents"]["count"] >= city.population.residents
    assert len(read_table(out / "residents")["household"]) == report["residents"]["count"]


def test_population_command_prints_a_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, printed = run_command_on_tiny_city(tmp_path, monkeypatch, capsys)
    assert "households" in printed and "residents" in printed
