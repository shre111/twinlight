import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely
from correction_fixture import read_stage_tables, table_digest
from osm_fixture import BBOX, OsmBuilder
from population_fixture import write_world
from test_population_stage import snapshot, tiny_config

from twl_citybake import cli
from twl_citybake.config import (
    CheckRegion,
    CityConfig,
    DataDirs,
    OsmSource,
    PopulationCheck,
    PopulationCorrection,
    WardGroup,
)
from twl_citybake.osm.clip import clip_extract
from twl_citybake.population.check import load_wards
from twl_citybake.population.raster import read_boundaries, read_boundary, read_raster
from twl_citybake.population.stage import (
    PopulationLayers,
    run_population,
    write_population_layers,
)
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table, write_json

UNCORRECTED_GOLDEN = "60886a97a731a3fb154b3ef267a75a764fc74e6dc5220c4d91bf2ab0b37b5af3"
WARD_SOURCE = "Ward polygons: Test Wards, CC BY-SA 2.5 IN"
WARD_NAMES = ("ALPHAWARD", "BETAWARD")
OLD_REPORT_KEYS = {
    "raster",
    "capacity",
    "districts",
    "households",
    "ipf",
    "residents",
    "population",
    "population_total",
}
TABLES = ("building_population", "households", "residents")


def correction() -> PopulationCorrection:
    return PopulationCorrection(
        weight=0.75,
        weight_range=(0.75, 0.9),
        weight_evidence="test evidence, chosen after seeing the ward data",
        reference_polygons=("way/5000", "way/5002"),
    )


def check(**over: Any) -> PopulationCheck:
    base: dict[str, Any] = {
        "url": "https://example.invalid/Wards.geojson",
        "md5": "2" * 32,
        "source": WARD_SOURCE,
        "name_property": "Name",
        "ward_people_2011": 20000,
        "total_people_2011": 40000,
        "total_growth": (0.5, 2.0),
        "ward_groups": (WardGroup("both", WARD_NAMES, mean_ratio=(0.0, 100.0)),),
        "regions": (
            CheckRegion("by polygon", (0, 10**7), polygon="way/5001"),
            CheckRegion("by box", (0, 10**7), box=(-500.0, -500.0, 500.0, 500.0)),
        ),
    }
    return PopulationCheck(**{**base, **over})


def corrected_city(**over: Any) -> CityConfig:
    return replace(
        tiny_config(), population_correction=correction(), population_check=check(), **over
    )


def write_ward_file(path: Path) -> Path:
    cx = (BBOX.west + BBOX.east) / 2
    south, north = BBOX.south - 0.01, BBOX.north + 0.01
    west, east = BBOX.west - 0.01, BBOX.east + 0.01

    def ring(x0: float, x1: float) -> list[list[float]]:
        return [[x0, south], [x1, south], [x1, north], [x0, north], [x0, south]]

    features = [
        {
            "type": "Feature",
            "properties": {"Name": f"{i + 1} {name}"},
            "geometry": {"type": "Polygon", "coordinates": [ring(x0, x1)]},
        }
        for i, (name, x0, x1) in enumerate([(WARD_NAMES[0], west, cx), (WARD_NAMES[1], cx, east)])
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def write_boundaries(path: Path) -> Path:
    b = OsmBuilder(LocalProjection.for_bbox(BBOX))
    b.rect(5000, -300, 0, 500, 500, {"landuse": "residential"})
    b.rect(5001, 0, 400, 600, 300, {"landuse": "residential"})
    b.rect(5002, 300, 0, 500, 500, {"landuse": "residential"})
    b.write(path)
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("correction-world")
    layers, tif = root / "layers", root / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    return {
        "layers": layers,
        "tif": tif,
        "pbf": write_boundaries(root / "boundaries.osm"),
        "wards": write_ward_file(root / "Wards.geojson"),
    }


def bake(world: dict[str, Path], city: CityConfig, with_wards: bool = True) -> PopulationLayers:
    wards = world["wards"] if with_wards else None
    return run_population(world["layers"], world["tif"], city, osm_pbf=world["pbf"], wards=wards)


@pytest.fixture(scope="module")
def baseline(world: dict[str, Path]) -> PopulationLayers:
    return run_population(world["layers"], world["tif"], tiny_config())


@pytest.fixture(scope="module")
def corrected(world: dict[str, Path]) -> PopulationLayers:
    return bake(world, corrected_city())


def written(layers: PopulationLayers, out: Path) -> Path:
    write_population_layers(layers, out)
    return out


def test_a_city_with_neither_section_gives_the_tables_it_gave_before_the_correction(
    baseline: PopulationLayers, tmp_path: Path
) -> None:
    out = written(baseline, tmp_path / "out")
    assert table_digest(read_stage_tables(out)) == UNCORRECTED_GOLDEN


def test_unset_fields_give_the_same_files_whether_passed_as_none_or_left_out(
    world: dict[str, Path], baseline: PopulationLayers, tmp_path: Path
) -> None:
    explicit = replace(tiny_config(), population_correction=None, population_check=None)
    first = written(baseline, tmp_path / "a")
    second = written(run_population(world["layers"], world["tif"], explicit), tmp_path / "b")
    assert snapshot(first) == snapshot(second)


def test_an_uncorrected_report_and_layer_only_gain_null_fields(
    baseline: PopulationLayers,
) -> None:
    assert set(baseline.report) == OLD_REPORT_KEYS | {"correction", "check"}
    assert baseline.report["correction"] is None
    assert baseline.report["check"] is None
    assert baseline.meta["source"]["population_correction"] is None
    assert baseline.meta["source"]["population_check"] is None
    assert set(baseline.meta["source"]) == {
        "raster",
        "population_total",
        "households",
        "population_correction",
        "population_check",
    }


def test_the_correction_changes_the_residents_and_building_population(
    corrected: PopulationLayers, baseline: PopulationLayers
) -> None:
    assert not np.array_equal(
        corrected.building_population["persons"], baseline.building_population["persons"]
    )
    assert not np.array_equal(
        corrected.building_population["target"], baseline.building_population["target"]
    )
    assert any(
        not np.array_equal(corrected.residents[k], baseline.residents[k])
        for k in baseline.residents
    )


def test_the_correction_keeps_the_whole_bbox_total_placed(
    corrected: PopulationLayers, baseline: PopulationLayers
) -> None:
    total = baseline.report["raster"]["people_in_bbox"]
    assert corrected.report["raster"]["people_in_bbox"] == total
    assert corrected.report["raster"]["cells"] == baseline.report["raster"]["cells"]
    assert corrected.report["districts"]["marginal_people"] == pytest.approx(total, rel=1e-6)
    assert corrected.building_population["target"].sum() == pytest.approx(total, rel=1e-6)


def test_tables_after_the_correction_stay_consistent_with_each_other_and_the_report(
    corrected: PopulationLayers, tmp_path: Path
) -> None:
    out = written(corrected, tmp_path / "out")
    report = json.loads((out / "report.json").read_text())
    hh, residents = read_table(out / "households"), read_table(out / "residents")
    bp = read_table(out / "building_population")
    assert len(hh["building"]) == report["households"]["count"]
    assert int(hh["size"].sum()) == report["households"]["people"]
    assert int(bp["persons"].sum()) == report["households"]["people"]
    assert len(residents["household"]) == report["residents"]["count"]
    assert residents["household"].max() < len(hh["building"])
    assert (bp["persons"] <= bp["capacity"]).all()
    assert report["capacity"]["buildings_over_capacity"] == 0
    assert report["ipf"]["all_converged"] is True


def test_the_correction_report_names_the_stated_weight_and_the_reference(
    world: dict[str, Path], corrected: PopulationLayers
) -> None:
    section = corrected.report["correction"]
    assert section["weight"] == 0.75
    assert list(section["weight_range"]) == [0.75, 0.9]
    assert section["weight_evidence"] == correction().weight_evidence
    assert list(section["reference_polygons"]) == ["way/5000", "way/5002"]
    grid = read_raster(world["tif"], BBOX)
    lon, lat = grid.centres()
    inside = shapely.contains_xy(
        shapely.union_all(list(read_boundaries(world["pbf"], ["way/5000", "way/5002"]).values())),
        lon,
        lat,
    )
    first_only = shapely.contains_xy(read_boundary(world["pbf"], "way/5000"), lon, lat)
    assert inside.sum() > first_only.sum() > 0
    assert section["reference_cells"] == int(inside.sum())
    assert section["people_moved"] > 0


def test_ward_polygons_never_enter_a_stage_table_or_the_layer_attribution(
    corrected: PopulationLayers, baseline: PopulationLayers, tmp_path: Path
) -> None:
    out = written(corrected, tmp_path / "out")
    reference = written(baseline, tmp_path / "ref")
    for name in TABLES:
        assert {p.name for p in (out / name).iterdir()} == {
            p.name for p in (reference / name).iterdir()
        }
        got, expected = read_table(out / name), read_table(reference / name)
        assert {k: v.dtype for k, v in got.items()} == {k: v.dtype for k, v in expected.items()}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "report.json":
            data = path.read_bytes()
            for ward in WARD_NAMES:
                assert ward.encode() not in data
                assert ward.encode("utf-16-le") not in data
                assert ward.encode("utf-32-le") not in data
    meta = json.loads((out / "layer.json").read_text())
    assert meta["attribution"] == baseline.meta["attribution"]
    assert not any(WARD_SOURCE in entry for entry in meta["attribution"])
    assert WARD_SOURCE not in json.dumps(meta["attribution"])
    for ward in WARD_NAMES:
        assert ward not in (out / "layer.json").read_text()


def test_the_report_credits_the_ward_source(corrected: PopulationLayers, tmp_path: Path) -> None:
    out = written(corrected, tmp_path / "out")
    report = json.loads((out / "report.json").read_text())
    assert report["check"]["source"] == WARD_SOURCE


def test_layer_json_records_the_correction_and_the_pinned_check_download(
    corrected: PopulationLayers, tmp_path: Path
) -> None:
    out = written(corrected, tmp_path / "out")
    source = json.loads((out / "layer.json").read_text())["source"]
    assert source["population_correction"] == json.loads(json.dumps(asdict(correction())))
    assert source["population_check"] == {
        "url": "https://example.invalid/Wards.geojson",
        "md5": "2" * 32,
        "source": WARD_SOURCE,
    }


def test_the_check_ward_people_are_the_corrected_persons_summed_in_each_ward(
    world: dict[str, Path], corrected: PopulationLayers
) -> None:
    buildings = read_table(world["layers"] / "buildings")
    lon, lat = LocalProjection.for_bbox(BBOX).inverse(buildings["x"], buildings["y"])
    persons = corrected.building_population["persons"].astype(np.float64)
    expected = [
        float(persons[shapely.contains_xy(w.geometry, lon, lat)].sum())
        for w in load_wards(world["wards"], "Name")
    ]
    rows = corrected.report["check"]["wards"]
    assert [r["people"] for r in rows] == pytest.approx(expected, abs=1e-3)
    assert [r["ratio"] for r in rows] == pytest.approx([e / 20000 for e in expected], abs=1e-6)
    assert corrected.report["check"]["total"]["people"] == pytest.approx(persons.sum(), abs=1e-3)


def test_the_check_before_figures_come_from_the_uncorrected_raster_allocation(
    world: dict[str, Path], corrected: PopulationLayers, baseline: PopulationLayers
) -> None:
    buildings = read_table(world["layers"] / "buildings")
    lon, lat = LocalProjection.for_bbox(BBOX).inverse(buildings["x"], buildings["y"])
    raster_people = baseline.building_population["target"]
    expected = [
        float(raster_people[shapely.contains_xy(w.geometry, lon, lat)].sum()) / 20000
        for w in load_wards(world["wards"], "Name")
    ]
    section = corrected.report["check"]
    assert [r["raster_ratio"] for r in section["wards"]] == pytest.approx(expected, abs=1e-6)
    assert [r["raster_ratio"] for r in section["wards"]] != pytest.approx(
        [r["ratio"] for r in section["wards"]], abs=1e-6
    )
    assert section["total"]["raster_ratio_to_2011"] == pytest.approx(
        float(raster_people.sum()) / 40000, abs=1e-6
    )


def test_a_check_without_a_correction_reports_equal_before_and_after_figures(
    world: dict[str, Path],
) -> None:
    result = bake(world, replace(tiny_config(), population_check=check()))
    assert result.report["correction"] is None
    rows = result.report["check"]["wards"]
    assert rows
    assert [r["raster_ratio"] for r in rows] == pytest.approx([r["ratio"] for r in rows], rel=0.02)


def test_a_correction_without_a_check_leaves_the_check_null(world: dict[str, Path]) -> None:
    result = bake(world, replace(corrected_city(), population_check=None), with_wards=False)
    assert result.report["check"] is None
    assert result.report["correction"]["weight"] == 0.75
    assert result.meta["source"]["population_check"] is None


def test_running_twice_writes_byte_identical_files(world: dict[str, Path], tmp_path: Path) -> None:
    first = written(bake(world, corrected_city()), tmp_path / "a")
    second = written(bake(world, corrected_city()), tmp_path / "b")
    assert snapshot(first) == snapshot(second)


def test_a_correction_without_the_osm_extract_is_refused(world: dict[str, Path]) -> None:
    city = replace(corrected_city(), population_check=None)
    with pytest.raises(ValueError):
        run_population(world["layers"], world["tif"], city)


def test_a_check_without_the_ward_file_is_refused(world: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match=r"(?i)ward"):
        bake(world, corrected_city(), with_wards=False)


def test_a_reference_polygon_missing_from_the_extract_is_refused_naming_it(
    world: dict[str, Path],
) -> None:
    pc = replace(correction(), reference_polygons=("way/5000", "way/424242"))
    with pytest.raises(ValueError, match="424242"):
        bake(world, replace(corrected_city(), population_correction=pc))


def test_a_check_region_polygon_missing_from_the_extract_is_refused_naming_it(
    world: dict[str, Path],
) -> None:
    regions = (CheckRegion("gone", (0, 1), polygon="relation/777777"),)
    with pytest.raises(ValueError, match="777777"):
        bake(world, replace(corrected_city(), population_check=check(regions=regions)))


def test_read_boundaries_matches_read_boundary_for_each_polygon(world: dict[str, Path]) -> None:
    polygons = ["way/5002", "way/5000", "way/5001"]
    found = read_boundaries(world["pbf"], polygons)
    assert set(found) == set(polygons)
    for polygon in polygons:
        assert shapely.equals(found[polygon], read_boundary(world["pbf"], polygon))


def test_read_boundaries_raises_naming_every_missing_polygon_in_one_go(
    world: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="way/999999"):
        read_boundaries(world["pbf"], ["way/5000", "way/999999"])


def test_read_boundary_still_names_the_polygon_in_its_message(world: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="population_total polygon way/999999 is not in"):
        read_boundary(world["pbf"], "way/999999")


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def set_up_cli(tmp_path: Path, city: CityConfig, with_wards: bool = True) -> DataDirs:
    dirs = DataDirs(tmp_path)
    assert city.population_raster is not None
    write_world(
        city.bbox,
        dirs.raw / "population" / city.population_raster.raster_name,
        dirs.stage(city.slug, "osm") / "layers",
        block=18,
    )
    clip_extract(
        write_boundaries(tmp_path / "boundaries.osm"),
        dirs.stage(city.slug, "osm") / f"{city.slug}.osm.pbf",
        BBOX,
    )
    if with_wards:
        assert city.population_check is not None
        write_ward_file(dirs.raw / "population_check" / city.slug / city.population_check.filename)
    return dirs


def run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    city: CityConfig,
) -> tuple[Any, str, str]:
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    code: Any
    try:
        code = cli.main(["--data-dir", str(tmp_path), "population", city.slug])
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_population_command_without_the_ward_file_names_citybake_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    city = corrected_city()
    set_up_cli(tmp_path, city, with_wards=False)
    code, _, err = run_cli(tmp_path, monkeypatch, capsys, city)
    assert code not in (0, None)
    assert "citybake fetch" in str(code) + err
    assert not (DataDirs(tmp_path).stage(city.slug, "population") / "report.json").exists()


def test_a_passing_check_exits_zero_prints_the_regions_and_writes_the_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    city = corrected_city()
    dirs = set_up_cli(tmp_path, city)
    code, out, _ = run_cli(tmp_path, monkeypatch, capsys, city)
    assert code == 0
    for name in ("by polygon", "by box"):
        assert name in out
    report = json.loads((dirs.stage(city.slug, "population") / "report.json").read_text())
    assert report["check"]["within_targets"]["all"] is True


def test_a_failed_check_writes_the_layers_then_exits_non_zero_naming_each_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failing = check(
        total_growth=(5.0, 6.0),
        ward_groups=(WardGroup("both", WARD_NAMES, mean_ratio=(1000.0, 2000.0)),),
    )
    city = replace(corrected_city(), population_check=failing)
    dirs = set_up_cli(tmp_path, city)
    code, _, err = run_cli(tmp_path, monkeypatch, capsys, city)
    message = (code if isinstance(code, str) else "") + err
    assert code not in (0, None)
    assert "both" in message
    assert "48-ward total" in message
    assert "by polygon" not in message
    assert "written" in message
    out = dirs.stage(city.slug, "population")
    assert (out / "report.json").exists() and (out / "building_population").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["check"]["within_targets"]["all"] is False
    assert report["check"]["within_targets"]["both"] is False


@pytest.mark.parametrize("label", ["correction", "check"])
def test_the_population_command_passes_the_pbf_when_only_a_check_or_a_correction_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    city = replace(
        tiny_config(),
        population_correction=correction() if label == "correction" else None,
        population_check=check() if label == "check" else None,
    )
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    dirs = set_up_cli(tmp_path, replace(city, population_check=check()))
    seen: list[tuple[Path | None, Path | None]] = []

    def spy(
        layers: Path,
        raster: Path,
        config: object,
        osm_pbf: Path | None = None,
        wards: Path | None = None,
    ) -> None:
        seen.append((osm_pbf, wards))
        raise StopIteration

    monkeypatch.setattr(cli, "run_population", spy)
    with pytest.raises(StopIteration):
        cli.main(["--data-dir", str(tmp_path), "population", city.slug])
    pbf = dirs.stage(city.slug, "osm") / f"{city.slug}.osm.pbf"
    ward_file = dirs.raw / "population_check" / city.slug / "Wards.geojson"
    assert seen == [(pbf, ward_file if label == "check" else None)]


def test_citybake_fetch_downloads_the_ward_file_pinned_by_md5(
    tmp_path: Path, tiny_city: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ward_file = write_ward_file(tmp_path / "remote" / "Wards.geojson")
    city = replace(
        tiny_config(),
        osm=OsmSource(url=tiny_city.path.as_uri(), md5=md5_of(tiny_city.path)),
        population_raster=None,
        population_check=check(url=ward_file.as_uri(), md5=md5_of(ward_file)),
    )
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    data = tmp_path / "data"
    assert cli.main(["--data-dir", str(data), "fetch", city.slug]) == 0
    fetched = data / "raw" / "population_check" / city.slug / "Wards.geojson"
    assert fetched.read_bytes() == ward_file.read_bytes()


def test_citybake_fetch_refuses_a_ward_file_that_fails_its_md5(
    tmp_path: Path, tiny_city: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ward_file = write_ward_file(tmp_path / "remote" / "Wards.geojson")
    city = replace(
        tiny_config(),
        osm=OsmSource(url=tiny_city.path.as_uri(), md5=md5_of(tiny_city.path)),
        population_raster=None,
        population_check=check(url=ward_file.as_uri(), md5="0" * 32),
    )
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    with pytest.raises(ValueError, match="pinned"):
        cli.main(["--data-dir", str(tmp_path / "data"), "fetch", city.slug])
    assert not (
        tmp_path / "data" / "raw" / "population_check" / city.slug / "Wards.geojson"
    ).exists()


LATER_STAGES = ["personas", "graph", "stores"]


def write_report(tmp_path: Path, check: Any) -> None:
    write_json(
        DataDirs(tmp_path).stage("ahmedabad", "population") / "report.json", {"check": check}
    )


@pytest.mark.parametrize("command", LATER_STAGES)
def test_a_later_stage_refuses_a_population_that_failed_its_check(
    command: str, tmp_path: Path
) -> None:
    write_report(tmp_path, {"within_targets": {"walled city": False, "box": True, "all": False}})
    with pytest.raises(SystemExit, match="failed its checks: walled city;") as caught:
        cli.main(["--data-dir", str(tmp_path), command, "ahmedabad"])
    assert "box" not in str(caught.value)


@pytest.mark.parametrize("command", LATER_STAGES)
@pytest.mark.parametrize("check", [None, {"within_targets": {"walled city": True, "all": True}}])
def test_a_later_stage_goes_on_when_the_check_passed_or_was_not_set(
    command: str, check: Any, tmp_path: Path
) -> None:
    write_report(tmp_path, check)
    with pytest.raises(SystemExit, match="is incomplete; run `citybake population ahmedabad`"):
        cli.main(["--data-dir", str(tmp_path), command, "ahmedabad"])
