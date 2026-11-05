import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import shapely
from osm_fixture import BBOX, OsmBuilder
from population_fixture import write_geotiff, write_world
from test_population_stage import tiny_config

from twl_citybake import cli
from twl_citybake.config import (
    DataDirs,
    OsmSource,
    PopulationRaster,
    PopulationTotal,
    load_city,
)
from twl_citybake.population.raster import (
    PopGrid,
    read_boundary,
    read_raster,
    rescale_to_total,
)
from twl_citybake.population.stage import run_population, write_population_layers
from twl_citybake.projection import LocalProjection
from twl_citybake.sources import fetch

SQUARE_M = 500.0
LAYER_FILES = {"building_population", "households", "residents", "report.json", "layer.json"}


def raster_kwargs(**over: str) -> dict[str, str]:
    return {"url": "https://x.invalid/d/pop.zip", "md5": "0" * 32, **over}


def total_kwargs(**over: object) -> dict[str, object]:
    return {
        "people": 1000,
        "source": "Census",
        "polygon": "relation/42",
        "as_of": "2027-03-01",
        **over,
    }


def test_population_raster_requires_source_and_licence() -> None:
    with pytest.raises(TypeError):
        PopulationRaster(**raster_kwargs(licence="CC BY 4.0"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PopulationRaster(**raster_kwargs(source="s"))  # type: ignore[arg-type]


def test_raster_name_is_the_member_else_the_download_filename() -> None:
    plain = PopulationRaster(**raster_kwargs(source="s", licence="l"))
    assert plain.raster_name == "pop.zip"
    zipped = PopulationRaster(**raster_kwargs(source="s", licence="l", member="in.tif"))
    assert zipped.raster_name == "in.tif"


def test_ahmedabad_population_raster_is_pinned_ghs_pop_without_a_total() -> None:
    city = load_city("ahmedabad")
    raster = city.population_raster
    assert raster is not None
    assert raster.md5 == "15cc4641a1ac01352f09a7d9cccadcfe"
    assert raster.member.endswith(".tif")
    assert raster.raster_name == raster.member
    assert raster.licence == "CC BY 4.0"
    assert city.population_total is None


def test_population_total_accepts_relation_and_way_polygons() -> None:
    assert PopulationTotal(**total_kwargs()).polygon == "relation/42"  # type: ignore[arg-type]
    assert PopulationTotal(**total_kwargs(polygon="way/7")).polygon == "way/7"  # type: ignore[arg-type]


@pytest.mark.parametrize("people", [0, -5])
def test_population_total_people_must_be_positive(people: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        PopulationTotal(**total_kwargs(people=people))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "polygon", ["", "42", "node/42", "relation/", "relation/x", "way/1/2", "Way/1", "way/1\n"]
)
def test_population_total_polygon_must_be_relation_or_way_id(polygon: str) -> None:
    with pytest.raises(ValueError, match="relation/<id> or way/<id>"):
        PopulationTotal(**total_kwargs(polygon=polygon))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["source", "as_of"])
def test_population_total_requires_source_and_as_of(field: str) -> None:
    with pytest.raises(ValueError, match="source and an as_of"):
        PopulationTotal(**total_kwargs(**{field: ""}))  # type: ignore[arg-type]
    kwargs = total_kwargs()
    del kwargs[field]
    with pytest.raises(TypeError):
        PopulationTotal(**kwargs)  # type: ignore[arg-type]


def boundary_osm(path: Path) -> Path:
    b = OsmBuilder(LocalProjection.for_bbox(BBOX))
    b.rect(77, 0, 0, 2 * SQUARE_M, 2 * SQUARE_M, {"landuse": "residential"})
    b.rect(5000, -700, -700, 200, 200, {})
    b.relation(
        77,
        [("way", 5000, "outer")],
        {"type": "multipolygon", "boundary": "administrative"},
    )
    b.write(path)
    return path


def test_read_boundary_finds_a_closed_way(tmp_path: Path) -> None:
    shape = read_boundary(boundary_osm(tmp_path / "b.osm"), "way/77")
    assert shapely.contains_xy(shape, *LocalProjection.for_bbox(BBOX).inverse(0.0, 0.0))
    assert shapely.area(shape) == pytest.approx(
        shapely.area(shapely.box(*shapely.bounds(shape))), rel=1e-3
    )


def test_read_boundary_finds_a_multipolygon_relation(tmp_path: Path) -> None:
    projection = LocalProjection.for_bbox(BBOX)
    shape = read_boundary(boundary_osm(tmp_path / "b.osm"), "relation/77")
    assert shapely.contains_xy(shape, *projection.inverse(-700.0, -700.0))
    assert not shapely.contains_xy(shape, *projection.inverse(0.0, 0.0))


def test_read_boundary_does_not_confuse_a_way_and_a_relation_with_the_same_id(
    tmp_path: Path,
) -> None:
    pbf = boundary_osm(tmp_path / "b.osm")
    way, relation = read_boundary(pbf, "way/77"), read_boundary(pbf, "relation/77")
    assert not shapely.intersects(way, relation)


@pytest.mark.parametrize("polygon", ["way/999", "relation/999", "way/5000"])
def test_read_boundary_unknown_id_raises_naming_it(tmp_path: Path, polygon: str) -> None:
    with pytest.raises(ValueError, match=polygon):
        read_boundary(boundary_osm(tmp_path / "b.osm"), polygon)


@pytest.fixture
def grid() -> PopGrid:
    values = np.arange(1, 6 * 8 + 1, dtype=np.float64).reshape(6, 8) * 1.37
    return PopGrid(values, west=72.0, north=24.0, dlon=0.25, dlat=0.25)


def inner_box(grid: PopGrid) -> shapely.Geometry:
    return shapely.box(grid.west + 0.3, grid.north - 1.4, grid.west + 1.7, grid.north - 0.3)


def test_rescale_makes_cells_with_centres_inside_sum_to_people(grid: PopGrid) -> None:
    boundary = inner_box(grid)
    scaled, _ = rescale_to_total(grid, 5000.0, boundary)
    lon, lat = grid.centres()
    inside = shapely.contains_xy(boundary, lon, lat)
    assert inside.sum() >= 2
    assert scaled.values[inside].sum() == pytest.approx(5000.0, rel=1e-12)


def test_rescale_leaves_cells_outside_bit_identical(grid: PopGrid) -> None:
    boundary = inner_box(grid)
    scaled, _ = rescale_to_total(grid, 5000.0, boundary)
    lon, lat = grid.centres()
    inside = shapely.contains_xy(boundary, lon, lat)
    assert np.array_equal(scaled.values[~inside], grid.values[~inside])
    assert (scaled.west, scaled.north, scaled.dlon, scaled.dlat) == (
        grid.west,
        grid.north,
        grid.dlon,
        grid.dlat,
    )


def test_rescale_does_not_mutate_the_input_grid(grid: PopGrid) -> None:
    before = grid.values.copy()
    rescale_to_total(grid, 5000.0, inner_box(grid))
    assert np.array_equal(grid.values, before)


def test_rescale_returns_the_rasters_own_total_inside(grid: PopGrid) -> None:
    boundary = inner_box(grid)
    lon, lat = grid.centres()
    expected = float(grid.values[shapely.contains_xy(boundary, lon, lat)].sum())
    assert rescale_to_total(grid, 5000.0, boundary)[1] == pytest.approx(expected)


def test_rescale_keeps_relative_shape_inside(grid: PopGrid) -> None:
    boundary = inner_box(grid)
    scaled, before = rescale_to_total(grid, 5000.0, boundary)
    lon, lat = grid.centres()
    inside = shapely.contains_xy(boundary, lon, lat)
    np.testing.assert_allclose(scaled.values[inside] / grid.values[inside], 5000.0 / before)


@pytest.mark.parametrize("dx,dy", [(-0.4, 0.0), (0.0, 0.4), (3.0, 0.0), (0.0, -3.0)])
def test_rescale_refuses_a_boundary_reaching_past_the_window(
    grid: PopGrid, dx: float, dy: float
) -> None:
    rows, cols = grid.values.shape
    west, north = grid.west, grid.north
    south, east = north - rows * grid.dlat, west + cols * grid.dlon
    box = shapely.box(
        west + 0.3 + min(dx, 0),
        south + 0.3 + min(dy, 0),
        east - 0.3 + max(dx, 0),
        north - 0.3 + max(dy, 0),
    )
    with pytest.raises(ValueError, match="beyond the raster window"):
        rescale_to_total(grid, 100.0, box)


def test_rescale_refuses_a_boundary_with_no_people_inside(grid: PopGrid) -> None:
    empty = PopGrid(np.zeros_like(grid.values), grid.west, grid.north, grid.dlon, grid.dlat)
    with pytest.raises(ValueError, match="no people"):
        rescale_to_total(empty, 100.0, inner_box(grid))
    tiny = shapely.box(grid.west + 0.01, grid.north - 0.1, grid.west + 0.1, grid.north - 0.02)
    with pytest.raises(ValueError, match="no people"):
        rescale_to_total(grid, 100.0, tiny)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("ghs-world")
    layers, tif = root / "layers", root / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    b = OsmBuilder(LocalProjection.for_bbox(BBOX))
    b.rect(5000, 0, 0, 2 * SQUARE_M, 2 * SQUARE_M, {"landuse": "residential"})
    b.write(root / "boundary.osm")
    return layers, tif, root / "boundary.osm"


def raster_people_in_square(tif: Path, pbf: Path) -> float:
    grid = read_raster(tif, BBOX)
    lon, lat = grid.centres()
    inside = shapely.contains_xy(read_boundary(pbf, "way/5000"), lon, lat)
    return float(grid.values[inside].sum())


def with_total(people: int) -> PopulationTotal:
    return PopulationTotal(
        people=people, source="Census 2027 provisional", polygon="way/5000", as_of="2027-03-01"
    )


@pytest.fixture(scope="module")
def baseline(world: tuple[Path, Path, Path]) -> dict:
    layers, tif, _ = world
    return run_population(layers, tif, tiny_config()).report


@pytest.fixture(scope="module")
def scaled(world: tuple[Path, Path, Path]) -> dict:
    layers, tif, pbf = world
    people = int(raster_people_in_square(tif, pbf) * 1.5)
    city = replace(tiny_config(), population_total=with_total(people))
    return run_population(layers, tif, city, pbf).report


def test_total_raises_the_synthesised_people_and_keeps_the_district_criteria(
    baseline: dict, scaled: dict
) -> None:
    assert scaled["households"]["people"] > 1.2 * baseline["households"]["people"]
    assert scaled["raster"]["people_in_bbox"] > baseline["raster"]["people_in_bbox"]
    assert scaled["districts"]["outside_1pct"] == []
    assert scaled["districts"]["max_rel_error_1000_plus"] <= 0.01
    assert scaled["capacity"]["buildings_over_capacity"] == 0


def test_report_carries_the_total_section(world: tuple[Path, Path, Path], scaled: dict) -> None:
    _, tif, pbf = world
    section = scaled["population_total"]
    raster_people = raster_people_in_square(tif, pbf)
    assert section["people"] == int(raster_people * 1.5)
    assert section["source"] == "Census 2027 provisional"
    assert section["polygon"] == "way/5000"
    assert section["as_of"] == "2027-03-01"
    assert section["raster_people_in_polygon"] == pytest.approx(raster_people, rel=1e-6)
    assert section["factor"] == pytest.approx(section["people"] / raster_people, rel=1e-6)
    assert section["factor"] == pytest.approx(1.5, rel=1e-3)


def test_report_total_is_none_when_unset(baseline: dict) -> None:
    assert "population_total" in baseline
    assert baseline["population_total"] is None


def test_layer_json_records_raster_and_total_sources_and_attribution(
    world: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    layers, tif, pbf = world
    total = with_total(int(raster_people_in_square(tif, pbf) * 1.5))
    city = replace(tiny_config(), population_total=total)
    write_population_layers(run_population(layers, tif, city, pbf), tmp_path)
    meta = json.loads((tmp_path / "layer.json").read_text())
    assert meta["source"]["raster"]["md5"] == "1" * 32
    assert meta["source"]["raster"]["licence"] == "CC BY 4.0"
    assert meta["source"]["population_total"] == {
        "people": total.people,
        "source": "Census 2027 provisional",
        "polygon": "way/5000",
        "as_of": "2027-03-01",
    }
    assert meta["attribution"] == [
        "Test raster, CC BY 4.0",
        "Census 2027 provisional",
        "Census of India 2011 (HH-01, C-14)",
    ]


def test_layer_json_total_source_is_none_when_unset(
    world: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    layers, tif, _ = world
    write_population_layers(run_population(layers, tif, tiny_config()), tmp_path)
    meta = json.loads((tmp_path / "layer.json").read_text())
    assert meta["source"]["population_total"] is None
    assert meta["attribution"] == ["Test raster, CC BY 4.0", "Census of India 2011 (HH-01, C-14)"]


def test_total_without_an_osm_extract_raises(world: tuple[Path, Path, Path]) -> None:
    layers, tif, _ = world
    city = replace(tiny_config(), population_total=with_total(100_000))
    with pytest.raises(ValueError, match="OSM extract"):
        run_population(layers, tif, city, None)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def zip_raster(tmp_path: Path, member: str) -> tuple[Path, Path]:
    tif = write_geotiff(tmp_path / "src" / "pop.tif", np.ones((2, 2)), 72.0, 24.0)
    archive = tmp_path / "remote" / "tile.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(tif, member)
    return archive, tif


def zip_source(archive: Path, member: str, md5: str | None = None) -> PopulationRaster:
    return PopulationRaster(
        url=archive.as_uri(),
        md5=md5 or md5_of(archive),
        source="Test raster",
        licence="CC BY 4.0",
        member=member,
    )


def test_fetch_population_raster_leaves_the_member_under_raw_population(tmp_path: Path) -> None:
    archive, tif = zip_raster(tmp_path, "tile.tif")
    dirs = DataDirs(tmp_path / "data")
    out = cli.fetch_population_raster(zip_source(archive, "tile.tif"), dirs)
    assert out == dirs.raw / "population" / "tile.tif"
    assert out.read_bytes() == tif.read_bytes()
    assert (dirs.raw / "population" / "tile.zip").exists()


def test_fetch_population_raster_without_a_member_returns_the_download(tmp_path: Path) -> None:
    tif = write_geotiff(tmp_path / "remote" / "pop.tif", np.ones((2, 2)), 72.0, 24.0)
    source = PopulationRaster(url=tif.as_uri(), md5=md5_of(tif), source="s", licence="l")
    out = cli.fetch_population_raster(source, DataDirs(tmp_path / "data"))
    assert out.name == "pop.tif"
    assert out.read_bytes() == tif.read_bytes()


def test_fetch_population_raster_enforces_the_md5_pin(tmp_path: Path) -> None:
    archive, _ = zip_raster(tmp_path, "tile.tif")
    dirs = DataDirs(tmp_path / "data")
    with pytest.raises(ValueError, match="pinned"):
        cli.fetch_population_raster(zip_source(archive, "tile.tif", "0" * 32), dirs)
    assert not (dirs.raw / "population" / "tile.tif").exists()


def test_citybake_fetch_downloads_the_zip_raster_and_extracts_the_member(
    tmp_path: Path, tiny_city, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, tif = zip_raster(tmp_path, "tile.tif")
    city = replace(
        tiny_config(),
        osm=OsmSource(url=tiny_city.path.as_uri(), md5=md5_of(tiny_city.path)),
        population_raster=zip_source(archive, "tile.tif"),
    )
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    data = tmp_path / "data"
    assert cli.main(["--data-dir", str(data), "fetch", city.slug]) == 0
    assert (data / "raw" / "population" / "tile.tif").read_bytes() == tif.read_bytes()
    assert fetch(archive.as_uri(), md5_of(archive), data / "raw" / "population" / "tile.zip")


@pytest.mark.parametrize("with_total_set", [True, False])
def test_citybake_population_passes_the_clipped_pbf_only_when_a_total_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_total_set: bool
) -> None:
    base = tiny_config()
    city = replace(base, population_total=with_total(1000) if with_total_set else None)
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    dirs = DataDirs(tmp_path)
    assert city.population_raster is not None
    write_world(
        city.bbox,
        dirs.raw / "population" / city.population_raster.raster_name,
        dirs.stage(city.slug, "osm") / "layers",
        block=18,
    )
    seen: list[Path | None] = []

    def spy(layers: Path, raster: Path, config: object, osm_pbf: Path | None = None) -> None:
        seen.append(osm_pbf)
        raise StopIteration

    monkeypatch.setattr(cli, "run_population", spy)
    with pytest.raises(StopIteration):
        cli.main(["--data-dir", str(tmp_path), "population", city.slug])
    expected = dirs.stage(city.slug, "osm") / f"{city.slug}.osm.pbf" if with_total_set else None
    assert seen == [expected]


@pytest.mark.parametrize("member", ["../x.tif", "a/b.tif", "a\\b.tif", "..x.tif"])
def test_a_raster_member_must_be_a_plain_file_name(member: str) -> None:
    with pytest.raises(ValueError, match="plain name"):
        PopulationRaster(
            url="https://example.invalid/pop.zip",
            md5="1" * 32,
            source="Test raster",
            licence="CC BY 4.0",
            member=member,
        )
