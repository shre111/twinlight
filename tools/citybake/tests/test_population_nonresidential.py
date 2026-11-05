from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import shapely
from osm_fixture import BBOX, OsmBuilder
from population_fixture import write_world
from test_population_correction_stage import corrected_city, write_ward_file

from twl_citybake.config import CityConfig, PopulationCorrection
from twl_citybake.population.nonresidential import nonresidential_buildings
from twl_citybake.population.raster import repaired_polygon
from twl_citybake.population.stage import PopulationLayers, run_population
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table

LANDUSE = ("commercial", "retail", "industrial")
AMENITY = ("university", "college", "school", "hospital")
PROJECTION = LocalProjection.for_bbox(BBOX)
HALF_SIZE = 80.0


def area_file(path: Path, areas: dict[tuple[float, float], dict[str, str]]) -> Path:
    b = OsmBuilder(PROJECTION)
    for i, ((x, y), tags) in enumerate(areas.items()):
        b.rect(7000 + i, x, y, 2 * HALF_SIZE, 2 * HALF_SIZE, tags)
    b.write(path)
    return path


def mask_of(path: Path, points: list[tuple[float, float]]) -> list[bool]:
    lon, lat = PROJECTION.inverse(
        np.array([p[0] for p in points]), np.array([p[1] for p in points])
    )
    return nonresidential_buildings(path, lon, lat, LANDUSE, AMENITY).mask.tolist()


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"landuse": "industrial"}, True),
        ({"landuse": "commercial"}, True),
        ({"landuse": "retail"}, True),
        ({"amenity": "school"}, True),
        ({"amenity": "college"}, True),
        ({"amenity": "university"}, True),
        ({"amenity": "hospital"}, True),
        ({"landuse": "residential"}, False),
        ({"amenity": "restaurant"}, False),
        ({"building": "yes"}, False),
    ],
)
def test_only_the_listed_tag_values_exclude_a_building(
    tmp_path: Path, tags: dict[str, str], expected: bool
) -> None:
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): tags})
    assert mask_of(path, [(0.0, 0.0)]) == [expected]


def test_a_building_outside_the_area_is_not_excluded(tmp_path: Path) -> None:
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): {"landuse": "industrial"}})
    inside, outside = (HALF_SIZE - 5, 0.0), (HALF_SIZE + 5, 0.0)
    assert mask_of(path, [inside, outside]) == [True, False]


def test_a_landuse_value_listed_only_for_amenity_does_not_match(tmp_path: Path) -> None:
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): {"landuse": "school"}})
    assert mask_of(path, [(0.0, 0.0)]) == [False]


def test_an_amenity_value_listed_only_for_landuse_does_not_match(tmp_path: Path) -> None:
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): {"amenity": "industrial"}})
    assert mask_of(path, [(0.0, 0.0)]) == [False]


def test_an_area_that_cannot_be_assembled_is_counted_not_silently_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenFactory:
        def create_multipolygon(self, area: object) -> str:
            raise RuntimeError("invalid area")

    monkeypatch.setattr("twl_citybake.population.nonresidential.WKBFactory", BrokenFactory)
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): {"landuse": "industrial"}})
    lon, lat = PROJECTION.inverse(np.array([0.0]), np.array([0.0]))
    result = nonresidential_buildings(path, lon, lat, LANDUSE, AMENITY)
    assert result.skipped == 1
    assert result.areas == {}
    assert result.mask.tolist() == [False]


def test_an_invalid_area_goes_through_the_repair_before_it_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bowtie = shapely.Polygon([(72.0, 23.0), (72.2, 23.2), (72.2, 23.0), (72.0, 23.2)])
    repaired: list[bool] = []

    class BowtieFactory:
        def create_multipolygon(self, area: object) -> str:
            return shapely.to_wkb(bowtie, hex=True)

    def spy(geometry: shapely.Geometry) -> shapely.Geometry:
        repaired.append(bool(shapely.equals_exact(geometry, bowtie)))
        return repaired_polygon(geometry)

    monkeypatch.setattr("twl_citybake.population.nonresidential.WKBFactory", BowtieFactory)
    monkeypatch.setattr("twl_citybake.population.nonresidential.repaired_polygon", spy)
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): {"landuse": "industrial"}})
    result = nonresidential_buildings(path, [72.19, 72.1], [23.1, 23.15], LANDUSE, AMENITY)
    assert repaired == [True]
    assert result.mask.tolist() == [True, False]


def test_repairing_a_bowtie_gives_its_two_valid_lobes() -> None:
    bowtie = shapely.Polygon([(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)])
    assert not shapely.is_valid(bowtie)
    fixed = repaired_polygon(bowtie)
    assert shapely.is_valid(fixed)
    assert shapely.area(fixed) == pytest.approx(2.0)
    assert shapely.contains_xy(fixed, 1.9, 1.0)
    assert not shapely.contains_xy(fixed, 1.0, 1.9)


def test_a_valid_area_is_left_exactly_as_it_was() -> None:
    square = shapely.box(0.0, 0.0, 1.0, 1.0)
    assert repaired_polygon(square) is square


def test_areas_are_counted_by_tag_in_sorted_order(tmp_path: Path) -> None:
    path = area_file(
        tmp_path / "areas.osm",
        {
            (0.0, 0.0): {"landuse": "industrial"},
            (400.0, 0.0): {"amenity": "school"},
            (800.0, 0.0): {"landuse": "industrial"},
            (1200.0, 0.0): {"landuse": "residential"},
        },
    )
    lon, lat = PROJECTION.inverse(np.array([0.0]), np.array([0.0]))
    areas = nonresidential_buildings(path, lon, lat, LANDUSE, AMENITY).areas
    assert areas == {"amenity=school": 1, "landuse=industrial": 2}
    assert list(areas) == sorted(areas)


def test_no_matching_area_excludes_nothing(tmp_path: Path) -> None:
    path = area_file(tmp_path / "areas.osm", {(0.0, 0.0): {"landuse": "residential"}})
    assert mask_of(path, [(0.0, 0.0), (500.0, 500.0)]) == [False, False]


def excluding(**over: object) -> CityConfig:
    city = corrected_city()
    assert city.population_correction is not None
    correction: PopulationCorrection = replace(
        city.population_correction,
        exclude_landuse=("industrial",),
        exclude_amenity=("school",),
        **over,
    )
    return replace(city, population_correction=correction)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("exclusion-world")
    layers, tif = root / "layers", root / "pop.tif"
    write_world(BBOX, tif, layers, block=18)
    b = OsmBuilder(PROJECTION)
    b.rect(5000, -300, 0, 500, 500, {"landuse": "residential"})
    b.rect(5001, 0, 400, 600, 300, {"landuse": "residential"})
    b.rect(5002, 300, 0, 500, 500, {"landuse": "residential"})
    x, y = read_table(layers / "buildings")["x"], read_table(layers / "buildings")["y"]
    centre = (float(np.median(x)), float(np.median(y)))
    b.rect(7000, centre[0], centre[1], 400, 400, {"landuse": "industrial"})
    pbf = root / "boundaries.osm"
    b.write(pbf)
    (root / "centre.txt").write_text(f"{centre[0]} {centre[1]}")
    return {
        "layers": layers,
        "tif": tif,
        "pbf": pbf,
        "wards": write_ward_file(root / "Wards.geojson"),
        "centre": root / "centre.txt",
    }


def bake(world: dict[str, Path], city: CityConfig) -> PopulationLayers:
    return run_population(
        world["layers"], world["tif"], city, osm_pbf=world["pbf"], wards=world["wards"]
    )


@pytest.fixture(scope="module")
def plain(world: dict[str, Path]) -> PopulationLayers:
    return bake(world, corrected_city())


@pytest.fixture(scope="module")
def excluded(world: dict[str, Path]) -> PopulationLayers:
    return bake(world, excluding())


def inside_mask(world: dict[str, Path]) -> np.ndarray:
    cx, cy = (float(v) for v in world["centre"].read_text().split())
    table = read_table(world["layers"] / "buildings")
    return (np.abs(table["x"] - cx) < 200) & (np.abs(table["y"] - cy) < 200)


def test_buildings_in_an_excluded_area_house_nobody_and_others_keep_their_capacity(
    world: dict[str, Path], plain: PopulationLayers, excluded: PopulationLayers
) -> None:
    inside = inside_mask(world)
    assert inside.any() and (~inside).any()
    full = plain.building_population["capacity"]
    assert (full[inside] > 0).any()
    capacity = excluded.building_population["capacity"]
    assert (capacity[inside] == 0).all()
    assert (excluded.building_population["persons"][inside] == 0).all()
    np.testing.assert_array_equal(capacity[~inside], full[~inside])


def test_the_whole_total_is_still_placed_after_the_exclusion(
    plain: PopulationLayers, excluded: PopulationLayers
) -> None:
    assert (
        excluded.building_population["persons"].sum() == plain.building_population["persons"].sum()
    )


def test_before_figures_still_use_the_full_capacity(
    plain: PopulationLayers, excluded: PopulationLayers
) -> None:
    before = [ward["raster_ratio"] for ward in plain.report["check"]["wards"]]
    after = [ward["raster_ratio"] for ward in excluded.report["check"]["wards"]]
    assert after == pytest.approx(before)
    assert excluded.report["raster"] == plain.report["raster"]


def test_the_report_names_what_was_excluded(
    world: dict[str, Path], plain: PopulationLayers, excluded: PopulationLayers
) -> None:
    inside = inside_mask(world)
    full = plain.building_population["capacity"]
    report = excluded.report["correction"]["excluded"]
    assert report["areas"] == {"landuse=industrial": 1}
    assert report["skipped_areas"] == 0
    assert report["buildings"] == int(inside.sum())
    assert report["capacity_people"] == int(full[inside].sum())
    assert report["capacity_share"] == pytest.approx(full[inside].sum() / full.sum(), abs=1e-6)


def test_without_exclude_lists_the_extract_is_not_scanned_for_areas(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("nonresidential_buildings ran without exclude lists")

    monkeypatch.setattr("twl_citybake.population.stage.nonresidential_buildings", must_not_run)
    assert bake(world, corrected_city()).report["correction"]["excluded"] is None


def test_edge_people_are_measured_over_every_housed_building_before_the_exclusion(
    world: dict[str, Path], plain: PopulationLayers, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    def spy(grid: object, bbox: object, lon: np.ndarray, lat: np.ndarray) -> float:
        seen.append(len(lon))
        return 0.0

    monkeypatch.setattr("twl_citybake.population.stage.edge_people", spy)
    bake(world, excluding())
    full = plain.building_population["capacity"]
    assert seen == [int((full > 0).sum())]
    assert (full[inside_mask(world)] > 0).any()


def test_exclude_lists_without_the_extract_are_refused(world: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="OSM extract"):
        run_population(world["layers"], world["tif"], excluding(), wards=world["wards"])


def test_a_city_without_a_correction_excludes_nothing_even_with_landuse_in_its_extract(
    world: dict[str, Path], plain: PopulationLayers
) -> None:
    city = replace(corrected_city(), population_correction=None)
    uncorrected = bake(world, city)
    assert (
        uncorrected.building_population["capacity"] == plain.building_population["capacity"]
    ).all()


def test_an_amenity_list_alone_excludes_buildings(world: dict[str, Path], tmp_path: Path) -> None:
    cx, cy = (float(v) for v in world["centre"].read_text().split())
    b = OsmBuilder(PROJECTION)
    b.rect(5000, -300, 0, 500, 500, {"landuse": "residential"})
    b.rect(5001, 0, 400, 600, 300, {"landuse": "residential"})
    b.rect(5002, 300, 0, 500, 500, {"landuse": "residential"})
    b.rect(7000, cx, cy, 400, 400, {"amenity": "school"})
    pbf = tmp_path / "amenity.osm"
    b.write(pbf)
    city = corrected_city()
    assert city.population_correction is not None
    only_amenity = replace(
        city,
        population_correction=replace(city.population_correction, exclude_amenity=("school",)),
    )
    layers = run_population(
        world["layers"], world["tif"], only_amenity, osm_pbf=pbf, wards=world["wards"]
    )
    inside = inside_mask(world)
    assert (layers.building_population["capacity"][inside] == 0).all()
    assert layers.report["correction"]["excluded"]["areas"] == {"amenity=school": 1}
