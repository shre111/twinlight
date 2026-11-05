import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely

from twl_citybake.config import CheckRegion, PopulationCheck, WardGroup
from twl_citybake.population.check import Ward, assign_wards, load_wards, population_check

LON = np.array([0.25, 0.75, 1.5, 2.5, 3.5, 10.0])
LAT = np.full(6, 0.5)
X = LON * 1000.0
Y = LAT * 1000.0
PERSONS = np.array([60.0, 40.0, 150.0, 50.0, 0.0, 25.0])
RASTER = np.array([20.0, 10.0, 100.0, 50.0, 10.0, 5.0])
NAMES = ("ALPHA", "BETA", "GAMMA", "DELTA")
RATIOS = [1.0, 1.5, 0.5, 0.0]
RASTER_RATIOS = [0.3, 1.0, 0.5, 0.1]
POLYGONS = {"relation/1": shapely.box(0, 0, 2, 1)}


def wards(names: tuple[str, ...] = NAMES) -> list[Ward]:
    return [Ward(n, shapely.box(i, 0, i + 1, 1)) for i, n in enumerate(names)]


def check_config(**over: Any) -> PopulationCheck:
    base: dict[str, Any] = {
        "url": "https://example.invalid/Wards.geojson",
        "md5": "0" * 32,
        "source": "Ward polygons, test",
        "name_property": "Name",
        "ward_people_2011": 100,
        "total_people_2011": 200,
        "total_growth": (1.5, 1.7),
        "ward_groups": (
            WardGroup(
                "core",
                ("BETA", "GAMMA", "DELTA"),
                mean_ratio=(0.5, 0.8),
                ward_ratio=(0.5, 1.2),
                known_misses=("DELTA",),
            ),
            WardGroup("loose", ("DELTA", "ALPHA")),
        ),
        "regions": (
            CheckRegion("by polygon", (250, 300), polygon="relation/1"),
            CheckRegion("by box", (150, 200), box=(1000.0, 0.0, 3000.0, 1000.0)),
        ),
    }
    return PopulationCheck(**{**base, **over})


def run(
    config: PopulationCheck | None = None,
    persons: np.ndarray = PERSONS,
    raster: np.ndarray = RASTER,
    ward_list: list[Ward] | None = None,
) -> dict[str, Any]:
    return population_check(
        config or check_config(),
        persons,
        raster,
        LON,
        LAT,
        X,
        Y,
        wards() if ward_list is None else ward_list,
        POLYGONS,
    )


def test_each_ward_reports_people_ratio_and_raster_ratio_in_file_order() -> None:
    rows = run()["wards"]
    assert [r["name"] for r in rows] == list(NAMES)
    assert [r["people"] for r in rows] == [100.0, 150.0, 50.0, 0.0]
    assert [r["ratio"] for r in rows] == RATIOS
    assert [r["raster_ratio"] for r in rows] == pytest.approx(RASTER_RATIOS, rel=1e-12)


def test_the_source_is_the_config_source_and_outside_people_are_counted_apart() -> None:
    result = run()
    assert result["source"] == "Ward polygons, test"
    assert result["outside_wards_people"] == 25.0


def test_ward_summary_covers_all_wards_and_log_std_only_those_with_people() -> None:
    summary = run()["ward_summary"]
    assert summary["count"] == 4
    assert summary["median"] == pytest.approx(0.75, rel=1e-12)
    assert summary["p10"] == pytest.approx(0.15, rel=1e-12)
    assert summary["p90"] == pytest.approx(1.35, rel=1e-12)
    assert summary["min"] == 0.0
    assert summary["max"] == 1.5
    expected = statistics.pstdev([math.log(r) for r in RATIOS if r > 0])
    assert summary["log_std"] == pytest.approx(expected, rel=1e-12)
    expected_raster = statistics.pstdev([math.log(r) for r in RASTER_RATIOS])
    assert summary["raster_log_std"] == pytest.approx(expected_raster, rel=1e-12)


def test_total_sums_the_people_in_wards_only_and_takes_the_before_figure_from_the_raster() -> None:
    total = run()["total"]
    assert total["people"] == 300.0
    assert total["ratio_to_2011"] == 1.5
    assert total["raster_ratio_to_2011"] == pytest.approx(190.0 / 200.0, rel=1e-12)
    assert list(total["band"]) == [1.5, 1.7]


def test_group_mean_is_the_mean_of_the_ward_ratios_and_not_their_median() -> None:
    group = run()["groups"]["core"]
    assert group["mean_ratio"] == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert group["mean_ratio"] != pytest.approx(0.5)
    assert group["raster_mean_ratio"] == pytest.approx((1.0 + 0.5 + 0.1) / 3.0, rel=1e-12)
    assert list(group["band"]) == [0.5, 0.8]
    assert group["wards"] == {"BETA": 1.5, "GAMMA": 0.5, "DELTA": 0.0}


def test_raster_figures_come_from_the_raster_people_not_the_corrected_people() -> None:
    swapped = run(persons=RASTER, raster=PERSONS)
    assert [r["ratio"] for r in swapped["wards"]] == pytest.approx(RASTER_RATIOS, rel=1e-12)
    assert [r["raster_ratio"] for r in swapped["wards"]] == RATIOS
    assert swapped["groups"]["core"]["raster_mean_ratio"] == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert swapped["regions"]["by polygon"]["raster_people"] == 250.0
    assert swapped["total"]["raster_ratio_to_2011"] == 1.5


def test_ward_band_misses_split_into_below_and_above_and_known_versus_unexpected() -> None:
    group = run()["groups"]["core"]
    assert group["below_ward_band"] == ["DELTA"]
    assert group["above_ward_band"] == ["BETA"]
    assert group["known_misses"] == ["DELTA"]
    assert group["unexpected_misses"] == ["BETA"]


def test_a_ward_exactly_on_the_ward_band_edge_is_not_a_miss() -> None:
    group = run()["groups"]["core"]
    assert "GAMMA" not in group["below_ward_band"]
    upper = check_config(ward_groups=(WardGroup("core", ("BETA", "GAMMA"), ward_ratio=(0.5, 1.5)),))
    edge = run(upper)["groups"]["core"]
    assert edge["below_ward_band"] == [] and edge["above_ward_band"] == []
    assert edge["unexpected_misses"] == []


def test_unexpected_misses_are_reported_without_failing_the_run() -> None:
    result = run()
    assert result["groups"]["core"]["unexpected_misses"] == ["BETA"]
    assert result["within_targets"]["core"] is True
    assert result["within_targets"]["all"] is True


def test_a_group_without_a_mean_band_is_reported_but_not_asserted() -> None:
    result = run()
    assert result["groups"]["loose"]["mean_ratio"] == pytest.approx(0.5, rel=1e-12)
    assert "loose" not in result["within_targets"]


def test_within_targets_has_the_documented_keys() -> None:
    assert set(run()["within_targets"]) == {
        "core",
        "48-ward total",
        "by polygon",
        "by box",
        "all",
    }


def test_polygon_region_sums_buildings_inside_the_lon_lat_polygon() -> None:
    region = run()["regions"]["by polygon"]
    assert region["people"] == 250.0
    assert region["raster_people"] == 130.0
    assert list(region["band"]) == [250, 300]


def test_box_region_uses_local_metres_not_lon_lat() -> None:
    region = run()["regions"]["by box"]
    assert region["people"] == 200.0
    assert region["raster_people"] == 150.0
    assert list(region["band"]) == [150, 200]


@pytest.mark.parametrize(
    ("band", "expected"),
    [((250, 300), True), ((200, 250), True), ((251, 300), False), ((0, 249), False)],
)
def test_region_bands_are_inclusive(band: tuple[int, int], expected: bool) -> None:
    config = check_config(regions=(CheckRegion("by polygon", band, polygon="relation/1"),))
    assert run(config)["within_targets"]["by polygon"] is expected


@pytest.mark.parametrize(
    ("band", "expected"),
    [((1.0, 1.1), True), ((0.9, 1.0), True), ((1.01, 1.1), False), ((0.9, 0.99), False)],
)
def test_group_mean_bands_are_inclusive(band: tuple[float, float], expected: bool) -> None:
    group = WardGroup("flat", ("ALPHA", "BETA", "GAMMA"), mean_ratio=band)
    result = run(check_config(ward_groups=(group,)))
    assert result["groups"]["flat"]["mean_ratio"] == 1.0
    assert result["within_targets"]["flat"] is expected


@pytest.mark.parametrize(
    ("band", "expected"),
    [((1.5, 2.0), True), ((1.0, 1.5), True), ((1.51, 2.0), False), ((1.0, 1.49), False)],
)
def test_total_band_is_inclusive_and_uses_unrounded_values(
    band: tuple[float, float], expected: bool
) -> None:
    result = run(check_config(total_growth=band))
    assert result["within_targets"]["48-ward total"] is expected


def test_all_is_false_when_any_single_target_fails_and_the_others_stay_true() -> None:
    config = check_config(regions=(CheckRegion("by polygon", (0, 100), polygon="relation/1"),))
    targets = run(config)["within_targets"]
    assert targets["by polygon"] is False
    assert targets["core"] is True
    assert targets["48-ward total"] is True
    assert targets["all"] is False


def test_all_is_false_when_the_group_mean_misses_and_the_rest_pass() -> None:
    group = WardGroup("core", ("BETA", "GAMMA", "DELTA"), mean_ratio=(0.9, 1.1))
    targets = run(check_config(ward_groups=(group,)))["within_targets"]
    assert targets["core"] is False
    assert targets["48-ward total"] is True
    assert targets["all"] is False


def test_an_overlap_between_wards_counts_the_building_for_the_lowest_index_only() -> None:
    overlapping = [Ward("W0", shapely.box(0, 0, 2, 1)), Ward("W1", shapely.box(1, 0, 3, 1))]
    config = check_config(ward_groups=(), regions=())
    persons = np.array([0.0, 0.0, 100.0, 40.0, 0.0, 0.0])
    result = run(config, persons=persons, ward_list=overlapping)
    assert [r["people"] for r in result["wards"]] == [100.0, 40.0]
    assert result["outside_wards_people"] == 0.0


def test_ward_names_with_a_leading_number_match_the_configured_names() -> None:
    numbered = wards(("28 ALPHA", "3 BETA", "17 GAMMA", "4 DELTA"))
    result = run(ward_list=numbered)
    assert result["groups"]["core"]["wards"] == {"BETA": 1.5, "GAMMA": 0.5, "DELTA": 0.0}


def test_a_configured_ward_missing_from_the_file_raises_naming_it() -> None:
    group = WardGroup("core", ("BETA", "ZETA"), mean_ratio=(0.5, 1.5))
    with pytest.raises(ValueError, match="ZETA"):
        run(check_config(ward_groups=(group,)))


def test_assign_wards_gives_the_index_inside_and_minus_one_outside() -> None:
    result = assign_wards(wards(), LON, LAT)
    assert result.dtype == np.int32
    assert result.tolist() == [0, 0, 1, 2, 3, -1]


def test_assign_wards_sends_an_overlap_to_the_lowest_index_whichever_order_they_come_in() -> None:
    a, b = shapely.box(0, 0, 2, 1), shapely.box(1, 0, 3, 1)
    lon, lat = np.array([0.5, 1.5, 2.5, 9.0]), np.full(4, 0.5)
    assert assign_wards([Ward("a", a), Ward("b", b)], lon, lat).tolist() == [0, 0, 1, -1]
    assert assign_wards([Ward("b", b), Ward("a", a)], lon, lat).tolist() == [1, 0, 0, -1]


def test_assign_wards_takes_the_lowest_of_three_overlapping_wards() -> None:
    boxes = [shapely.box(0, 0, 3, 1), shapely.box(0, 0, 2, 1), shapely.box(0, 0, 1, 1)]
    result = assign_wards(
        [Ward(str(i), g) for i, g in enumerate(boxes)], np.array([0.5]), np.array([0.5])
    )
    assert result.tolist() == [0]


def write_geojson(path: Path, features: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def feature(name: str, ring: Sequence[Sequence[float]], **props: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"Name": name, **props},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def test_load_wards_keeps_file_order_and_the_raw_name_and_drops_z(tmp_path: Path) -> None:
    ring_z = [[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5], [0, 0, 5]]
    ring = [[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]
    path = write_geojson(
        tmp_path / "w.geojson",
        [feature("28 KHADIA", ring_z, Other="x"), feature("3 SHAHPUR", ring, Other="y")],
    )
    loaded = load_wards(path, "Name")
    assert [w.name for w in loaded] == ["28 KHADIA", "3 SHAHPUR"]
    assert not any(shapely.has_z(w.geometry) for w in loaded)
    assert shapely.area(loaded[0].geometry) == pytest.approx(1.0)


def test_load_wards_reads_the_name_from_the_named_property(tmp_path: Path) -> None:
    ring = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    path = write_geojson(tmp_path / "w.geojson", [feature("1 A", ring, Other="OTHERNAME")])
    assert [w.name for w in load_wards(path, "Other")] == ["OTHERNAME"]


def test_load_wards_repairs_an_invalid_geometry(tmp_path: Path) -> None:
    bowtie = [[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]]
    path = write_geojson(tmp_path / "w.geojson", [feature("1 BOW", bowtie)])
    (ward,) = load_wards(path, "Name")
    assert shapely.is_valid(ward.geometry)
    assert shapely.area(ward.geometry) == pytest.approx(2.0)
