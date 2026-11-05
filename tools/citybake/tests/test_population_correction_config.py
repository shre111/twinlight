import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from twl_citybake import config as config_module
from twl_citybake.config import (
    CheckRegion,
    PinnedDownload,
    PopulationCheck,
    PopulationCorrection,
    WardGroup,
    load_city,
)


def valid_kwargs(**over: Any) -> dict[str, Any]:
    return {
        "weight": 0.75,
        "weight_range": (0.75, 0.9),
        "weight_evidence": "chosen after seeing the ward data",
        "reference_polygons": ("relation/1", "way/2"),
        **over,
    }


def test_a_valid_correction_builds_with_the_documented_defaults() -> None:
    pc = PopulationCorrection(**valid_kwargs())
    assert tuple(pc.height_edges) == (2.0, 3.0, 5.0, 8.0)
    assert tuple(pc.footprint_edges) == (100.0, 250.0, 600.0, 1500.0)
    assert tuple(pc.coverage_edges) == (0.10, 0.25, 0.45)
    assert pc.fit_min_m2_per_person == 8.0
    assert pc.min_class_share == 0.002


@pytest.mark.parametrize(
    "missing", ["weight", "weight_range", "weight_evidence", "reference_polygons"]
)
def test_required_correction_fields_have_no_default(missing: str) -> None:
    kwargs = valid_kwargs()
    del kwargs[missing]
    with pytest.raises(TypeError):
        PopulationCorrection(**kwargs)


def test_weight_at_either_end_of_its_range_is_accepted() -> None:
    PopulationCorrection(**valid_kwargs(weight=0.75))
    PopulationCorrection(**valid_kwargs(weight=0.9))
    PopulationCorrection(**valid_kwargs(weight=1.0, weight_range=(0.0, 1.0)))


@pytest.mark.parametrize("weight", [0.7499, 0.9001])
def test_weight_outside_its_range_is_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match=r"weight(?!_)"):
        PopulationCorrection(**valid_kwargs(weight=weight))


@pytest.mark.parametrize("bounds", [(0.9, 0.75), (-0.1, 0.9), (0.75, 1.1)])
def test_weight_range_must_be_ordered_and_inside_zero_one(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="weight_range"):
        PopulationCorrection(**valid_kwargs(weight=0.8, weight_range=bounds))


@pytest.mark.parametrize("evidence", ["", "   "])
def test_weight_evidence_must_not_be_empty(evidence: str) -> None:
    with pytest.raises(ValueError, match="weight_evidence"):
        PopulationCorrection(**valid_kwargs(weight_evidence=evidence))


def test_at_least_one_reference_polygon_is_required() -> None:
    with pytest.raises(ValueError, match="reference_polygons"):
        PopulationCorrection(**valid_kwargs(reference_polygons=()))


@pytest.mark.parametrize(
    "polygon", ["", "42", "node/42", "relation/", "relation/x", "Way/1", "way/1/2"]
)
def test_reference_polygons_must_be_relation_or_way_ids(polygon: str) -> None:
    with pytest.raises(ValueError, match="reference_polygons"):
        PopulationCorrection(**valid_kwargs(reference_polygons=("relation/1", polygon)))


@pytest.mark.parametrize("key", ["height_edges", "footprint_edges", "coverage_edges"])
@pytest.mark.parametrize("edges", [(3.0, 2.0, 5.0), (1.0, 1.0, 2.0), (1.0, 2.0, 2.0)])
def test_edge_lists_must_be_strictly_ascending(key: str, edges: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match=key):
        PopulationCorrection(**valid_kwargs(**{key: edges}))


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_the_cap_must_be_positive(cap: float) -> None:
    with pytest.raises(ValueError, match="fit_min_m2_per_person"):
        PopulationCorrection(**valid_kwargs(fit_min_m2_per_person=cap))


@pytest.mark.parametrize("share", [-0.001, 1.0, 1.5])
def test_min_class_share_must_be_in_zero_to_below_one(share: float) -> None:
    with pytest.raises(ValueError, match="min_class_share"):
        PopulationCorrection(**valid_kwargs(min_class_share=share))


@pytest.mark.parametrize("share", [0.0, 0.999])
def test_min_class_share_accepts_zero_and_values_just_below_one(share: float) -> None:
    assert PopulationCorrection(**valid_kwargs(min_class_share=share)).min_class_share == share


def test_the_correction_config_is_frozen() -> None:
    pc = PopulationCorrection(**valid_kwargs())
    with pytest.raises(AttributeError):
        pc.weight = 0.8  # type: ignore[misc]


def test_ward_group_and_region_validation() -> None:
    WardGroup("g", ("A", "B"), mean_ratio=(0.8, 1.2), ward_ratio=(0.5, 1.5), known_misses=("A",))
    with pytest.raises(ValueError, match="mean_ratio"):
        WardGroup("g", ("A",), mean_ratio=(1.2, 0.8))
    with pytest.raises(ValueError, match="ward_ratio"):
        WardGroup("g", ("A",), ward_ratio=(1.5, 0.5))
    with pytest.raises(ValueError):
        WardGroup("", ("A",))
    with pytest.raises(ValueError, match="C"):
        WardGroup("g", ("A", "B"), known_misses=("C",))
    CheckRegion("r", (1, 5), polygon="relation/1")
    CheckRegion("r", (1, 5), box=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        CheckRegion("r", (1, 5))
    with pytest.raises(ValueError):
        CheckRegion("r", (1, 5), polygon="relation/1", box=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        CheckRegion("r", (5, 1), polygon="relation/1")
    with pytest.raises(ValueError):
        CheckRegion("", (1, 5), polygon="relation/1")


def test_population_check_is_a_pinned_download() -> None:
    assert issubclass(PopulationCheck, PinnedDownload)
    names = {f.name for f in fields(PopulationCheck)}
    assert {
        "url",
        "md5",
        "source",
        "name_property",
        "ward_people_2011",
        "total_people_2011",
        "total_growth",
        "ward_groups",
        "regions",
    } <= names


CITY_HEAD = """
slug = "tinyville"
name = "Tinyville"
bbox = [72.0, 22.9, 72.1, 23.0]
[osm]
url = "https://example.invalid/t.osm.pbf"
md5 = "00000000000000000000000000000000"
"""

CORRECTION_TOML = """
[population_correction]
weight = 0.8
weight_range = [0.75, 0.9]
weight_evidence = "chosen after seeing the ward data"
reference_polygons = ["relation/1", "way/2"]
"""

CHECK_TOML = """
[population_check]
url = "https://example.invalid/Wards.geojson"
md5 = "11111111111111111111111111111111"
source = "Ward polygons, test"
name_property = "Name"
ward_people_2011 = 100
total_people_2011 = 1000
total_growth = [1.2, 1.5]

[[population_check.ward_groups]]
name = "core"
wards = ["A", "B"]
mean_ratio = [0.75, 1.25]
ward_ratio = [0.5, 1.5]
known_misses = ["A"]

[[population_check.ward_groups]]
name = "loose"
wards = ["C"]

[[population_check.regions]]
name = "by polygon"
polygon = "relation/9"
people = [10, 20]

[[population_check.regions]]
name = "by box"
box = [-10.0, 0.0, 10.0, 20.0]
people = [30, 40]
"""


def load_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Any:
    (tmp_path / "tinyville.toml").write_text(CITY_HEAD + text)
    monkeypatch.setattr(config_module, "files", lambda package: tmp_path)
    return load_city("tinyville")


def test_a_city_without_the_sections_has_both_fields_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    city = load_text(tmp_path, monkeypatch, "")
    assert city.population_correction is None
    assert city.population_check is None


def test_toml_parses_the_correction_section_with_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pc = load_text(tmp_path, monkeypatch, CORRECTION_TOML).population_correction
    assert isinstance(pc, PopulationCorrection)
    assert pc.weight == 0.8
    assert tuple(pc.weight_range) == (0.75, 0.9)
    assert pc.weight_evidence == "chosen after seeing the ward data"
    assert tuple(pc.reference_polygons) == ("relation/1", "way/2")
    assert tuple(pc.height_edges) == (2.0, 3.0, 5.0, 8.0)
    assert pc.fit_min_m2_per_person == 8.0
    assert pc.min_class_share == 0.002


def test_toml_edges_override_the_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = CORRECTION_TOML + "height_edges = [1.0, 2.0]\nmin_class_share = 0.01\n"
    pc = load_text(tmp_path, monkeypatch, text).population_correction
    assert pc is not None
    assert tuple(pc.height_edges) == (1.0, 2.0)
    assert pc.min_class_share == 0.01


@pytest.mark.parametrize("key", ["weight", "weight_range", "weight_evidence", "reference_polygons"])
def test_toml_correction_requires_its_four_keys_naming_the_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    text = "\n".join(
        line for line in CORRECTION_TOML.splitlines() if not re.match(rf"{key}\s*=", line)
    )
    with pytest.raises(ValueError, match=key):
        load_text(tmp_path, monkeypatch, text)


def test_toml_correction_refuses_unknown_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="wieght"):
        load_text(tmp_path, monkeypatch, CORRECTION_TOML + "wieght = 1\n")


def test_toml_parses_the_check_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pc = load_text(tmp_path, monkeypatch, CHECK_TOML).population_check
    assert isinstance(pc, PopulationCheck)
    assert pc.filename == "Wards.geojson"
    assert (pc.name_property, pc.ward_people_2011, pc.total_people_2011) == ("Name", 100, 1000)
    assert tuple(pc.total_growth) == (1.2, 1.5)
    core, loose = pc.ward_groups
    assert core == WardGroup(
        "core", ("A", "B"), mean_ratio=(0.75, 1.25), ward_ratio=(0.5, 1.5), known_misses=("A",)
    )
    assert loose.name == "loose" and loose.mean_ratio is None and loose.ward_ratio is None
    assert tuple(loose.known_misses) == ()
    by_polygon, by_box = pc.regions
    assert (by_polygon.polygon, tuple(by_polygon.people), by_polygon.box) == (
        "relation/9",
        (10, 20),
        None,
    )
    assert (by_box.polygon, tuple(by_box.people), tuple(by_box.box or ())) == (
        "",
        (30, 40),
        (-10.0, 0.0, 10.0, 20.0),
    )


@pytest.mark.parametrize(
    "bad",
    [
        "wrong = 1\n",
        '[[population_check.ward_groups]]\nname = "g"\nwards = ["A"]\nmystery = 1\n',
        '[[population_check.regions]]\nname = "r"\npolygon = "relation/1"\n'
        "people = [1, 2]\nmystery = 1\n",
    ],
)
def test_toml_check_refuses_unknown_keys_at_every_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    head = CHECK_TOML.split("[[population_check.ward_groups]]")[0]
    with pytest.raises(ValueError, match=r"wrong|mystery"):
        load_text(tmp_path, monkeypatch, head + bad)


def comment_text(path: Path) -> str:
    lines = [
        ln.strip().lstrip("#").strip()
        for ln in path.read_text().splitlines()
        if ln.strip().startswith("#")
    ]
    return re.sub(r"\s+", " ", " ".join(lines))


def ahmedabad_toml() -> Path:
    from importlib.resources import files

    return Path(str(files("twl_citybake.cities") / "ahmedabad.toml"))


def test_ahmedabad_correction_values_are_pinned() -> None:
    pc = load_city("ahmedabad").population_correction
    assert pc is not None
    assert pc.weight == 0.75
    assert tuple(pc.weight_range) == (0.71, 0.83)
    assert tuple(pc.reference_polygons) == (
        "relation/16736466",
        "relation/16736464",
        "relation/16736468",
        "relation/16736465",
        "relation/16736469",
    )
    assert tuple(pc.height_edges) == (2.0, 3.0, 5.0, 8.0)
    assert tuple(pc.footprint_edges) == (100.0, 250.0, 600.0, 1500.0)
    assert tuple(pc.coverage_edges) == (0.10, 0.25, 0.45)
    assert pc.fit_min_m2_per_person == 8.0
    assert pc.min_class_share == 0.002
    assert pc.weight_evidence.strip() != ""


def test_ahmedabad_check_values_are_pinned() -> None:
    check = load_city("ahmedabad").population_check
    assert check is not None
    assert check.url == (
        "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/"
        "46679b9c173e4f13745974134c3802f92d80bf8f/Ahmedabad/Wards.geojson"
    )
    assert check.md5 == "165e7dcd6710c7525b8ea169ca5cdeed"
    assert check.source == (
        "Ward polygons: DataMeet Municipal_Spatial_Data, Ahmedabad/Wards.geojson, "
        "commit 46679b9c, CC BY-SA 2.5 IN"
    )
    assert check.name_property == "Name"
    assert check.ward_people_2011 == 118001
    assert check.total_people_2011 == 5664062
    assert tuple(check.total_growth) == (1.27, 1.6)
    walled, north_west = check.ward_groups
    assert walled.name == "walled city"
    assert tuple(walled.wards) == ("DARIYAPUR", "JAMALPUR", "KHADIA", "SHAHPUR")
    assert tuple(walled.mean_ratio or ()) == (0.75, 1.25)
    assert tuple(walled.ward_ratio or ()) == (0.75, 1.25)
    assert tuple(walled.known_misses) == ("DARIYAPUR",)
    assert north_west.name == "north-west five"
    assert tuple(north_west.wards) == ("GOTA", "CHANDLODIA", "GHATLODIA", "THALTEJ", "BODAKDEV")
    assert north_west.mean_ratio is None and north_west.ward_ratio is None
    taluka, box, daskroi = check.regions
    assert (taluka.name, taluka.polygon, tuple(taluka.people)) == (
        "Ghatlodiya taluka",
        "relation/16736467",
        (405000, 830000),
    )
    assert (box.name, tuple(box.box or ()), tuple(box.people)) == (
        "north-west box",
        (-20000.0, 0.0, -3000.0, 20000.0),
        (970000, 1420000),
    )
    assert (daskroi.name, daskroi.polygon, tuple(daskroi.people)) == (
        "Daskroi part in the bbox",
        "relation/9661743",
        (0, 409000),
    )


def test_ahmedabad_toml_says_the_weight_is_in_the_lower_part_and_not_an_optimum() -> None:
    text = comment_text(ahmedabad_toml())
    assert "0.75 sits in the lower part of the range that passes the ward checks" in text
    assert "is not an optimum" in text
    assert "0.71 clears it by 391 people" in text
    assert "Minimising spread across all 48 wards assumes equal growth" in text
    assert "likely understates the north-west" in text
    assert "Revisit when AMC's ward-wise 2011 file or Census 2027 ward data arrives" in text


def test_a_ward_group_refuses_a_single_string_for_its_wards() -> None:
    with pytest.raises(ValueError, match="wards"):
        WardGroup("g", "GOTA")  # type: ignore[arg-type]


def test_a_ward_group_refuses_a_repeated_ward() -> None:
    with pytest.raises(ValueError, match="repeated"):
        WardGroup("g", ("A", "A"))


@pytest.mark.parametrize("key", ["height_edges", "footprint_edges", "coverage_edges"])
def test_a_scalar_edge_list_raises_a_value_error_naming_the_key(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        PopulationCorrection(**valid_kwargs(**{key: 2.0}))
