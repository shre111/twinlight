from dataclasses import replace
from pathlib import Path

import pytest
from personas_fixture import (
    DIPLOMA,
    GRADUATE,
    MARRIED,
    NO_OCCUPATION,
    Row,
    build_pool,
    personas_config,
    realistic_rows,
    uuid_for,
)

from twl_citybake.config import SegmentRule, _segment_rule
from twl_citybake.personas.segments import check_categories, derive_segments

NAMES = ["senior", "young_single", "homemaker", "working_graduate", "working_other", "other_adult"]


def segment_of(tmp_path: Path, **fields: object) -> str:
    name = f"pool{len(list(tmp_path.glob('*.parquet')))}"
    pool = build_pool(tmp_path, [replace(Row(uuid=uuid_for(1)), **fields)], name)  # type: ignore[arg-type]
    return NAMES[int(derive_segments(pool, personas_config())[0])]


def rule(name: str, **kw: object) -> SegmentRule:
    return SegmentRule(name=name, **kw)  # type: ignore[arg-type]


def with_rules(*rules: SegmentRule) -> None:
    replace(personas_config(), segments=tuple(rules))


def test_ahmedabad_toml_round_trips_the_six_ordered_rules() -> None:
    personas = personas_config()
    assert [r.name for r in personas.segments] == NAMES
    assert personas.segments_version == 1
    assert personas.revision == "adefeefcc0fc3f85726d20bed0e4ed8d66372a54"
    assert (personas.state, personas.district) == ("Gujarat", "Ahmadabad")
    assert personas.no_occupation == NO_OCCUPATION
    assert personas.married == MARRIED


def test_ahmedabad_toml_rule_fields_match_d026() -> None:
    rules = {r.name: r for r in personas_config().segments}
    assert rules["senior"].min_age == 60
    assert rules["young_single"].max_age == 29
    assert tuple(rules["young_single"].marital) == ("Never Married",)
    assert rules["homemaker"].sex == "Female" and rules["homemaker"].occupied is False
    assert rules["working_graduate"].occupied is True
    assert set(rules["working_graduate"].education) == {GRADUATE, DIPLOMA}
    assert rules["working_other"].occupied is True and not rules["working_other"].education
    other = rules["other_adult"]
    assert other.min_age is None and other.max_age is None and other.sex is None
    assert not other.marital and not other.education and other.occupied is None


def test_age_sixty_is_senior_and_fifty_nine_is_not(tmp_path: Path) -> None:
    assert segment_of(tmp_path, age=60, occupation="Driver") == "senior"
    assert segment_of(tmp_path, age=59, occupation="Driver") == "working_other"


def test_never_married_up_to_twenty_nine_is_young_single(tmp_path: Path) -> None:
    assert segment_of(tmp_path, age=29, marital_status="Never Married") == "young_single"
    assert segment_of(tmp_path, age=30, marital_status="Never Married") == "working_other"
    assert segment_of(tmp_path, age=25, marital_status=MARRIED) == "working_other"


def test_woman_without_occupation_is_homemaker_whatever_her_marital_status(
    tmp_path: Path,
) -> None:
    for marital in (MARRIED, "Divorced", "Widowed", "Separated"):
        got = segment_of(
            tmp_path, sex="Female", age=40, marital_status=marital, occupation=NO_OCCUPATION
        )
        assert got == "homemaker"


def test_man_without_occupation_is_other_adult(tmp_path: Path) -> None:
    assert segment_of(tmp_path, sex="Male", age=40, occupation=NO_OCCUPATION) == "other_adult"


def test_occupied_graduate_or_diploma_is_working_graduate(tmp_path: Path) -> None:
    assert segment_of(tmp_path, age=40, education_level=GRADUATE) == "working_graduate"
    assert segment_of(tmp_path, age=40, education_level=DIPLOMA) == "working_graduate"
    assert segment_of(tmp_path, age=40, education_level="Secondary") == "working_other"


def test_first_match_order_decides_overlaps(tmp_path: Path) -> None:
    assert segment_of(tmp_path, age=70, occupation=NO_OCCUPATION, sex="Female") == "senior"
    young_homemaker = segment_of(
        tmp_path, age=22, occupation=NO_OCCUPATION, sex="Female", marital_status="Never Married"
    )
    assert young_homemaker == "young_single"
    young_graduate = segment_of(
        tmp_path, age=22, education_level=GRADUATE, marital_status="Never Married"
    )
    assert young_graduate == "young_single"


def test_derive_segments_labels_every_pool_row_with_a_uint16_rule_index(tmp_path: Path) -> None:
    rows = [
        Row(uuid_for(0), age=70),
        Row(uuid_for(1), age=22),
        Row(uuid_for(2), age=40, occupation=NO_OCCUPATION),
        Row(uuid_for(3), age=40, education_level=GRADUATE),
        Row(uuid_for(4), age=40),
        Row(uuid_for(5), sex="Male", age=40, occupation=NO_OCCUPATION),
    ]
    ids = derive_segments(build_pool(tmp_path, rows), personas_config())
    assert ids.dtype.name == "uint16"
    assert ids.tolist() == [0, 1, 2, 3, 4, 5]


def test_one_rule_is_rejected() -> None:
    with pytest.raises(ValueError):
        with_rules(rule("everyone"))


def test_duplicate_rule_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        with_rules(rule("a", min_age=60), rule("a", max_age=20), rule("rest"))


def test_last_rule_must_be_a_catch_all() -> None:
    with pytest.raises(ValueError):
        with_rules(rule("a", min_age=60), rule("b", max_age=20))


def test_an_earlier_catch_all_is_rejected() -> None:
    with pytest.raises(ValueError):
        with_rules(rule("a"), rule("b", min_age=60), rule("rest"))


def test_unknown_rule_keys_raise() -> None:
    with pytest.raises((TypeError, ValueError)):
        SegmentRule(name="x", mystery=1)  # type: ignore[call-arg]


def test_two_valid_rules_are_accepted() -> None:
    with_rules(rule("old", min_age=60), rule("rest"))


@pytest.mark.parametrize(
    "raw",
    [
        {"name": "x", "marital": "Never Married"},
        {"name": "x", "education": "Graduate & above"},
        {"name": "x", "occupied": "yes"},
    ],
)
def test_a_segment_rule_rejects_values_of_the_wrong_shape(raw: dict) -> None:
    with pytest.raises(ValueError, match="segment x"):
        _segment_rule(raw)


def test_a_segment_rule_rejects_an_empty_age_range() -> None:
    with pytest.raises(ValueError, match="min_age above max_age"):
        SegmentRule(name="x", min_age=40, max_age=30)


def test_check_categories_catches_misspelt_names(tmp_path: Path) -> None:
    pool = build_pool(tmp_path, realistic_rows(800))
    config = personas_config()
    check_categories(pool, config)

    with pytest.raises(ValueError, match="no occupation"):
        check_categories(pool, replace(config, no_occupation="Not Working"))

    misspelt = SegmentRule(name="young_single", max_age=29, marital=("Never Maried",))
    rules = (*config.segments[:1], misspelt, *config.segments[2:])
    with pytest.raises(ValueError, match=r"segment young_single.*Never Maried"):
        check_categories(pool, replace(config, segments=rules))
