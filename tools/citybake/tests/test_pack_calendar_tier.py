from dataclasses import replace
from pathlib import Path

import pytest
from pack_fixture import build_pack_inputs, fixture_city, residents_table, rewrite_meta
from test_pack_stage import write_stage

from twl_citybake.pack.rules import PackStageError
from twl_citybake.pack.stage import StaleInputError, run_pack
from twl_citybake.store import read_table, table_digest, write_table
from twl_pack import PackReader


def baked_reader(tmp_path: Path, **city_changes: object) -> PackReader:
    city = replace(fixture_city(), **city_changes)
    inputs = build_pack_inputs(tmp_path / "data", city)
    pack_dir, _ = write_stage(run_pack(*inputs.args()), tmp_path)
    return PackReader(pack_dir)


def test_manifest_carries_the_city_timezone_and_weekend_days(tmp_path: Path) -> None:
    manifest = baked_reader(tmp_path).manifest
    assert manifest["timezone"] == "Asia/Kolkata"
    assert manifest["weekend_days"] == [6, 7]


def test_manifest_follows_other_calendar_values_in_the_city_config(tmp_path: Path) -> None:
    manifest = baked_reader(tmp_path, timezone="Asia/Dubai", weekend_days=(5, 6)).manifest
    assert manifest["timezone"] == "Asia/Dubai"
    assert manifest["weekend_days"] == [5, 6]


def test_pack_tier_equals_the_residents_table_tier_row_for_row(tmp_path: Path) -> None:
    inputs = build_pack_inputs(tmp_path / "data")
    pack_dir, _ = write_stage(run_pack(*inputs.args()), tmp_path)
    tier = PackReader(pack_dir).residents_engine().tier
    assert tier.tolist() == residents_table(inputs)["tier"].tolist()
    assert tier.tolist() == [i % 3 for i in range(len(tier))]


@pytest.mark.parametrize("missing", ["timezone", "weekend_days"])
def test_a_city_without_calendar_values_is_refused_before_reading_anything(
    tmp_path: Path, missing: str
) -> None:
    city = replace(fixture_city(), **{missing: None})
    nowhere = tmp_path / "nowhere"
    with pytest.raises(PackStageError, match=missing):
        run_pack(nowhere, nowhere, nowhere, nowhere, nowhere, nowhere, city, None)


def test_a_residents_table_without_a_tier_column_is_stale(tmp_path: Path) -> None:
    inputs = build_pack_inputs(tmp_path / "data")
    table = read_table(inputs.population / "residents")
    del table["tier"]
    write_table(inputs.population / "residents", table)
    sha = table_digest(inputs.population / "residents")
    for layer in (inputs.personas, inputs.graph, inputs.stores):
        rewrite_meta(layer, population_residents_sha256=sha)
    with pytest.raises(StaleInputError, match="tier"):
        run_pack(*inputs.args())
