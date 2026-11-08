import dataclasses
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from twl_pack.format import PackError
from twl_pack.reader import PackReader
from twl_pack.writer import encode_pack, write_pack

ZONES = set(files("tzdata").joinpath("zones").read_text(encoding="utf-8").split())


def with_tier(c, tier):
    engine = dataclasses.replace(c.residents_engine, tier=tier)
    return dataclasses.replace(c, residents_engine=engine)


def reopen(contents, directory: Path) -> PackReader:
    write_pack(encode_pack(contents), directory)
    return PackReader(directory)


def with_manifest(contents, directory: Path, **changes) -> PackReader:
    write_pack(encode_pack(contents), directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    for key, value in changes.items():
        if value is None:
            del manifest[key]
        else:
            manifest[key] = value
    path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return PackReader(directory)


def test_tier_timezone_and_weekend_days_round_trip(city, tmp_path):
    reader = reopen(city, tmp_path)
    tier = reader.residents_engine().tier
    assert tier.dtype == np.uint8
    assert tier.tolist() == [2, 0, 1]
    assert reader.manifest["timezone"] == "Asia/Dubai"
    assert reader.manifest["weekend_days"] == [6, 7]


def test_toy_pack_round_trips_its_calendar_and_tier(toy, tmp_path):
    reader = reopen(toy, tmp_path)
    assert reader.residents_engine().tier.tolist() == [1, 1, 1, 1]
    assert reader.manifest["timezone"] == "Asia/Kolkata"
    assert reader.manifest["weekend_days"] == [6, 7]


def test_tier_column_follows_weight_padded_to_eight_bytes(city):
    raw = encode_pack(city).raw["residents_engine.bin"]
    assert len(raw) == 16 * 3 + 8
    assert raw[48:51] == bytes([2, 0, 1])
    assert raw[51:56] == bytes(5)


def test_manifest_lists_weekend_days_as_ints_and_timezone_as_given(city):
    manifest = json.loads(encode_pack(city).manifest)
    assert manifest["weekend_days"] == [6, 7]
    assert all(type(d) is int for d in manifest["weekend_days"])
    assert manifest["timezone"] == "Asia/Dubai"


def test_numpy_integer_weekend_days_are_accepted(city):
    out = encode_pack(dataclasses.replace(city, weekend_days=[np.int64(5), np.uint8(6)]))
    assert json.loads(out.manifest)["weekend_days"] == [5, 6]


def test_empty_weekend_days_are_accepted_and_round_trip(city, tmp_path):
    assert (
        reopen(dataclasses.replace(city, weekend_days=[]), tmp_path).manifest["weekend_days"] == []
    )


@pytest.mark.parametrize("name", ["Asia/Kolkata", "UTC", "Etc/UTC"])
def test_listed_timezone_is_accepted(city, name):
    encode_pack(dataclasses.replace(city, timezone=name))


def test_bombay_alias_follows_the_tzdata_list(city):
    candidate = dataclasses.replace(city, timezone="Asia/Bombay")
    if "Asia/Bombay" in ZONES:
        encode_pack(candidate)
    else:
        with pytest.raises(PackError, match="timezone"):
            encode_pack(candidate)


BAD_TIERS = {
    "above 2": [3, 0, 1],
    "negative": [-1, 0, 1],
    "fractional float": [1.5, 0.0, 1.0],
    "too short": [1, 1],
    "too long": [1, 1, 1, 1],
}


@pytest.mark.parametrize("tier", BAD_TIERS.values(), ids=BAD_TIERS.keys())
def test_bad_tier_is_refused_naming_the_column(city, tier):
    with pytest.raises(PackError, match="tier"):
        encode_pack(with_tier(city, np.array(tier)))


def test_whole_valued_float_tier_is_not_a_crash(city):
    try:
        encode_pack(with_tier(city, np.array([2.0, 0.0, 1.0])))
    except PackError as e:
        assert "tier" in str(e)


BAD_ZONES = ["", "asia/kolkata", "Mars/Olympus", "Asia/Kolkata ", "../etc/passwd", None, 5, b"UTC"]


@pytest.mark.parametrize("name", BAD_ZONES)
def test_bad_timezone_is_refused_by_the_writer(city, name):
    with pytest.raises(PackError, match="timezone"):
        encode_pack(dataclasses.replace(city, timezone=name))


BAD_DAYS = {
    "str": "67",
    "bytes": b"\x06\x07",
    "not a sequence": 6,
    "bool": [True, 6],
    "float": [6.0, 7],
    "zero": [0, 6],
    "eight": [6, 8],
    "negative": [-1, 6],
    "descending": [7, 6],
    "duplicate": [6, 6],
}


@pytest.mark.parametrize("days", BAD_DAYS.values(), ids=BAD_DAYS.keys())
def test_bad_weekend_days_are_refused_by_the_writer(city, days):
    with pytest.raises(PackError, match="weekend_days"):
        encode_pack(dataclasses.replace(city, weekend_days=days))


def test_pack_error_is_a_value_error(city):
    with pytest.raises(ValueError, match="timezone"):
        encode_pack(dataclasses.replace(city, timezone="Mars/Olympus"))


@pytest.mark.parametrize("name", [n for n in BAD_ZONES if isinstance(n, str)])
def test_reader_refuses_a_bad_timezone(city, tmp_path, name):
    with pytest.raises(PackError, match="timezone"):
        with_manifest(city, tmp_path, timezone=name)


@pytest.mark.parametrize("name", [5, None])
def test_reader_refuses_a_non_string_timezone(city, tmp_path, name):
    with pytest.raises(PackError, match="timezone"):
        with_manifest(city, tmp_path, timezone=name)


@pytest.mark.parametrize(
    "days",
    [[7, 6], [6, 6], [0], [8], [True], [6.0], "67", 6, None],
    ids=str,
)
def test_reader_refuses_bad_weekend_days(city, tmp_path, days):
    with pytest.raises(PackError, match="weekend_days"):
        with_manifest(city, tmp_path, weekend_days=days)


@pytest.mark.parametrize("key", ["timezone", "weekend_days"])
def test_reader_refuses_a_manifest_lacking_the_key(city, tmp_path, key):
    with pytest.raises(PackError, match=key):
        with_manifest(city, tmp_path, **{key: None})


def test_reader_still_opens_an_edited_valid_manifest(city, tmp_path):
    reader = with_manifest(city, tmp_path, timezone="Asia/Kolkata", weekend_days=[1])
    assert reader.manifest["weekend_days"] == [1]


@given(days=st.sets(st.integers(1, 7)).map(sorted))
@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_any_ascending_subset_of_weekdays_round_trips(city, tmp_path_factory, days):
    reader = reopen(dataclasses.replace(city, weekend_days=days), tmp_path_factory.mktemp("d"))
    assert reader.manifest["weekend_days"] == days


@given(tier=st.lists(st.integers(0, 2), min_size=3, max_size=3))
@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_any_tier_array_in_range_round_trips(city, tmp_path_factory, tier):
    reader = reopen(with_tier(city, np.array(tier)), tmp_path_factory.mktemp("t"))
    assert reader.residents_engine().tier.tolist() == tier


def test_changing_only_timezone_changes_the_hash(city):
    other = dataclasses.replace(city, timezone="Asia/Kolkata")
    assert encode_pack(other).pack_hash != encode_pack(city).pack_hash


def test_changing_only_weekend_days_changes_the_hash(city):
    other = dataclasses.replace(city, weekend_days=[5, 6])
    assert encode_pack(other).pack_hash != encode_pack(city).pack_hash


def test_changing_one_tier_value_changes_the_hash(city):
    other = with_tier(city, np.array([2, 0, 0]))
    assert encode_pack(other).pack_hash != encode_pack(city).pack_hash
