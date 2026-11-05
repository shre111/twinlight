from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pytest
from personas_fixture import (
    DECOY,
    DISTRICT,
    NO_OCCUPATION,
    STATE,
    Row,
    build_pool,
    decoy_rows,
    personas_config,
    uuid_for,
    write_source,
)

from twl_citybake.personas.pool import fetch_pool, read_pool

KEPT = [
    Row(uuid_for(5), "Male", 41, "Currently Married", "Graduate & above", "Teacher", "Hindi"),
    Row(uuid_for(1), "Female", 29, "Never Married", "Secondary", NO_OCCUPATION, "Gujarati"),
    Row(uuid_for(3), "Female", 67, "Widowed", "Primary", NO_OCCUPATION, "Gujarati"),
    Row(uuid_for(2), "Male", 22, "Never Married", "Secondary", "Driver", "Urdu"),
]
OTHER_STATE_SAME_DISTRICT = Row(uuid_for(900), state="Maharashtra", district=DISTRICT)


def columns_of(path: Path) -> list[str]:
    with duckdb.connect() as con:
        return [
            r[0]
            for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        ]


def test_fetch_keeps_only_the_configured_state_and_district(tmp_path: Path) -> None:
    source = write_source(
        tmp_path / "src.parquet", [*KEPT, OTHER_STATE_SAME_DISTRICT, *decoy_rows()]
    )
    out = fetch_pool(personas_config(), tmp_path / "pool.parquet", urls=[str(source)])
    with duckdb.connect() as con:
        in_source = con.execute("SELECT count(*) FROM read_parquet(?)", [str(source)]).fetchone()
        kept = con.execute(
            "SELECT count(*), min(state), max(state), min(district), max(district) "
            "FROM read_parquet(?)",
            [str(out)],
        ).fetchone()
    assert in_source is not None and in_source[0] > len(KEPT) + 100
    assert kept == (len(KEPT), STATE, STATE, DISTRICT, DISTRICT)


def test_fetch_drops_persona_text_columns(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src.parquet", KEPT)
    out = fetch_pool(personas_config(), tmp_path / "pool.parquet", urls=[str(source)])
    assert set(columns_of(out)) == {
        "uuid",
        "sex",
        "age",
        "marital_status",
        "education_level",
        "occupation",
        "first_language",
        "state",
        "district",
    }
    assert DECOY.encode() not in out.read_bytes()


def test_fetch_reads_several_parquet_files_into_one_uuid_ordered_table(tmp_path: Path) -> None:
    a = write_source(tmp_path / "a.parquet", KEPT[:2])
    b = write_source(tmp_path / "b.parquet", KEPT[2:])
    out = fetch_pool(personas_config(), tmp_path / "pool.parquet", urls=[str(a), str(b)])
    with duckdb.connect() as con:
        uuids = [
            r[0] for r in con.execute("SELECT uuid FROM read_parquet(?)", [str(out)]).fetchall()
        ]
    assert uuids == sorted(r.uuid for r in KEPT)


def test_read_pool_orders_rows_by_uuid(tmp_path: Path) -> None:
    pool = build_pool(tmp_path, KEPT)
    assert pool.uuid.dtype == np.dtype("S36")
    assert [u.decode() for u in pool.uuid] == sorted(r.uuid for r in KEPT)


def test_read_pool_codes_sex_zero_female_one_male(tmp_path: Path) -> None:
    pool = build_pool(tmp_path, KEPT)
    by_uuid = {r.uuid: r for r in KEPT}
    for i, raw in enumerate(pool.uuid):
        assert pool.sex[i] == (1 if by_uuid[raw.decode()].sex == "Male" else 0)
    assert pool.sex.dtype == np.uint8 and pool.age.dtype == np.uint8


def test_read_pool_codes_decode_back_through_sorted_vocabularies(tmp_path: Path) -> None:
    pool = build_pool(tmp_path, KEPT)
    by_uuid = {r.uuid: r for r in KEPT}
    for name in ("marital", "education", "occupation", "language"):
        assert pool.vocab[name] == sorted(pool.vocab[name])
    for i, raw in enumerate(pool.uuid):
        src = by_uuid[raw.decode()]
        assert pool.age[i] == src.age
        assert pool.vocab["marital"][pool.marital[i]] == src.marital_status
        assert pool.vocab["education"][pool.education[i]] == src.education_level
        assert pool.vocab["occupation"][pool.occupation[i]] == src.occupation
        assert pool.vocab["language"][pool.language[i]] == src.first_language
    assert pool.education.dtype == np.uint8 and pool.occupation.dtype == np.uint16


def test_read_pool_is_deterministic_regardless_of_source_row_order(tmp_path: Path) -> None:
    forward = build_pool(tmp_path, KEPT, "fwd")
    backward = build_pool(tmp_path, KEPT[::-1], "bwd")
    assert forward.uuid.tolist() == backward.uuid.tolist()
    assert forward.vocab == backward.vocab
    assert np.array_equal(forward.marital, backward.marital)
    assert np.array_equal(forward.occupation, backward.occupation)


def test_fetch_raises_when_the_cached_pool_is_for_another_revision(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src.parquet", KEPT)
    dest = tmp_path / "pool.parquet"
    fetch_pool(personas_config(), dest, urls=[str(source)])
    other = replace(personas_config(), revision="0" * 40)
    with pytest.raises((ValueError, RuntimeError)):
        fetch_pool(other, dest, urls=[str(source)])


def test_fetch_raises_when_the_cached_pool_is_for_another_district(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src.parquet", KEPT)
    dest = tmp_path / "pool.parquet"
    fetch_pool(personas_config(), dest, urls=[str(source)])
    with pytest.raises((ValueError, RuntimeError)):
        fetch_pool(replace(personas_config(), district="Surat"), dest, urls=[str(source)])


def test_fetch_skips_when_the_cache_key_matches(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src.parquet", KEPT)
    dest = tmp_path / "pool.parquet"
    fetch_pool(personas_config(), dest, urls=[str(source)])
    before = dest.read_bytes()
    again = fetch_pool(personas_config(), dest, urls=[str(tmp_path / "missing.parquet")])
    assert again == dest
    assert dest.read_bytes() == before


def test_fetched_pool_records_the_revision_it_was_built_from(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src.parquet", KEPT)
    out = fetch_pool(personas_config(), tmp_path / "pool.parquet", urls=[str(source)])
    assert personas_config().revision.encode() in out.read_bytes()


def test_fetch_gives_the_same_pool_on_every_run(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src.parquet", [*KEPT, *decoy_rows()])
    a = read_pool(fetch_pool(personas_config(), tmp_path / "a.parquet", urls=[str(source)]))
    b = read_pool(fetch_pool(personas_config(), tmp_path / "b.parquet", urls=[str(source)]))
    assert a.uuid.tolist() == b.uuid.tolist()
    assert np.array_equal(a.age, b.age)
    assert a.vocab == b.vocab
