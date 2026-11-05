import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from personas_fixture import (
    DECOY,
    MARRIED,
    NONE,
    decoy_rows,
    household_adults,
    personas_config,
    realistic_rows,
    write_residents,
    write_source,
)

from twl_citybake.cli import main
from twl_citybake.config import CityConfig, DataDirs, load_city
from twl_citybake.personas.pool import fetch_pool
from twl_citybake.personas.stage import run_personas, write_persona_layers
from twl_citybake.store import read_table

SEGMENTS = [
    "senior",
    "young_single",
    "homemaker",
    "working_graduate",
    "working_other",
    "other_adult",
]


@dataclass(frozen=True)
class Built:
    city: CityConfig
    population: Path
    pool: Path
    out: Path
    residents: int


def city_config(seed: int = 20260101) -> CityConfig:
    return replace(load_city("ahmedabad"), seed=seed)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp("personas-stage")
    source = write_source(root / "source.parquet", [*realistic_rows(), *decoy_rows(400)])
    pool = fetch_pool(personas_config(), root / "pool.parquet", urls=[str(source)])
    residents = write_residents(root / "population", household_adults(38_500))
    city = city_config()
    out = root / "out"
    write_persona_layers(run_personas(root / "population", pool, city), out)
    return Built(city, root / "population", pool, out, residents)


@pytest.fixture(scope="module")
def report(built: Built) -> dict:
    return json.loads((built.out / "report.json").read_text())


def test_fixture_is_about_one_hundred_thousand_residents(built: Built) -> None:
    assert 95_000 <= built.residents <= 115_000


def test_every_segment_share_is_within_one_point_of_the_pool(built: Built, report: dict) -> None:
    pool = read_table(built.out / "pool")
    assignment = read_table(built.out / "assignment")
    n_pool = len(pool["segment"])
    assert 5500 <= n_pool <= 6500
    pool_share = np.bincount(pool["segment"], minlength=6) / n_pool
    resident_share = np.bincount(assignment["segment"], minlength=6) / len(assignment["segment"])
    assert np.abs(pool_share - resident_share).max() < 0.01
    assert all(pool_share > 0)
    assert report["within_tolerance"] is True
    assert report["max_segment_difference"] < 0.01


def test_married_share_of_residents_is_within_one_point_of_the_pool(built: Built) -> None:
    pool = read_table(built.out / "pool")
    assignment = read_table(built.out / "assignment")
    vocab = json.loads((built.out / "vocab.json").read_text())
    code = vocab["marital"].index(MARRIED)
    pool_married = float((pool["marital"] == code).mean())
    resident_married = float((pool["marital"][assignment["persona_ref"]] == code).mean())
    assert abs(resident_married - pool_married) < 0.01


def test_sex_split_of_residents_matches_the_pool(built: Built) -> None:
    pool = read_table(built.out / "pool")
    assignment = read_table(built.out / "assignment")
    assert abs(pool["sex"][assignment["persona_ref"]].mean() - pool["sex"].mean()) < 0.015


def test_report_segments_agree_with_the_written_tables(built: Built, report: dict) -> None:
    pool = read_table(built.out / "pool")
    assignment = read_table(built.out / "assignment")
    resident_share = np.bincount(assignment["segment"], minlength=6) / len(assignment["segment"])
    for i, name in enumerate(SEGMENTS):
        entry = report["segments"][name]
        assert entry["resident_share"] == pytest.approx(resident_share[i], abs=1e-4)
        assert entry["pool_share"] == pytest.approx(float((pool["segment"] == i).mean()), abs=1e-4)
        assert entry["difference"] == pytest.approx(
            entry["resident_share"] - entry["pool_share"], abs=2e-4
        )


def test_report_has_the_documented_keys(report: dict) -> None:
    assert {
        "pool",
        "segments",
        "marginals",
        "couples",
        "married",
        "max_segment_difference",
        "within_tolerance",
    } <= set(report)
    assert set(report["pool"]) >= {"rows", "state", "district"}
    assert (report["pool"]["state"], report["pool"]["district"]) == ("Gujarat", "Ahmadabad")
    assert set(report["segments"]) == set(SEGMENTS)
    for entry in report["segments"].values():
        assert set(entry) == {"pool_share", "resident_share", "difference"}
    assert set(report["marginals"]) == {"sex", "age_band", "education", "occupation_group"}
    assert set(report["couples"]) >= {
        "count",
        "residents_in_couples",
        "gap_p10",
        "gap_p50",
        "gap_p90",
        "husband_older_or_equal_share",
    }
    assert set(report["married"]) == {
        "pool_share",
        "resident_share",
        "free_draw_married_probability",
    }


def test_marginals_report_shares_and_a_small_total_variation_distance(report: dict) -> None:
    for name, entry in report["marginals"].items():
        text = json.dumps(entry)
        assert "pool" in text and "resident" in text, name
        tv = next(v for k, v in entry.items() if "variation" in k or k in ("tv", "tvd"))
        assert 0 <= tv < 0.03, name


def test_couples_in_report_cover_a_plausible_share_with_husbands_usually_older(
    built: Built, report: dict
) -> None:
    couples = report["couples"]
    assignment = read_table(built.out / "assignment")
    paired = int((assignment["spouse"] != NONE).sum())
    assert couples["residents_in_couples"] == paired
    assert couples["count"] * 2 == paired
    assert paired / built.residents > 0.25
    assert couples["husband_older_or_equal_share"] > 0.88
    assert couples["gap_p10"] <= couples["gap_p50"] <= couples["gap_p90"]
    assert 1 <= couples["gap_p50"] <= 6


def test_written_assignment_has_one_row_per_resident_with_valid_references(
    built: Built,
) -> None:
    assignment = read_table(built.out / "assignment")
    pool = read_table(built.out / "pool")
    assert len(assignment["persona_ref"]) == built.residents
    assert assignment["persona_ref"].dtype == np.uint32
    assert assignment["segment"].dtype == np.uint16
    assert assignment["spouse"].dtype == np.uint32
    assert int(assignment["persona_ref"].max()) < len(pool["uuid"])
    paired = np.flatnonzero(assignment["spouse"] != NONE)
    assert (assignment["spouse"][assignment["spouse"][paired]] == paired).all()
    assert (assignment["segment"] == pool["segment"][assignment["persona_ref"]]).all()


def test_written_pool_table_and_vocab_hold_the_documented_columns(built: Built) -> None:
    pool = read_table(built.out / "pool")
    assert set(pool) == {
        "uuid",
        "sex",
        "age",
        "marital",
        "education",
        "occupation",
        "language",
        "segment",
    }
    vocab = json.loads((built.out / "vocab.json").read_text())
    for name in ("marital", "education", "occupation", "language"):
        assert vocab[name] == sorted(vocab[name])
    assert vocab.get("segment", vocab.get("segments")) == SEGMENTS


def test_layer_json_carries_attribution_and_the_pinned_revision(built: Built) -> None:
    meta = json.loads((built.out / "layer.json").read_text())
    assert meta["city"] == "ahmedabad"
    assert meta["seed"] == 20260101
    assert meta["attribution"] == ["Nemotron-Personas-India (NVIDIA), CC BY 4.0"]
    assert len(meta["population_residents_sha256"]) == 64
    source = meta["source"]
    assert source["dataset"] == "nvidia/Nemotron-Personas-India"
    assert source["revision"] == "adefeefcc0fc3f85726d20bed0e4ed8d66372a54"
    assert (source["state"], source["district"]) == ("Gujarat", "Ahmadabad")
    assert source["parquet"] == [f"en_IN-{i:05d}-of-00011.parquet" for i in range(11)]
    assert meta["segments_version"] == 1
    assert meta["segments"] == SEGMENTS


def test_no_persona_text_reaches_any_written_file(built: Built) -> None:
    for name, data in snapshot(built.out).items():
        assert DECOY.encode() not in data, name


def test_floats_in_report_are_rounded_to_four_places(report: dict) -> None:
    def walk(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, float):
            assert round(node, 4) == node

    walk(report)


def test_rerunning_writes_byte_identical_layers(built: Built, tmp_path: Path) -> None:
    write_persona_layers(run_personas(built.population, built.pool, built.city), tmp_path / "again")
    assert snapshot(tmp_path / "again") == snapshot(built.out)


def test_a_different_seed_changes_the_assignment_but_not_the_pool_tables(
    built: Built, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    write_persona_layers(run_personas(built.population, built.pool, city_config(seed=7)), other)
    a, b = snapshot(built.out), snapshot(other)
    assert any(a[k] != b[k] for k in a if k.startswith("assignment/") and k.endswith(".npy"))
    assert all(a[k] == b[k] for k in a if k.startswith("pool/"))


def place_cli_inputs(root: Path, residents: int = 20_000) -> tuple[CityConfig, DataDirs]:
    city = city_config()
    dirs = DataDirs(root)
    source = write_source(root / "source.parquet", [*realistic_rows(2500), *decoy_rows(100)])
    fetch_pool(
        personas_config(),
        dirs.stage(city.slug, "personas") / "pool.parquet",
        urls=[str(source)],
    )
    write_residents(dirs.stage(city.slug, "population"), household_adults(int(residents / 2.6)))
    return city, dirs


def test_personas_command_writes_the_stage_from_a_cached_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    city, dirs = place_cli_inputs(tmp_path)
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    assert main(["--data-dir", str(tmp_path), "personas", city.slug]) == 0
    out = dirs.stage(city.slug, "personas")
    assert {"pool", "assignment", "vocab.json", "report.json", "layer.json"} <= {
        p.name for p in out.iterdir()
    }
    meta = json.loads((out / "layer.json").read_text())
    assert meta["attribution"] == ["Nemotron-Personas-India (NVIDIA), CC BY 4.0"]
    residents = len(read_table(dirs.stage(city.slug, "population") / "residents")["household"])
    assert len(read_table(out / "assignment")["persona_ref"]) == residents
    assert capsys.readouterr().out.strip()


def test_personas_command_without_population_exits_with_a_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    city = city_config()
    monkeypatch.setattr("twl_citybake.cli.load_city", lambda slug: city)
    source = write_source(tmp_path / "source.parquet", realistic_rows(300))
    fetch_pool(
        personas_config(),
        DataDirs(tmp_path).stage(city.slug, "personas") / "pool.parquet",
        urls=[str(source)],
    )
    with pytest.raises(SystemExit) as raised:
        main(["--data-dir", str(tmp_path), "personas", city.slug])
    assert raised.value.code not in (0, None)
