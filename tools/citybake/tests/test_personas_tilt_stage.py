import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from personas_fixture import (
    DIPLOMA,
    NONE,
    decoy_rows,
    household_adults,
    household_runs,
    personas_config,
    realistic_rows,
    write_source,
)

from twl_citybake.config import CityConfig, PersonasConfig, load_city
from twl_citybake.personas.assign import assign_personas
from twl_citybake.personas.pool import FEMALE, MALE, fetch_pool, read_pool
from twl_citybake.personas.segments import derive_segments
from twl_citybake.personas.stage import PersonaLayers, run_personas, write_persona_layers
from twl_citybake.rng import Stream, stream
from twl_citybake.store import write_table

SEED = 20260101
TEN_PLUS = ("Secondary", "Higher Secondary", "Graduate & above", DIPLOMA)
TIER_SHARES = (0.38, 0.42, 0.20)
MAPPED_FEMALE = [0.2433, 0.4601, 0.534]
MAPPED_MALE = [0.3353, 0.5637, 0.636]


@dataclass(frozen=True)
class Built:
    city: CityConfig
    population: Path
    pool_path: Path
    tier: NDArray[np.uint8]
    tilted: PersonaLayers
    plain: PersonaLayers


def odds(share: float) -> float:
    return share / (1 - share)


def tilted_city(seed: int = SEED) -> CityConfig:
    city = load_city("ahmedabad")
    assert city.personas is not None and city.personas.tier_tilt is not None
    tilt = replace(city.personas.tier_tilt, education_10plus=TEN_PLUS)
    return replace(city, seed=seed, personas=replace(city.personas, tier_tilt=tilt))


def plain_city(city: CityConfig) -> CityConfig:
    assert city.personas is not None
    return replace(city, personas=replace(city.personas, tier_tilt=None))


def write_population(path: Path, tier: NDArray[np.uint8] | None, adults: NDArray[np.int64]) -> None:
    household = household_runs(adults)
    columns: dict[str, NDArray] = {
        "household": household,
        "weight": np.ones(len(household), np.float32),
    }
    if tier is not None:
        columns["tier"] = tier[household]
    write_table(path / "residents", columns)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp("tilt-stage")
    source = write_source(root / "source.parquet", [*realistic_rows(), *decoy_rows(400)])
    pool_path = fetch_pool(personas_config(), root / "pool.parquet", urls=[str(source)])
    adults = household_adults(30_000)
    tier = np.random.default_rng(17).choice(3, len(adults), p=TIER_SHARES).astype(np.uint8)
    write_population(root / "population", tier, adults)
    city = tilted_city()
    return Built(
        city=city,
        population=root / "population",
        pool_path=pool_path,
        tier=tier[household_runs(adults)],
        tilted=run_personas(root / "population", pool_path, city),
        plain=run_personas(root / "population", pool_path, plain_city(city)),
    )


@pytest.fixture(scope="module")
def tilt_report(built: Built) -> dict:
    report = built.tilted.report["tier_tilt"]
    assert report is not None
    return report


def sex_of(built: Built, layers: PersonaLayers) -> NDArray[np.uint8]:
    return layers.pool["sex"][layers.assignment["persona_ref"]]


def ten_plus_of(built: Built, layers: PersonaLayers) -> NDArray[np.bool_]:
    vocab = layers.vocab["education"]
    codes = [vocab.index(name) for name in TEN_PLUS]
    return np.isin(layers.pool["education"][layers.assignment["persona_ref"]], codes)


def free_slots(layers: PersonaLayers) -> NDArray[np.bool_]:
    spouse = layers.assignment["spouse"].astype(np.int64)
    return (spouse == NONE) | (spouse > np.arange(len(spouse)))


def shares_by_tier(
    built: Built, layers: PersonaLayers, sex: int, mask: NDArray[np.bool_]
) -> list[float]:
    chosen = sex_of(built, layers) == sex
    ten = ten_plus_of(built, layers)
    out = []
    for tier in range(3):
        cell = chosen & (built.tier == tier) & mask
        out.append(float(ten[cell].mean()) if cell.any() else 0.0)
    return out


def test_without_a_tilt_the_assignment_equals_assign_personas_without_free_weights(
    built: Built,
) -> None:
    pool = read_pool(built.pool_path)
    config = built.city.personas
    assert config is not None
    segments = derive_segments(pool, config)
    expected = assign_personas(
        household_runs(household_adults(30_000)),
        pool,
        segments,
        pool.vocab["marital"].index(config.married),
        stream(SEED, Stream.PERSONAS),
    )
    np.testing.assert_array_equal(built.plain.assignment["persona_ref"], expected.persona_ref)
    np.testing.assert_array_equal(built.plain.assignment["spouse"], expected.spouse)


def test_without_a_tilt_the_report_and_meta_carry_none(built: Built) -> None:
    assert built.plain.report["tier_tilt"] is None
    assert built.plain.meta["source"]["tier_tilt"] is None


def test_without_a_tilt_residents_need_no_tier_column(built: Built, tmp_path: Path) -> None:
    write_population(tmp_path / "population", None, household_adults(2000))
    layers = run_personas(tmp_path / "population", built.pool_path, plain_city(built.city))
    assert len(layers.assignment["persona_ref"]) > 0


def test_a_tilt_changes_the_assignment(built: Built) -> None:
    assert not np.array_equal(
        built.tilted.assignment["persona_ref"], built.plain.assignment["persona_ref"]
    )


def test_a_tilted_run_still_pairs_spouses_mutually(built: Built) -> None:
    spouse = built.tilted.assignment["spouse"]
    paired = np.flatnonzero(spouse != NONE)
    assert len(paired) > 0
    assert (spouse[spouse[paired]] == paired).all()


def test_a_tilted_run_keeps_the_segment_shares_within_a_point_of_the_pool(
    built: Built,
) -> None:
    assert built.tilted.report["within_tolerance"] is True
    assert built.tilted.report["max_segment_difference"] < 0.01


def test_rerunning_a_tilted_run_gives_the_same_assignment(built: Built) -> None:
    again = run_personas(built.population, built.pool_path, built.city)
    for key in ("persona_ref", "spouse", "segment"):
        np.testing.assert_array_equal(again.assignment[key], built.tilted.assignment[key])
    assert again.report == built.tilted.report


def test_the_report_carries_the_documented_keys(tilt_report: dict) -> None:
    assert set(tilt_report) == {
        "source",
        "mapped",
        "odds_ratios",
        "pool_10plus",
        "free_draws_10plus",
        "spouses_10plus",
        "residents_10plus",
        "segments_by_tier",
    }
    for key in (
        "mapped",
        "odds_ratios",
        "free_draws_10plus",
        "spouses_10plus",
        "residents_10plus",
    ):
        assert set(tilt_report[key]) == {"female", "male"}
        assert all(len(v) == 3 for v in tilt_report[key].values())
    assert set(tilt_report["pool_10plus"]) == {"female", "male"}


def test_the_report_source_is_the_configs_source(built: Built, tilt_report: dict) -> None:
    assert built.city.personas is not None and built.city.personas.tier_tilt is not None
    assert tilt_report["source"] == built.city.personas.tier_tilt.source


def test_the_report_maps_ahmedabads_nfhs_shares_and_their_odds_ratios(tilt_report: dict) -> None:
    assert tilt_report["mapped"]["female"] == pytest.approx(MAPPED_FEMALE, abs=1e-4)
    assert tilt_report["mapped"]["male"] == pytest.approx(MAPPED_MALE, abs=1e-4)
    for sex, mapped in (("female", MAPPED_FEMALE), ("male", MAPPED_MALE)):
        expected = [odds(s) / odds(mapped[1]) for s in mapped]
        assert tilt_report["odds_ratios"][sex] == pytest.approx(expected, abs=1e-3)
        assert tilt_report["odds_ratios"][sex][1] == 1.0


def test_the_report_pool_ten_plus_share_is_the_pools_own(built: Built, tilt_report: dict) -> None:
    pool = built.tilted.pool
    codes = [built.tilted.vocab["education"].index(name) for name in TEN_PLUS]
    ten = np.isin(pool["education"], codes)
    assert tilt_report["pool_10plus"]["female"] == pytest.approx(
        ten[pool["sex"] == FEMALE].mean(), abs=1e-4
    )
    assert tilt_report["pool_10plus"]["male"] == pytest.approx(
        ten[pool["sex"] == MALE].mean(), abs=1e-4
    )


def test_the_report_residents_ten_plus_matches_the_assignment(
    built: Built, tilt_report: dict
) -> None:
    everyone = np.ones(len(built.tier), np.bool_)
    for name, sex in (("female", FEMALE), ("male", MALE)):
        expected = shares_by_tier(built, built.tilted, sex, everyone)
        assert tilt_report["residents_10plus"][name] == pytest.approx(expected, abs=1e-4)


def test_the_report_free_draws_ten_plus_counts_only_free_slots(
    built: Built, tilt_report: dict
) -> None:
    free = free_slots(built.tilted)
    assert 0.4 < free.mean() < 0.95
    for name, sex in (("female", FEMALE), ("male", MALE)):
        expected = shares_by_tier(built, built.tilted, sex, free)
        assert tilt_report["free_draws_10plus"][name] == pytest.approx(expected, abs=1e-4)


def test_the_report_segments_by_tier_are_each_tiers_resident_shares(
    built: Built, tilt_report: dict
) -> None:
    names = built.tilted.vocab["segment"]
    assert set(tilt_report["segments_by_tier"]) == set(names)
    segment = built.tilted.assignment["segment"]
    for i, name in enumerate(names):
        expected = [float((segment[built.tier == t] == i).mean()) for t in range(3)]
        assert tilt_report["segments_by_tier"][name] == pytest.approx(expected, abs=1e-4)
    for t in range(3):
        assert sum(v[t] for v in tilt_report["segments_by_tier"].values()) == pytest.approx(
            1.0, abs=1e-3
        )


def test_the_tilt_report_floats_are_rounded_to_four_places(tilt_report: dict) -> None:
    def walk(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, float):
            assert round(node, 4) == node

    walk(tilt_report)


@pytest.mark.parametrize("sex", [FEMALE, MALE], ids=["female", "male"])
def test_free_draw_ten_plus_share_rises_from_low_to_high_tier(built: Built, sex: int) -> None:
    low, middle, high = shares_by_tier(built, built.tilted, sex, free_slots(built.tilted))
    assert low < middle < high
    assert high - low > 0.1


@pytest.mark.parametrize("sex", [FEMALE, MALE], ids=["female", "male"])
def test_without_a_tilt_free_draw_ten_plus_shares_do_not_differ_by_tier(
    built: Built, sex: int
) -> None:
    low, middle, high = shares_by_tier(built, built.plain, sex, free_slots(built.plain))
    assert max(low, middle, high) - min(low, middle, high) < 0.05


def test_the_meta_source_records_the_tilt_config(built: Built) -> None:
    assert built.city.personas is not None and built.city.personas.tier_tilt is not None
    assert built.tilted.meta["source"]["tier_tilt"] == asdict(built.city.personas.tier_tilt)


def test_written_layers_keep_the_tilt_report_and_meta(built: Built, tmp_path: Path) -> None:
    write_persona_layers(built.tilted, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    meta = json.loads((tmp_path / "layer.json").read_text())
    assert report["tier_tilt"]["mapped"]["female"] == pytest.approx(MAPPED_FEMALE, abs=1e-4)
    assert meta["source"]["tier_tilt"]["quintile_population"] == [3.2, 7.2, 15.5, 28.6, 45.5]


def test_residents_without_a_tier_column_are_refused_naming_the_rerun(
    built: Built, tmp_path: Path
) -> None:
    write_population(tmp_path / "population", None, household_adults(2000))
    with pytest.raises(ValueError, match="tier") as raised:
        run_personas(tmp_path / "population", built.pool_path, built.city)
    assert "citybake population" in str(raised.value)


def test_an_education_name_missing_from_the_pool_is_refused_naming_it(built: Built) -> None:
    assert built.city.personas is not None and built.city.personas.tier_tilt is not None
    tilt = replace(
        built.city.personas.tier_tilt, education_10plus=(*TEN_PLUS, "Postdoctoral Wizardry")
    )
    config: PersonasConfig = replace(built.city.personas, tier_tilt=tilt)
    with pytest.raises(ValueError, match="Postdoctoral Wizardry"):
        run_personas(built.population, built.pool_path, replace(built.city, personas=config))
