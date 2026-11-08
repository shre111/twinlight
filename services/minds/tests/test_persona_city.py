import json
from pathlib import Path

import numpy as np
import pytest

from twl_minds.personas.city import CitySample, pack_entropy, sample_city, sample_rows
from twl_minds.personas.sampling import SampleConfig, sample_personas
from twl_pack import (
    NO_PERSONA,
    BuildingColumns,
    GraphColumns,
    PackContents,
    PackError,
    PackReader,
    ResidentColumns,
    ResidentEngineColumns,
    RiverColumns,
    RoadColumns,
    StoreColumns,
    encode_pack,
    write_pack,
)

SMALL = Path(__file__).resolve().parents[3] / "fixtures" / "pack" / "v2" / "small"
SMALL_HASH = "793993366bc30c21b2a8080a3142bca61c1fdd07cc326098de9bfa49e67526a8"
SEGMENTS = ("senior", "young_single", "working_graduate")
LOOSE = SampleConfig(size=60, min_per_segment=5, seed=3)


def toy_pack() -> PackContents:
    return PackContents(
        kind="toy",
        name="toy",
        crs="local",
        timezone="Asia/Kolkata",
        weekend_days=(6, 7),
        segments=["a0", "a1", "a2", "a3", "a4"],
        bands=[],
        attributions=[],
        districts=[
            {"id": 0, "name": "West", "x": -100.0, "y": 0.0, "east": 0.0},
            {"id": 1, "name": "East", "x": 100.0, "y": 0.0, "east": 1.0},
        ],
        buildings=BuildingColumns(
            x=np.array([-300.0, 250.0, 10.0]),
            y=np.array([-200.0, 120.0, 0.0]),
            width=np.array([20.0, 15.0, 30.0]),
            depth=np.array([20.0, 15.0, 30.0]),
            height=np.array([9.0, 12.0, 30.0]),
            rotation=np.array([0.0, 1.0, 2.0]),
            district=np.array([0, 1, 0]),
        ),
        roads=RoadColumns(
            points=np.array([2]),
            road_class=np.array([1]),
            x=np.array([-300.0, 250.0]),
            y=np.array([-200.0, 120.0]),
        ),
        residents=ResidentColumns(
            x=np.array([-290.0, 240.0, 12.0, 0.0]),
            y=np.array([-190.0, 110.0, 3.0, 0.0]),
            floor=np.array([0, 1, 2, 0]),
            building=np.array([0, 1, 2, 2]),
            segment=np.array([0, 4, 2, 1]),
        ),
        residents_engine=ResidentEngineColumns(
            hex=np.zeros(4, dtype=int),
            persona_ref=np.full(4, NO_PERSONA),
            weight=np.ones(4),
            tier=np.ones(4, dtype=np.uint8),
        ),
        stores=StoreColumns(
            x=np.array([0.0]),
            y=np.array([0.0]),
            hex=np.array([0]),
            radius_m=np.array([900.0]),
            district=np.array([1]),
        ),
        river=RiverColumns(
            offsets=np.array([0, 4]),
            x=np.array([-6000, 6000, 6000, -6000], dtype=np.float32),
            y=np.array([-6000, -6000, 6000, 6000], dtype=np.float32),
            kind=np.array([0]),
            rings=np.array([1]),
        ),
        graph=GraphColumns(
            offsets=np.array([0, 1, 3, 3, 4]),
            targets=np.array([1, 0, 2, 0]),
            kind=np.array([0, 1, 2, 3]),
            strength=np.array([1.0, 0.5, 0.5, 0.25]),
        ),
        seed=7,
        tile_level=2,
    )


def by_hand(config: SampleConfig):
    reader = PackReader(SMALL)
    residents = reader.residents()
    engine = reader.residents_engine()
    return sample_personas(
        residents.segment,
        engine.persona_ref,
        engine.weight,
        engine.tier,
        len(SEGMENTS),
        config,
        pack_entropy(reader.pack_hash),
    )


def same(a: CitySample, b: CitySample) -> bool:
    return (
        a.pack_hash == b.pack_hash
        and a.segments == b.segments
        and a.config == b.config
        and np.array_equal(a.sample.resident, b.sample.resident)
        and np.array_equal(a.sample.persona_ref, b.sample.persona_ref)
        and np.array_equal(a.sample.design_weight, b.sample.design_weight)
        and np.array_equal(a.district, b.district)
    )


def test_pack_entropy_splits_hash_into_eight_big_endian_words() -> None:
    digest = "00000001ffffffff00000010deadbeef00000000800000000000abcd12345678"
    assert pack_entropy(digest) == (
        1,
        0xFFFFFFFF,
        16,
        0xDEADBEEF,
        0,
        0x80000000,
        0xABCD,
        0x12345678,
    )


def test_pack_entropy_of_the_fixture_hash_matches_slicing() -> None:
    words = pack_entropy(SMALL_HASH)
    assert len(words) == 8
    assert words[0] == 0x79399336
    assert words[-1] == int(SMALL_HASH[56:], 16)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "0x" + "a" * 62,
        " " + "a" * 63,
        "a" * 63 + "\n",
        "a" * 32 + "É" + "a" * 31,
    ],
)
def test_pack_entropy_refuses_anything_but_64_lowercase_hex(bad: str) -> None:
    with pytest.raises(ValueError):
        pack_entropy(bad)


def test_fixture_pack_is_the_one_the_spec_describes() -> None:
    reader = PackReader(SMALL)
    assert reader.pack_hash == SMALL_HASH
    assert reader.manifest["segments"] == list(SEGMENTS)


def test_sample_city_equals_sample_personas_on_the_readers_arrays() -> None:
    for config in (SampleConfig(), LOOSE, SampleConfig(size=60, min_per_segment=5, by_tier=True)):
        city = sample_city(SMALL, config)
        expected = by_hand(config)
        got = city.sample
        assert np.array_equal(got.resident, expected.resident)
        assert np.array_equal(got.persona_ref, expected.persona_ref)
        assert np.array_equal(got.segment, expected.segment)
        assert (got.tier is None) == (expected.tier is None)
        assert got.tier is not None and expected.tier is not None
        assert np.array_equal(got.tier, expected.tier)
        assert np.array_equal(got.stratum, expected.stratum)
        assert np.array_equal(got.design_weight, expected.design_weight)
        assert got.strata == expected.strata
        assert city.config == config


def test_sample_city_carries_pack_hash_and_segment_names() -> None:
    city = sample_city(SMALL, LOOSE)
    assert city.pack_hash == SMALL_HASH
    assert city.segments == SEGMENTS
    assert isinstance(city.segments, tuple)


def test_default_config_is_used_when_none_is_given() -> None:
    assert sample_city(SMALL).config == SampleConfig()


def test_districts_follow_each_sampled_residents_building() -> None:
    reader = PackReader(SMALL)
    building = reader.residents().building
    district = reader.buildings().district
    city = sample_city(SMALL, LOOSE)
    assert len(city.sample) > 0
    assert city.district.dtype == np.uint8
    assert len(city.district) == len(city.sample)
    expected = district[building[city.sample.resident]]
    assert np.array_equal(city.district, expected)


def test_sampled_personas_are_real_and_unique() -> None:
    city = sample_city(SMALL, LOOSE)
    refs = city.sample.persona_ref
    assert (refs != NO_PERSONA).all()
    assert len(set(refs.tolist())) == len(refs)


def test_rows_are_json_ready_and_in_sample_order() -> None:
    city = sample_city(SMALL, LOOSE)
    rows = sample_rows(city)
    assert len(rows) == len(city.sample)
    json.loads(json.dumps(rows))
    assert [r["resident"] for r in rows] == city.sample.resident.tolist()
    for key in ("resident", "persona_ref", "tier", "district", "stratum"):
        assert all(type(r[key]) is int for r in rows)
    assert all(type(r["design_weight"]) is float for r in rows)
    assert all(type(r["segment"]) is str for r in rows)
    assert all(
        set(r)
        == {
            "resident",
            "persona_ref",
            "segment",
            "tier",
            "district",
            "stratum",
            "design_weight",
        }
        for r in rows
    )


def test_rows_map_every_field_from_the_sample() -> None:
    city = sample_city(SMALL, LOOSE)
    sample = city.sample
    assert sample.tier is not None
    for i, row in enumerate(sample_rows(city)):
        assert row["persona_ref"] == int(sample.persona_ref[i])
        assert row["segment"] == SEGMENTS[int(sample.segment[i])]
        assert row["tier"] == int(sample.tier[i])
        assert row["district"] == int(city.district[i])
        assert row["stratum"] == int(sample.stratum[i])
        assert row["design_weight"] == float(sample.design_weight[i])


def test_rows_report_tier_even_when_not_stratified() -> None:
    city = sample_city(SMALL, SampleConfig(size=60, min_per_segment=5, by_tier=False))
    assert city.sample.tier is not None
    assert all(r["tier"] in (0, 1, 2) for r in sample_rows(city))


def test_same_pack_twice_gives_identical_samples() -> None:
    assert same(sample_city(SMALL, LOOSE), sample_city(SMALL, LOOSE))


def test_different_config_seed_changes_the_sample() -> None:
    a = sample_city(SMALL, SampleConfig(size=60, min_per_segment=5, seed=1))
    b = sample_city(SMALL, SampleConfig(size=60, min_per_segment=5, seed=2))
    assert not np.array_equal(a.sample.resident, b.sample.resident)


def test_by_tier_stratifies_by_the_packs_tier() -> None:
    config = SampleConfig(size=60, min_per_segment=6, by_tier=True)
    city = sample_city(SMALL, config)
    sample = city.sample
    assert sample.tier is not None
    assert len(city.sample.strata) == len(SEGMENTS) * 3
    assert {(s.segment, s.tier) for s in sample.strata} == {
        (s, t) for s in range(len(SEGMENTS)) for t in range(3)
    }
    assert np.array_equal(sample.stratum, sample.segment.astype(int) * 3 + sample.tier)
    assert len(sample) == 60
    plain = sample_city(SMALL, SampleConfig(size=60, min_per_segment=6, by_tier=False))
    assert len(plain.sample.strata) == len(SEGMENTS)
    assert plain.sample.tier is not None


def test_by_tier_samples_match_the_packs_resident_tiers() -> None:
    reader = PackReader(SMALL)
    tiers = reader.residents_engine().tier
    city = sample_city(SMALL, SampleConfig(size=60, min_per_segment=6, by_tier=True))
    assert city.sample.tier is not None
    assert np.array_equal(city.sample.tier, tiers[city.sample.resident])


def test_toy_pack_gives_an_empty_sample(tmp_path: Path) -> None:
    out = tmp_path / "toy"
    write_pack(encode_pack(toy_pack()), out)
    city = sample_city(out)
    assert len(city.sample) == 0
    assert len(city.district) == 0
    assert city.segments == ("a0", "a1", "a2", "a3", "a4")
    assert sample_rows(city) == []


def test_toy_pack_with_min_floor_still_samples_empty(tmp_path: Path) -> None:
    out = tmp_path / "toy"
    write_pack(encode_pack(toy_pack()), out)
    assert len(sample_city(out, SampleConfig(size=10, min_per_segment=5)).sample) == 0


def test_missing_directory_raises_pack_error(tmp_path: Path) -> None:
    with pytest.raises(PackError):
        sample_city(tmp_path / "nowhere")


def test_corrupt_manifest_raises_pack_error(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b"not json")
    with pytest.raises(PackError):
        sample_city(tmp_path)
