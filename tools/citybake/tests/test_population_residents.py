import numpy as np
import pytest
from population_fixture import (
    CENSUS_COUNTS,
    CENSUS_SIZES,
    synthetic_buildings,
    synthetic_households_input,
)

from twl_citybake.population.households import Households, synthesise_households
from twl_citybake.population.residents import adults, expected_adults, sample_residents

ADULT_SHARE = 0.69
SHOPPER_SHARE = 0.27
TARGET = 500
STOREY_M = 3.2


@pytest.fixture(scope="module")
def households() -> Households:
    case = synthetic_households_input(11)
    return synthesise_households(
        case["placed"],
        case["marginal"],
        case["capacity"],
        case["district"],
        4,
        np.asarray(CENSUS_SIZES, np.int64),
        np.asarray(CENSUS_COUNTS, np.int64),
    )


@pytest.fixture(scope="module")
def buildings() -> dict[str, np.ndarray]:
    case = synthetic_households_input(11)
    table = synthetic_buildings(len(case["capacity"]))
    table["district"] = case["district"]
    return table


ADULTS_BY_SIZE = expected_adults(CENSUS_SIZES, CENSUS_COUNTS, ADULT_SHARE)


def adults_of(h: Households, seed: int = 7) -> np.ndarray:
    return adults(h, ADULTS_BY_SIZE, np.random.default_rng(seed))


def sample(
    h: Households, b: dict[str, np.ndarray], seed: int = 7, target: int = TARGET
) -> dict[str, np.ndarray]:
    return sample_residents(
        h, b, ADULTS_BY_SIZE, target, SHOPPER_SHARE, np.random.default_rng(seed)
    )


def test_resident_count_reaches_target_without_overshooting_a_household(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    count = len(out["household"])
    assert TARGET <= count < TARGET + adults_of(households, 7).max()


def test_columns_have_documented_dtypes_and_equal_lengths(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    dtypes = {
        "household": np.uint32,
        "building": np.uint32,
        "district": np.uint16,
        "bank": np.uint8,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "weight": np.float32,
    }
    assert {k: v.dtype for k, v in out.items()} == {k: np.dtype(t) for k, t in dtypes.items()}
    assert len({len(v) for v in out.values()}) == 1


def test_taken_households_are_the_prefix_of_the_seeded_permutation(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings, seed=7)
    rng = np.random.default_rng(7)
    per_household = adults(households, ADULTS_BY_SIZE, rng)
    order = rng.permutation(len(households.size))
    reached = np.cumsum(per_household[order])
    taken = np.sort(order[: int(np.searchsorted(reached, TARGET)) + 1])
    assert np.unique(out["household"]).tolist() == taken.tolist()


def test_members_of_a_household_are_contiguous_and_share_a_building(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    hh = out["household"].astype(np.int64)
    assert (np.diff(hh) >= 0).all()
    members = np.bincount(hh)
    taken = np.unique(hh)
    assert (members[taken] == adults_of(households, 7)[taken]).all()
    assert np.array_equal(out["building"], households.building[hh])
    assert np.array_equal(out["district"], households.district[hh])
    assert np.array_equal(out["bank"], buildings["bank"][out["building"]])


def test_members_of_a_household_stand_within_a_metre_of_each_other(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    hh = out["household"]
    for axis in ("x", "y", "z"):
        spread = np.zeros(hh.max() + 1)
        low = np.full(hh.max() + 1, np.inf)
        np.minimum.at(low, hh, out[axis].astype(np.float64))
        high = np.full(hh.max() + 1, -np.inf)
        np.maximum.at(high, hh, out[axis].astype(np.float64))
        taken = np.unique(hh)
        spread[taken] = high[taken] - low[taken]
        limit = 1e-3 if axis == "z" else 1.0 + 1e-3
        assert (spread <= limit).all()


def test_residents_stand_inside_their_buildings_oriented_box(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    b = out["building"]
    dx = out["x"].astype(np.float64) - buildings["x"][b]
    dy = out["y"].astype(np.float64) - buildings["y"][b]
    cos, sin = np.cos(buildings["rot"][b].astype(np.float64)), np.sin(buildings["rot"][b])
    along = dx * cos + dy * sin
    across = -dx * sin + dy * cos
    jitter = 0.5 * np.sqrt(2.0) + 0.01
    assert (np.abs(along) <= 0.4 * buildings["w"][b] + jitter).all()
    assert (np.abs(across) <= 0.4 * buildings["d"][b] + jitter).all()


def test_z_is_the_mid_height_of_a_floor_inside_the_building(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    storeys = np.maximum(1, np.round(buildings["height_m"][out["building"]] / STOREY_M))
    floor = out["z"].astype(np.float64) / STOREY_M - 0.5
    assert np.allclose(floor, np.round(floor), atol=1e-3)
    assert (np.round(floor) >= 0).all()
    assert (np.round(floor) <= storeys - 1).all()


def test_every_resident_carries_the_same_weight_and_weights_sum_to_shoppers(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    out = sample(households, buildings)
    total_adults = adults_of(households, 7).sum()
    assert (out["weight"] == out["weight"][0]).all()
    assert float(out["weight"].sum(dtype=np.float64)) == pytest.approx(
        SHOPPER_SHARE * total_adults, rel=1e-4
    )


def test_same_seed_gives_identical_arrays(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    a, b = sample(households, buildings, seed=3), sample(households, buildings, seed=3)
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_different_seed_gives_different_residents(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    a, b = sample(households, buildings, seed=3), sample(households, buildings, seed=4)
    assert not np.array_equal(a["household"], b["household"])
    assert not np.array_equal(a["x"], b["x"])


def test_sampling_does_not_modify_its_inputs(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    before = {k: v.copy() for k, v in buildings.items()}
    sample(households, buildings)
    assert all(np.array_equal(before[k], buildings[k]) for k in before)


def test_a_larger_target_yields_more_residents(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    small = sample(households, buildings, target=200)
    large = sample(households, buildings, target=1000)
    assert len(small["household"]) < len(large["household"])
    assert len(large["household"]) >= 1000


def test_expected_adults_for_ahmedabad_census_has_r_near_0_47_and_hits_the_share() -> None:
    mean = expected_adults(CENSUS_SIZES, CENSUS_COUNTS, ADULT_SHARE)
    assert mean[1] == 1 and mean[2] == 2
    r = (mean[3] - 2) / 1
    assert r == pytest.approx(0.470, abs=0.002)
    for size in CENSUS_SIZES[2:]:
        assert mean[size] == pytest.approx(2 + r * (size - 2))
    sizes, counts = np.asarray(CENSUS_SIZES), np.asarray(CENSUS_COUNTS)
    weighted = sum(n * mean[int(s)] for s, n in zip(sizes, counts, strict=True))
    assert weighted / (counts * sizes).sum() == pytest.approx(ADULT_SHARE, abs=1e-9)


def test_expected_adults_clips_r_to_the_unit_interval() -> None:
    sizes, counts = [1, 2, 4], [10, 10, 10]
    assert expected_adults(sizes, counts, 1.0)[4] == 4
    assert expected_adults(sizes, counts, 0.0)[4] == 2
    assert expected_adults(sizes, counts, 0.0)[1] == 1


def test_two_person_households_always_give_two_residents() -> None:
    twos = Households(
        building=np.zeros(300, np.uint32),
        size=np.full(300, 2, np.uint8),
        district=np.zeros(300, np.uint16),
        report={},
    )
    got = adults(twos, ADULTS_BY_SIZE, np.random.default_rng(1))
    assert (got == 2).all()


def test_mean_adults_per_size_is_close_to_the_expectation(households: Households) -> None:
    many = Households(
        building=np.zeros(40000, np.uint32),
        size=np.tile(np.asarray(CENSUS_SIZES, np.uint8), 40000 // len(CENSUS_SIZES) + 1)[:40000],
        district=np.zeros(40000, np.uint16),
        report={},
    )
    got = adults(many, ADULTS_BY_SIZE, np.random.default_rng(5))
    for size, mean in ADULTS_BY_SIZE.items():
        assert got[many.size == size].mean() == pytest.approx(mean, abs=0.03)
        assert np.isin(got[many.size == size], [np.floor(mean), np.ceil(mean)]).all()


def test_sampling_is_deterministic_for_one_seed_in_draw_order(
    households: Households, buildings: dict[str, np.ndarray]
) -> None:
    a = sample(households, buildings, seed=9)
    b = sample(households, buildings, seed=9)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert len(a["household"]) == adults_of(households, 9)[np.unique(a["household"])].sum()
