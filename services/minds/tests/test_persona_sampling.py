from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import ArrayLike, NDArray

from twl_minds.personas.sampling import PersonaSample, SampleConfig, sample_personas

NO_PERSONA = 0xFFFFFFFF
Group = tuple[int, int, float, int, int]


def city(
    groups: Sequence[Group],
) -> tuple[NDArray[np.uint8], NDArray[np.uint32], NDArray[np.float64], NDArray[np.uint8]]:
    """Each group is (segment, tier, weight, residents per persona, personas)."""
    segment: list[int] = []
    tier: list[int] = []
    weight: list[float] = []
    ref: list[int] = []
    next_ref = 0
    for seg, t, w, per, count in groups:
        for _ in range(count):
            for _ in range(per):
                segment.append(seg)
                tier.append(t)
                weight.append(w)
                ref.append(next_ref)
            next_ref += 1
    return (
        np.array(segment, dtype=np.uint8),
        np.array(ref, dtype=np.uint32),
        np.array(weight, dtype=np.float64),
        np.array(tier, dtype=np.uint8),
    )


def run(
    groups: Sequence[Group],
    segments: int,
    use_tier: bool = False,
    entropy: Sequence[int] = (),
    **config: Any,
) -> PersonaSample:
    segment, ref, weight, tier = city(groups)
    return sample_personas(
        segment, ref, weight, tier if use_tier else None, segments, SampleConfig(**config), entropy
    )


def sampled_per_stratum(out: PersonaSample) -> list[int]:
    return [s.sampled for s in out.strata]


def oracle_residents(
    segment: NDArray[Any],
    persona_ref: NDArray[Any],
    weight: NDArray[Any],
    tier: NDArray[Any] | None,
    out: PersonaSample,
    seed: int,
    entropy: Sequence[int] = (),
) -> list[int]:
    n = len(segment)
    rng = np.random.default_rng(np.random.SeedSequence([seed, *entropy]))
    priority = rng.random(n)
    best: dict[int, int] = {}
    for i in range(n):
        if persona_ref[i] == NO_PERSONA or weight[i] <= 0:
            continue
        ref = int(persona_ref[i])
        if ref not in best or priority[i] < priority[best[ref]]:
            best[ref] = i
    chosen: list[int] = []
    for k, stratum in enumerate(out.strata):
        members = [
            i
            for i in best.values()
            if segment[i] == stratum.segment
            and (stratum.tier is None or (tier is not None and tier[i] == stratum.tier))
        ]
        assert len(members) == stratum.candidates, k
        members.sort(key=lambda i: (priority[i], i))
        chosen.extend(members[: stratum.sampled])
    return sorted(chosen)


def test_largest_remainder_splits_floors_then_weight_share() -> None:
    out = run([(0, 0, 1.0, 1, 4), (1, 0, 1.0, 1, 6)], 2, size=6, min_per_segment=1)
    assert sampled_per_stratum(out) == [3, 3]
    assert len(out.resident) == 6


def test_remainder_over_floors_follows_population_weight() -> None:
    out = run([(0, 0, 1.0, 1, 100), (1, 0, 1.0, 1, 200)], 2, size=30, min_per_segment=3)
    assert sampled_per_stratum(out) == [11, 19]


def test_leftover_ties_go_to_lower_segment() -> None:
    out = run([(s, 0, 1.0, 1, 10) for s in range(3)], 3, size=10, min_per_segment=0)
    assert sampled_per_stratum(out) == [4, 3, 3]


def test_segment_capped_by_candidates_hands_rest_to_others() -> None:
    out = run([(0, 0, 10.0, 1, 2), (1, 0, 1.0, 1, 20)], 2, size=10, min_per_segment=0)
    assert sampled_per_stratum(out) == [2, 8]


def test_floor_is_clipped_to_candidates_of_small_segment() -> None:
    out = run([(0, 0, 1.0, 1, 2), (1, 0, 1.0, 1, 20)], 2, size=10, min_per_segment=5)
    assert sampled_per_stratum(out) == [2, 8]


def test_size_beyond_distinct_personas_takes_every_persona() -> None:
    out = run([(0, 0, 1.0, 3, 4), (1, 0, 1.0, 2, 5)], 2, size=300)
    assert len(out.resident) == 9
    assert sorted(out.persona_ref.tolist()) == list(range(9))


def test_empty_segment_is_listed_and_gets_nothing() -> None:
    out = run([(0, 0, 1.0, 1, 5), (2, 0, 1.0, 1, 5)], 3, size=6, min_per_segment=2)
    assert [s.segment for s in out.strata] == [0, 1, 2]
    assert (out.strata[1].residents, out.strata[1].candidates, out.strata[1].sampled) == (0, 0, 0)
    assert sampled_per_stratum(out) == [3, 0, 3]


def test_by_tier_splits_segment_count_with_ties_to_lower_tier() -> None:
    groups = [(0, 0, 1.0, 1, 3), (0, 1, 1.0, 1, 3), (0, 2, 1.0, 1, 4)]
    out = run(groups, 1, use_tier=True, by_tier=True, size=5, min_per_segment=0)
    assert [(s.segment, s.tier) for s in out.strata] == [(0, 0), (0, 1), (0, 2)]
    assert sampled_per_stratum(out) == [2, 1, 2]


def test_by_tier_strata_are_segment_major() -> None:
    out = run([(1, 2, 1.0, 1, 3)], 2, use_tier=True, by_tier=True, size=3, min_per_segment=0)
    assert [(s.segment, s.tier) for s in out.strata] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]  # fmt: skip
    assert sampled_per_stratum(out) == [0, 0, 0, 0, 0, 3]


def test_by_tier_cell_given_nothing_leaves_its_weight_uncovered() -> None:
    groups = [(0, 0, 1.0, 1, 3), (0, 1, 1.0, 1, 3), (0, 2, 1.0, 1, 4)]
    out = run(groups, 1, use_tier=True, by_tier=True, size=2, min_per_segment=0)
    assert sampled_per_stratum(out) == [1, 0, 1]
    assert out.design_weight.sum() == pytest.approx(7.0)


def test_by_tier_cell_never_exceeds_its_candidates() -> None:
    groups = [(0, 0, 1.0, 1, 1), (0, 1, 50.0, 1, 30)]
    out = run(groups, 1, use_tier=True, by_tier=True, size=20, min_per_segment=0)
    assert out.strata[0].sampled <= 1
    assert sum(sampled_per_stratum(out)) == 20


def test_stratified_by_segment_when_tier_given_but_by_tier_false() -> None:
    out = run([(0, 1, 1.0, 1, 5)], 1, use_tier=True, size=3, min_per_segment=0)
    assert len(out.strata) == 1
    assert out.strata[0].tier is None
    assert out.tier is not None and out.tier.tolist() == [1, 1, 1]


def test_tier_is_none_when_not_given() -> None:
    out = run([(0, 0, 1.0, 1, 5)], 1, size=3, min_per_segment=0)
    assert out.tier is None


def test_changing_only_tier_keeps_residents_when_not_by_tier() -> None:
    segment, ref, weight, _ = city([(0, 0, 1.0, 2, 200), (1, 0, 2.0, 2, 100)])
    rng = np.random.default_rng(5)
    t1 = rng.integers(0, 3, len(segment)).astype(np.uint8)
    t2 = rng.integers(0, 3, len(segment)).astype(np.uint8)
    cfg = SampleConfig(size=60, min_per_segment=10, seed=3)
    a = sample_personas(segment, ref, weight, t1, 2, cfg)
    b = sample_personas(segment, ref, weight, t2, 2, cfg)
    np.testing.assert_array_equal(a.resident, b.resident)


def test_outputs_ordered_by_resident_with_matching_dtypes() -> None:
    out = run(
        [(0, 1, 1.0, 3, 50), (1, 2, 1.0, 3, 50)], 2, use_tier=True, size=40, min_per_segment=0
    )
    assert np.all(np.diff(out.resident.astype(np.int64)) > 0)
    assert out.resident.dtype == np.uint32
    assert out.persona_ref.dtype == np.uint32
    assert out.segment.dtype == np.uint8
    assert out.stratum.dtype == np.uint16
    assert out.design_weight.dtype == np.float64


def test_sample_matches_priority_oracle_for_seed_and_entropy() -> None:
    groups = [(0, 0, 1.0, 5, 60), (1, 1, 0.5, 3, 80), (2, 2, 2.0, 1, 40)]
    segment, ref, weight, tier = city(groups)
    cfg = SampleConfig(size=50, min_per_segment=5, seed=11)
    out = sample_personas(segment, ref, weight, tier, 3, cfg, entropy=(7, 9))
    expected = oracle_residents(segment, ref, weight, tier, out, 11, (7, 9))
    assert out.resident.tolist() == expected


def test_representative_is_lowest_priority_eligible_resident_of_persona() -> None:
    segment = np.array([0, 1], dtype=np.uint8)
    ref = np.array([4, 4], dtype=np.uint32)
    weight = np.array([1.0, 1.0])
    cfg = SampleConfig(size=5, min_per_segment=0, seed=2)
    out = sample_personas(segment, ref, weight, None, 2, cfg)
    priority = np.random.default_rng(np.random.SeedSequence([2])).random(2)
    winner = int(np.argmin(priority))
    assert out.resident.tolist() == [winner]
    assert out.segment.tolist() == [winner]
    assert [s.residents for s in out.strata] == [1, 1]
    assert [s.candidates for s in out.strata] == [int(winner == 0), int(winner == 1)]


def test_ineligible_residents_are_neither_sampled_nor_counted() -> None:
    segment = np.array([0, 0, 0, 0], dtype=np.uint8)
    ref = np.array([1, NO_PERSONA, 2, 2], dtype=np.uint32)
    weight = np.array([1.0, 5.0, 0.0, 3.0])
    out = sample_personas(segment, ref, weight, None, 1, SampleConfig(size=10, min_per_segment=0))
    assert out.resident.tolist() == [0, 3]
    assert out.strata[0].residents == 2
    assert out.strata[0].weight == pytest.approx(4.0)
    assert out.strata[0].candidates == 2


def test_persona_with_only_ineligible_residents_is_not_a_candidate() -> None:
    segment = np.zeros(3, dtype=np.uint8)
    ref = np.array([1, 2, 2], dtype=np.uint32)
    weight = np.array([1.0, 0.0, 0.0])
    out = sample_personas(segment, ref, weight, None, 1, SampleConfig(size=10, min_per_segment=0))
    assert out.persona_ref.tolist() == [1]


def test_design_weights_cover_total_eligible_weight_when_all_strata_sampled() -> None:
    out = run([(0, 0, 1.0, 2, 30), (1, 0, 3.0, 2, 30)], 2, size=20, min_per_segment=5)
    assert out.design_weight.sum() == pytest.approx(30 * 2 * 1.0 + 30 * 2 * 3.0)


def test_design_weight_is_stratum_weight_over_stratum_sample_size() -> None:
    out = run([(0, 0, 1.0, 4, 30), (1, 0, 2.0, 1, 30)], 2, size=20, min_per_segment=5)
    for i, k in enumerate(out.stratum):
        s = out.strata[int(k)]
        assert out.design_weight[i] == pytest.approx(s.weight / s.sampled)


def test_design_weights_skip_unsampled_strata() -> None:
    out = run([(0, 0, 1.0, 1, 10), (1, 0, 4.0, 1, 10)], 3, size=5, min_per_segment=1)
    covered = sum(s.weight for s in out.strata if s.sampled > 0)
    assert out.design_weight.sum() == pytest.approx(covered)
    assert out.strata[2].sampled == 0


def test_same_inputs_give_identical_sample() -> None:
    segment, ref, weight, tier = city([(0, 0, 1.0, 3, 400), (1, 1, 2.0, 3, 400)])
    cfg = SampleConfig(size=100, min_per_segment=10, seed=4)
    a = sample_personas(segment, ref, weight, tier, 2, cfg, (1, 2))
    b = sample_personas(segment, ref, weight, tier, 2, cfg, (1, 2))
    np.testing.assert_array_equal(a.resident, b.resident)
    np.testing.assert_array_equal(a.design_weight, b.design_weight)
    assert a.strata == b.strata


def test_other_seed_or_entropy_gives_different_sample() -> None:
    segment, ref, weight, _ = city([(0, 0, 1.0, 3, 400), (1, 0, 2.0, 3, 400)])

    def draw(seed: int, entropy: Sequence[int]) -> list[int]:
        cfg = SampleConfig(size=100, min_per_segment=10, seed=seed)
        return sample_personas(segment, ref, weight, None, 2, cfg, entropy).resident.tolist()

    base = draw(0, ())
    assert draw(1, ()) != base
    assert draw(0, (123,)) != base
    assert draw(0, (123,)) != draw(0, (124,))


def test_selection_rate_rises_with_resident_count() -> None:
    segment, ref, weight, _ = city([(0, 0, 1.0, 1, 20), (0, 0, 1.0, 8, 20)])
    light = set(range(20))
    heavy_hits = light_hits = 0
    for seed in range(60):
        cfg = SampleConfig(size=10, min_per_segment=0, seed=seed)
        picked = sample_personas(segment, ref, weight, None, 1, cfg).persona_ref.tolist()
        light_hits += sum(p in light for p in picked)
        heavy_hits += sum(p not in light for p in picked)
    assert heavy_hits > 2 * light_hits


@pytest.mark.parametrize(
    "override",
    [
        {"weight": [1.0, 1.0]},
        {"persona_ref": [0, 1]},
        {"tier": [0, 0]},
        {"segment": [[0, 0], [0, 0]]},
        {"segment": [0, 2, 0]},
        {"segment": [0, -1, 0]},
        {"tier": [0, 3, 0]},
        {"tier": [0, -1, 0]},
        {"weight": [1.0, float("nan"), 1.0]},
        {"weight": [1.0, float("inf"), 1.0]},
        {"weight": [1.0, -0.5, 1.0]},
        {"config": SampleConfig(size=0)},
        {"config": SampleConfig(size=-3)},
        {"config": SampleConfig(min_per_segment=-1)},
        {"config": SampleConfig(by_tier=True), "tier": None},
    ],
    ids=lambda o: str(sorted(o)),
)
def test_invalid_input_is_refused(override: dict[str, Any]) -> None:
    args: dict[str, Any] = {
        "segment": [0, 1, 0],
        "persona_ref": [0, 1, 2],
        "weight": [1.0, 1.0, 1.0],
        "tier": [0, 1, 2],
        "segments": 2,
        "config": SampleConfig(size=3, min_per_segment=0),
    }
    args.update(override)
    with pytest.raises(ValueError):
        sample_personas(**args)


def test_floors_exceeding_target_are_refused() -> None:
    with pytest.raises(ValueError):
        run([(0, 0, 1.0, 1, 5), (1, 0, 1.0, 1, 5)], 2, size=8, min_per_segment=5)


def test_floors_exactly_filling_target_are_accepted() -> None:
    out = run([(0, 0, 1.0, 1, 5), (1, 0, 1.0, 1, 5)], 2, size=10, min_per_segment=5)
    assert sampled_per_stratum(out) == [5, 5]


def test_floors_are_clipped_to_candidates_before_comparing_with_target() -> None:
    out = run([(0, 0, 1.0, 1, 3), (1, 0, 1.0, 1, 3)], 2, size=300, min_per_segment=10)
    assert sampled_per_stratum(out) == [3, 3]


def test_floors_beyond_a_small_size_are_refused_even_with_few_candidates() -> None:
    with pytest.raises(ValueError):
        run([(0, 0, 1.0, 1, 3), (1, 0, 1.0, 1, 3)], 2, size=4, min_per_segment=3)


def random_city(
    seed: int, n: int, personas: int, segments: int
) -> tuple[NDArray[np.uint8], NDArray[np.uint32], NDArray[np.float64], NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    segment = rng.integers(0, segments, n).astype(np.uint8)
    ref = rng.integers(0, personas, n).astype(np.uint32)
    ref[rng.random(n) < 0.1] = NO_PERSONA
    weight = rng.random(n) * 3
    weight[rng.random(n) < 0.1] = 0.0
    tier = rng.integers(0, 3, n).astype(np.uint8)
    return segment, ref, weight, tier


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(1, 400),
    personas=st.integers(1, 120),
    segments=st.integers(1, 5),
    size=st.integers(25, 150),
    floor=st.integers(0, 5),
    by_tier=st.booleans(),
)
def test_sample_invariants_hold_on_random_cities(
    seed: int, n: int, personas: int, segments: int, size: int, floor: int, by_tier: bool
) -> None:
    segment, ref, weight, tier = random_city(seed, n, personas, segments)
    cfg = SampleConfig(size=size, min_per_segment=floor, by_tier=by_tier, seed=seed)
    out = sample_personas(segment, ref, weight, tier, segments, cfg)

    eligible = (ref != NO_PERSONA) & (weight > 0)
    distinct = len(set(ref[eligible].tolist()))
    assert len(out.resident) == min(size, distinct)
    assert len(set(out.persona_ref.tolist())) == len(out.resident)
    assert np.all(eligible[out.resident])
    assert np.array_equal(ref[out.resident], out.persona_ref)
    assert out.resident.tolist() == oracle_residents(segment, ref, weight, tier, out, seed)

    assert len(out.strata) == segments * (3 if by_tier else 1)
    assert sum(s.sampled for s in out.strata) == len(out.resident)
    assert all(s.sampled <= s.candidates for s in out.strata)
    assert sum(s.residents for s in out.strata) == int(eligible.sum())

    per_segment = [0] * segments
    cand_segment = [0] * segments
    for s in out.strata:
        per_segment[s.segment] += s.sampled
        cand_segment[s.segment] += s.candidates
    for sampled, cands in zip(per_segment, cand_segment, strict=True):
        if cands > 0:
            assert sampled >= min(floor, cands)

    covered = sum(s.weight for s in out.strata if s.sampled > 0)
    assert float(out.design_weight.sum()) == pytest.approx(covered)


def test_plain_lists_are_accepted_as_inputs() -> None:
    segment: ArrayLike = [0, 0]
    out = sample_personas(segment, [0, 1], [1, 1], None, 1, SampleConfig(size=2, min_per_segment=0))
    assert len(out.resident) == 2
