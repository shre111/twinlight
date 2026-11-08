from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import CityConfig, PersonasConfig, TierTiltConfig
from twl_citybake.personas.assign import (
    NO_SPOUSE,
    Assignment,
    adults_per_household,
    assign_personas,
    free_draw_weights,
)
from twl_citybake.personas.pool import (
    DATASET,
    FEMALE,
    MALE,
    PersonaPool,
    parquet_names,
    read_pool,
)
from twl_citybake.personas.segments import check_categories, derive_segments, occupied_rows
from twl_citybake.personas.tilt import TIERS, odds_ratios, tier_education_shares, tilted_weights
from twl_citybake.rng import Stream, stream
from twl_citybake.store import read_table, table_digest, write_json, write_table

ATTRIBUTION = "Nemotron-Personas-India (NVIDIA), CC BY 4.0"
SEGMENT_TOLERANCE = 0.01
AGE_BAND_NAMES = ("18-24", "25-34", "35-44", "45-59", "60+")
AGE_BAND_STARTS = (25, 35, 45, 60)
OCCUPATION_GROUP_NAMES = ("not occupied", "occupied")
SEXES = (("female", FEMALE), ("male", MALE))


@dataclass(frozen=True)
class PersonaLayers:
    pool: dict[str, np.ndarray]
    vocab: dict[str, list[str]]
    assignment: dict[str, np.ndarray]
    report: dict[str, Any]
    meta: dict[str, Any]


@dataclass(frozen=True)
class _Tilt:
    config: TierTiltConfig
    slot_tier: NDArray[np.uint8]
    ten_plus: NDArray[np.bool_]
    mapped: dict[str, tuple[float, float, float]]
    ratios: dict[str, tuple[float, float, float]]
    weights: NDArray[np.float64]


def run_personas(population: Path, pool_path: Path, city: CityConfig) -> PersonaLayers:
    config = city.personas
    if config is None:
        raise ValueError(f"{city.slug} has no [personas] section")
    residents = read_table(population / "residents")
    household = residents["household"]
    pool = read_pool(pool_path)
    if config.married not in pool.vocab["marital"]:
        raise ValueError(f"{pool_path} has no marital status {config.married!r}")
    married_code = pool.vocab["marital"].index(config.married)
    check_categories(pool, config)
    segments = derive_segments(pool, config)
    weights = free_draw_weights(adults_per_household(household), pool, married_code)
    tilt = (
        None
        if config.tier_tilt is None
        else _tier_tilt(city, config.tier_tilt, population, residents, pool, married_code, weights)
    )
    assignment = assign_personas(
        household,
        pool,
        segments,
        married_code,
        stream(city.seed, Stream.PERSONAS),
        slot_tier=None if tilt is None else tilt.slot_tier,
        free_weights=None if tilt is None else tilt.weights,
        spouse_weights=None if tilt is None else tilt.weights,
    )
    names = [rule.name for rule in config.segments]
    married = pool.marital == married_code
    return PersonaLayers(
        pool={
            "uuid": pool.uuid,
            "sex": pool.sex,
            "age": pool.age,
            "marital": pool.marital,
            "education": pool.education,
            "occupation": pool.occupation,
            "language": pool.language,
            "segment": segments,
        },
        vocab={**pool.vocab, "segment": names},
        assignment={
            "persona_ref": assignment.persona_ref,
            "segment": segments[assignment.persona_ref],
            "spouse": assignment.spouse,
        },
        report=_report(pool, config, segments, names, assignment, married, weights, tilt),
        meta={
            "city": city.slug,
            "seed": city.seed,
            "attribution": [ATTRIBUTION],
            "population_residents_sha256": table_digest(population / "residents"),
            "source": {
                "dataset": DATASET,
                "revision": config.revision,
                "state": config.state,
                "district": config.district,
                "parquet": parquet_names(),
                "tier_tilt": None if config.tier_tilt is None else asdict(config.tier_tilt),
            },
            "segments_version": config.segments_version,
            "segments": names,
        },
    )


def _tier_tilt(
    city: CityConfig,
    config: TierTiltConfig,
    population: Path,
    residents: dict[str, np.ndarray],
    pool: PersonaPool,
    married_code: int,
    weights: NDArray[np.float64],
) -> _Tilt:
    if "tier" not in residents:
        raise ValueError(
            f"{population / 'residents'} has no tier column; rerun `citybake population "
            f"{city.slug}`"
        )
    if city.tier is None:
        raise ValueError(f"{city.slug} has [personas.tier_tilt] but no [tier] to tilt by")
    vocab = pool.vocab["education"]
    if missing := [name for name in config.education_10plus if name not in vocab]:
        raise ValueError(
            f"personas.tier_tilt.education_10plus {missing[0]!r} is not an education level "
            "in the pool"
        )
    ten_plus = np.isin(pool.education, [vocab.index(name) for name in config.education_10plus])
    slot_tier = residents["tier"]
    if len(slot_tier) and int(slot_tier.max()) >= TIERS:
        raise ValueError(f"residents' tier must be in 0..{TIERS - 1}")
    counts = np.bincount(slot_tier, minlength=TIERS)
    tier_share = counts / counts.sum() if counts.sum() else np.full(TIERS, 1 / TIERS)
    mapped = {
        name: tier_education_shares(
            config.quintile_population, by_quintile, city.tier.low_share, city.tier.high_share
        )
        for name, by_quintile in (("female", config.female_10plus), ("male", config.male_10plus))
    }
    ratios = {name: odds_ratios(shares) for name, shares in mapped.items()}
    tilted = tilted_weights(
        weights, pool, married_code, ten_plus, tier_share, ratios["female"], ratios["male"]
    )
    return _Tilt(config, slot_tier, ten_plus, mapped, ratios, tilted)


def write_persona_layers(layers: PersonaLayers, out_dir: Path) -> None:
    write_table(out_dir / "pool", layers.pool)
    write_json(out_dir / "vocab.json", layers.vocab)
    write_table(out_dir / "assignment", layers.assignment)
    write_json(out_dir / "report.json", layers.report)
    write_json(out_dir / "layer.json", layers.meta)


def _report(
    pool: PersonaPool,
    config: PersonasConfig,
    segments: NDArray[np.uint16],
    names: list[str],
    assignment: Assignment,
    married: NDArray[np.bool_],
    weights: NDArray[np.float64],
    tilt: _Tilt | None,
) -> dict[str, Any]:
    ref = assignment.persona_ref
    pool_shares = _shares(segments, len(names))
    resident_shares = _shares(segments[ref], len(names))
    differences = resident_shares - pool_shares
    largest = float(np.abs(differences).max())
    tilt_report = None if tilt is None else _tilt_report(pool, tilt, segments, names, assignment)
    return {
        "pool": {"rows": len(pool.uuid), "state": config.state, "district": config.district},
        "segments": {
            name: {
                "pool_share": _share(pool_shares[i]),
                "resident_share": _share(resident_shares[i]),
                "difference": _share(differences[i]),
            }
            for i, name in enumerate(names)
        },
        "marginals": {
            "sex": _marginal(pool.sex, ref, pool.vocab["sex"]),
            "age_band": _marginal(
                np.searchsorted(AGE_BAND_STARTS, pool.age, side="right"), ref, AGE_BAND_NAMES
            ),
            "education": _marginal(pool.education, ref, pool.vocab["education"]),
            "occupation_group": _marginal(
                occupied_rows(pool, config).astype(np.uint8), ref, OCCUPATION_GROUP_NAMES
            ),
        },
        "couples": _couples(pool, assignment),
        "married": {
            "pool_share": _share(married.mean()),
            "resident_share": _share(married[ref].mean() if len(ref) else 0.0),
            "free_draw_married_probability": _share(weights[married].sum()),
        },
        "max_segment_difference": _share(largest),
        "within_tolerance": largest <= SEGMENT_TOLERANCE,
        "tier_tilt": tilt_report,
    }


def _tilt_report(
    pool: PersonaPool,
    tilt: _Tilt,
    segments: NDArray[np.uint16],
    names: list[str],
    assignment: Assignment,
) -> dict[str, Any]:
    ref = assignment.persona_ref
    tier = tilt.slot_tier
    sex = pool.sex[ref]
    ten_plus = tilt.ten_plus[ref]
    spouse_slot = (assignment.spouse != NO_SPOUSE) & (assignment.spouse < np.arange(len(ref)))
    everyone = np.ones(len(ref), np.bool_)
    return {
        "source": tilt.config.source,
        "mapped": {name: [_share(v) for v in shares] for name, shares in tilt.mapped.items()},
        "odds_ratios": {name: [_share(v) for v in ratios] for name, ratios in tilt.ratios.items()},
        "pool_10plus": {
            name: _share(_mean(tilt.ten_plus[pool.sex == code])) for name, code in SEXES
        },
        "free_draws_10plus": {
            name: _by_tier(ten_plus, tier, ~spouse_slot & (sex == code)) for name, code in SEXES
        },
        "spouses_10plus": {
            name: _by_tier(ten_plus, tier, spouse_slot & (sex == code)) for name, code in SEXES
        },
        "residents_10plus": {name: _by_tier(ten_plus, tier, sex == code) for name, code in SEXES},
        "segments_by_tier": {
            name: _by_tier(segments[ref] == i, tier, everyone) for i, name in enumerate(names)
        },
    }


def _by_tier(
    flag: NDArray[np.bool_], tier: NDArray[np.uint8], within: NDArray[np.bool_]
) -> list[float]:
    """The share of `within` slots in each tier that have `flag`; 0.0 for an empty tier."""
    total = np.bincount(tier[within], minlength=TIERS).tolist()
    hits = np.bincount(tier[within & flag], minlength=TIERS).tolist()
    return [_share(h / t) if t else 0.0 for h, t in zip(hits, total, strict=True)]


def _mean(flags: NDArray[np.bool_]) -> float:
    return float(flags.mean()) if len(flags) else 0.0


def _shares(codes: NDArray[np.integer], categories: int) -> NDArray[np.float64]:
    return np.bincount(codes, minlength=categories) / max(len(codes), 1)


def _share(value: float | np.floating) -> float:
    return round(float(value), 4)


def _marginal(
    codes: NDArray[np.integer], ref: NDArray[np.uint32], names: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    pool_shares = _shares(codes, len(names))
    resident_shares = _shares(codes[ref], len(names))
    return {
        "categories": {
            name: {
                "pool_share": _share(pool_shares[i]),
                "resident_share": _share(resident_shares[i]),
            }
            for i, name in enumerate(names)
        },
        "total_variation": round(float(np.abs(pool_shares - resident_shares).sum() / 2), 3),
    }


def _couples(pool: PersonaPool, assignment: Assignment) -> dict[str, Any]:
    in_couple = assignment.spouse != NO_SPOUSE
    first = np.flatnonzero(in_couple & (assignment.spouse > np.arange(len(in_couple))))
    if len(first) == 0:
        return {
            "count": 0,
            "residents_in_couples": 0,
            "gap_p10": 0.0,
            "gap_p50": 0.0,
            "gap_p90": 0.0,
            "husband_older_or_equal_share": 0.0,
        }
    age = pool.age[assignment.persona_ref].astype(np.int64)
    one, other = first, assignment.spouse[first].astype(np.intp)
    male_first = pool.sex[assignment.persona_ref[one]] == MALE
    gap = np.where(male_first, age[one] - age[other], age[other] - age[one])
    p10, p50, p90 = np.percentile(gap, [10, 50, 90])
    return {
        "count": len(first),
        "residents_in_couples": int(in_couple.sum()),
        "gap_p10": round(float(p10), 3),
        "gap_p50": round(float(p50), 3),
        "gap_p90": round(float(p90), 3),
        "husband_older_or_equal_share": _share((gap >= 0).mean()),
    }
