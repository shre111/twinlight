from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import CityConfig, PersonasConfig
from twl_citybake.personas.assign import (
    NO_SPOUSE,
    Assignment,
    adults_per_household,
    assign_personas,
    free_draw_weights,
)
from twl_citybake.personas.pool import DATASET, MALE, PersonaPool, parquet_names, read_pool
from twl_citybake.personas.segments import check_categories, derive_segments, occupied_rows
from twl_citybake.rng import Stream, stream
from twl_citybake.store import read_table, table_digest, write_json, write_table

ATTRIBUTION = "Nemotron-Personas-India (NVIDIA), CC BY 4.0"
SEGMENT_TOLERANCE = 0.01
AGE_BAND_NAMES = ("18-24", "25-34", "35-44", "45-59", "60+")
AGE_BAND_STARTS = (25, 35, 45, 60)
OCCUPATION_GROUP_NAMES = ("not occupied", "occupied")


@dataclass(frozen=True)
class PersonaLayers:
    pool: dict[str, np.ndarray]
    vocab: dict[str, list[str]]
    assignment: dict[str, np.ndarray]
    report: dict[str, Any]
    meta: dict[str, Any]


def run_personas(population: Path, pool_path: Path, city: CityConfig) -> PersonaLayers:
    config = city.personas
    if config is None:
        raise ValueError(f"{city.slug} has no [personas] section")
    household = read_table(population / "residents")["household"]
    pool = read_pool(pool_path)
    if config.married not in pool.vocab["marital"]:
        raise ValueError(f"{pool_path} has no marital status {config.married!r}")
    married_code = pool.vocab["marital"].index(config.married)
    check_categories(pool, config)
    segments = derive_segments(pool, config)
    assignment = assign_personas(
        household, pool, segments, married_code, stream(city.seed, Stream.PERSONAS)
    )
    names = [rule.name for rule in config.segments]
    weights = free_draw_weights(adults_per_household(household), pool, married_code)
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
        report=_report(pool, config, segments, names, assignment, married, weights),
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
            },
            "segments_version": config.segments_version,
            "segments": names,
        },
    )


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
) -> dict[str, Any]:
    ref = assignment.persona_ref
    pool_shares = _shares(segments, len(names))
    resident_shares = _shares(segments[ref], len(names))
    differences = resident_shares - pool_shares
    largest = float(np.abs(differences).max())
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
    }


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
