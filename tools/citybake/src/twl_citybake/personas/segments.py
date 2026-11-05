import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import PersonasConfig, SegmentRule
from twl_citybake.personas.pool import PersonaPool


def derive_segments(pool: PersonaPool, config: PersonasConfig) -> NDArray[np.uint16]:
    """Per pool row, the index of the first rule it matches; the last rule catches the rest."""
    occupied = occupied_rows(pool, config)
    segment = np.zeros(len(pool.uuid), np.uint16)
    unmatched = np.ones(len(pool.uuid), bool)
    for index, rule in enumerate(config.segments):
        match = unmatched & _matches(rule, pool, occupied)
        segment[match] = index
        unmatched &= ~match
    return segment


def check_categories(pool: PersonaPool, config: PersonasConfig) -> None:
    """Every name in the table must exist in the pool: a misspelt one would match nobody without
    any error, and a misspelt `no_occupation` would mark everyone as occupied."""
    if config.no_occupation not in pool.vocab["occupation"]:
        raise ValueError(f"the pool has no occupation {config.no_occupation!r}")
    for rule in config.segments:
        for column, names in (("marital", rule.marital), ("education", rule.education)):
            for name in names:
                if name not in pool.vocab[column]:
                    raise ValueError(f"segment {rule.name}: the pool has no {column} {name!r}")


def occupied_rows(pool: PersonaPool, config: PersonasConfig) -> NDArray[np.bool_]:
    return ~_in_categories(pool.occupation, pool.vocab["occupation"], (config.no_occupation,))


def _matches(
    rule: SegmentRule, pool: PersonaPool, occupied: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    match = np.ones(len(pool.uuid), bool)
    if rule.min_age is not None:
        match &= pool.age >= rule.min_age
    if rule.max_age is not None:
        match &= pool.age <= rule.max_age
    if rule.sex is not None:
        match &= _in_categories(pool.sex, pool.vocab["sex"], (rule.sex,))
    if rule.marital:
        match &= _in_categories(pool.marital, pool.vocab["marital"], rule.marital)
    if rule.education:
        match &= _in_categories(pool.education, pool.vocab["education"], rule.education)
    if rule.occupied is not None:
        match &= occupied == rule.occupied
    return match


def _in_categories(
    codes: NDArray[np.unsignedinteger], vocab: list[str], names: tuple[str, ...]
) -> NDArray[np.bool_]:
    return np.isin(codes, [code for code, name in enumerate(vocab) if name in names])
