from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt

from twl_citybake.config import PersonasConfig, load_city
from twl_citybake.personas.pool import PersonaPool, fetch_pool, read_pool
from twl_citybake.store import write_table

MARRIED = "Currently Married"
NO_OCCUPATION = "No Occupation / Retired / Homemaker"
GRADUATE = "Graduate & above"
DIPLOMA = "Technical diploma or certificate not equal to degree"
DECOY = "DECOY-PERSONA-TEXT-MUST-NEVER-LEAK"
STATE = "Gujarat"
DISTRICT = "Ahmadabad"
NONE = 0xFFFFFFFF

EDUCATION = (
    "Illiterate",
    "Primary",
    "Middle",
    "Secondary",
    "Higher Secondary",
    "Graduate & above",
    DIPLOMA,
)
EDUCATION_MIX = (0.1, 0.14, 0.16, 0.2, 0.1, 0.22, 0.08)
OCCUPATIONS = ("Shopkeeper", "Driver", "Teacher", "Clerk", "Tailor", "Factory worker", "Engineer")
LANGUAGES = ("Gujarati", "Hindi", "Urdu", "English")


@dataclass(frozen=True)
class Row:
    uuid: str
    sex: str = "Female"
    age: int = 30
    marital_status: str = "Never Married"
    education_level: str = "Secondary"
    occupation: str = "Driver"
    first_language: str = "Gujarati"
    state: str = STATE
    district: str = DISTRICT


def uuid_for(i: int) -> str:
    return f"{i:08x}-0000-4000-8000-{i:012x}"


def personas_config() -> PersonasConfig:
    config = load_city("ahmedabad").personas
    assert config is not None
    return config


def write_source(path: Path, rows: Sequence[Row]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "uuid": [r.uuid for r in rows],
        "sex": [r.sex for r in rows],
        "age": [r.age for r in rows],
        "marital_status": [r.marital_status for r in rows],
        "education_level": [r.education_level for r in rows],
        "occupation": [r.occupation for r in rows],
        "first_language": [r.first_language for r in rows],
        "state": [r.state for r in rows],
        "district": [r.district for r in rows],
        "persona": [f"{DECOY} {r.uuid}" for r in rows],
        "professional_persona": [f"{DECOY} work" for r in rows],
    }
    selects = ", ".join(f"unnest(${name}) AS {name}" for name in columns)
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT {selects}) TO '{path}' (FORMAT PARQUET)", columns)
    return path


def build_pool(root: Path, rows: Sequence[Row], name: str = "pool") -> PersonaPool:
    source = write_source(root / f"{name}-source.parquet", rows)
    cached = fetch_pool(personas_config(), root / f"{name}.parquet", urls=[str(source)])
    return read_pool(cached)


def married_code(pool: PersonaPool) -> int:
    return pool.vocab["marital"].index(MARRIED)


def realistic_rows(n: int = 6000, seed: int = 5) -> list[Row]:
    rng = np.random.default_rng(seed)
    ages = np.arange(18, 86)
    age = rng.choice(ages, n, p=_normalised(np.exp(-0.5 * ((ages - 40) / 17.0) ** 2) + 0.05))
    female = rng.random(n) < 0.48
    rows: list[Row] = []
    for i in range(n):
        a, f = int(age[i]), bool(female[i])
        p_married = _p_married(a, f)
        draw = rng.random()
        if draw < p_married:
            marital = MARRIED
        elif a < 38 and rng.random() < 0.85:
            marital = "Never Married"
        elif a >= 55 and rng.random() < (0.8 if f else 0.4):
            marital = "Widowed"
        elif rng.random() < 0.06:
            marital = "Divorced"
        else:
            marital = "Never Married" if a < 45 else "Widowed"
        working = rng.random() < (0.25 if f else 0.85) * (0.15 if a >= 62 else 1.0)
        education = EDUCATION[int(rng.choice(len(EDUCATION), p=EDUCATION_MIX))]
        rows.append(
            Row(
                uuid=uuid_for(i * 7919 % 1_000_003),
                sex="Female" if f else "Male",
                age=a,
                marital_status=marital,
                education_level=education,
                occupation=OCCUPATIONS[int(rng.integers(0, len(OCCUPATIONS)))]
                if working
                else NO_OCCUPATION,
                first_language=LANGUAGES[int(rng.choice(4, p=[0.8, 0.12, 0.05, 0.03]))],
            )
        )
    return rows


def decoy_rows(count: int = 200, seed: int = 9) -> list[Row]:
    rng = np.random.default_rng(seed)
    return [
        Row(
            uuid=uuid_for(2_000_000 + i),
            sex="Male" if rng.random() < 0.5 else "Female",
            age=int(rng.integers(18, 70)),
            marital_status=MARRIED,
            state="Maharashtra" if i % 2 else STATE,
            district="Pune" if i % 2 else "Surat",
        )
        for i in range(count)
    ]


def _p_married(age: int, female: bool) -> float:
    start = 20 if female else 24
    ramp = np.clip((age - start) / 10.0, 0.0, 1.0)
    decline = np.clip((age - 70) / 30.0, 0.0, 0.5)
    return float(0.88 * ramp - decline)


def _normalised(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return w / w.sum()


def married_pool_rows(
    ages: range, per_age: int = 40, married_share: float = 1.0, seed: int = 3
) -> list[Row]:
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    for age in ages:
        for sex in ("Female", "Male"):
            for _ in range(per_age):
                married = rng.random() < married_share
                rows.append(
                    Row(
                        uuid=uuid_for(len(rows)),
                        sex=sex,
                        age=age,
                        marital_status=MARRIED if married else "Never Married",
                    )
                )
    return rows


def household_runs(adults: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    sizes = np.asarray(adults, np.int64)
    return np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes)


def write_residents(population: Path, adults: npt.ArrayLike) -> int:
    household = household_runs(adults)
    write_table(
        population / "residents",
        {"household": household, "weight": np.ones(len(household), np.float32)},
    )
    return len(household)


def household_adults(households: int, seed: int = 21) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(1, 7), households, p=[0.12, 0.3, 0.26, 0.2, 0.07, 0.05])
