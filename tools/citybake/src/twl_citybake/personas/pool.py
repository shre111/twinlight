from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import PersonasConfig
from twl_citybake.parquet import kv_clause, sql_literal, stored_key

DATASET = "nvidia/Nemotron-Personas-India"
PARTS = 11
SEX_VOCAB = ["Female", "Male"]
FEMALE, MALE = 0, 1

# Only these reach disk: the dataset's persona text columns must never be read (D-012, D-026).
_COLUMNS = (
    "uuid",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "first_language",
    "state",
    "district",
)


@dataclass(frozen=True)
class PersonaPool:
    """One slice of the dataset, by row in uuid order; `vocab` holds each coded column's sorted
    categories, so a code is an index into it."""

    uuid: NDArray[np.bytes_]
    sex: NDArray[np.uint8]
    age: NDArray[np.uint8]
    marital: NDArray[np.uint8]
    education: NDArray[np.uint8]
    occupation: NDArray[np.uint16]
    language: NDArray[np.uint8]
    vocab: dict[str, list[str]]


def parquet_names() -> list[str]:
    return [f"en_IN-{part:05d}-of-{PARTS:05d}.parquet" for part in range(PARTS)]


def parquet_urls(revision: str) -> list[str]:
    root = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/data"
    return [f"{root}/{name}" for name in parquet_names()]


def fetch_pool(config: PersonasConfig, dest: Path, urls: Sequence[str] | None = None) -> Path:
    """The configured state and district as Parquet in uuid order. The slice's key sits in the
    file's key-value metadata: a matching cache is kept, any other cache raises."""
    key = {"revision": config.revision, "state": config.state, "district": config.district}
    if dest.exists():
        if (cached := stored_key(dest, key)) != key:
            raise ValueError(f"{dest} holds {cached}, not {key}; delete it to fetch again")
        return dest
    sources = [str(url) for url in urls] if urls is not None else parquet_urls(config.revision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with duckdb.connect() as con:
            if any(source.startswith(("http://", "https://")) for source in sources):
                con.execute("INSTALL httpfs; LOAD httpfs")
            con.execute(
                f"CREATE TEMP TABLE slice AS SELECT {', '.join(_COLUMNS)} FROM read_parquet(?) "
                "WHERE state = ? AND district = ?",
                [sources, config.state, config.district],
            )
            found = con.execute("SELECT count(*) FROM slice").fetchone()
            if found is None or found[0] == 0:
                raise ValueError(f"no personas for {config.district}, {config.state}")
            con.execute(
                f"COPY (SELECT * FROM slice ORDER BY uuid) TO {sql_literal(str(partial))} "
                f"(FORMAT parquet, COMPRESSION zstd, {kv_clause(key)})"
            )
        partial.rename(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def read_pool(path: Path) -> PersonaPool:
    with duckdb.connect() as con:
        rows = con.execute(
            "SELECT uuid::VARCHAR AS uuid, sex, age, marital_status, education_level, occupation, "
            "first_language FROM read_parquet(?) ORDER BY uuid",
            [str(path)],
        ).fetchnumpy()
    if nulls := [name for name, values in rows.items() if np.ma.is_masked(values)]:
        raise ValueError(f"{path} has nulls in {nulls[0]}")
    uuid = np.asarray(rows["uuid"].tolist(), "S36")
    if len(np.unique(uuid)) != len(uuid):
        raise ValueError(f"{path} repeats a uuid, so its row order is not total")
    sex_names = rows["sex"].astype(str)
    if unknown := sorted(set(sex_names.tolist()) - set(SEX_VOCAB)):
        raise ValueError(f"{path} has sex {unknown[0]!r}, expected {SEX_VOCAB}")
    age = np.asarray(rows["age"])
    if ((age < 0) | (age > 255)).any():
        raise ValueError(f"{path} has an age outside 0 to 255")
    marital, marital_vocab = _coded(rows["marital_status"], np.uint8)
    education, education_vocab = _coded(rows["education_level"], np.uint8)
    occupation, occupation_vocab = _coded(rows["occupation"], np.uint16)
    language, language_vocab = _coded(rows["first_language"], np.uint8)
    return PersonaPool(
        uuid=uuid,
        sex=(sex_names == SEX_VOCAB[MALE]).astype(np.uint8),
        age=age.astype(np.uint8),
        marital=marital,
        education=education,
        occupation=occupation,
        language=language,
        vocab={
            "sex": list(SEX_VOCAB),
            "marital": marital_vocab,
            "education": education_vocab,
            "occupation": occupation_vocab,
            "language": language_vocab,
        },
    )


def _coded[T: np.unsignedinteger](
    values: NDArray[np.object_], dtype: type[T]
) -> tuple[NDArray[T], list[str]]:
    categories, codes = np.unique(values.astype(str), return_inverse=True)
    if len(categories) > int(np.iinfo(dtype).max) + 1:
        raise ValueError(f"{len(categories)} categories do not fit {dtype.__name__}")
    return codes.astype(dtype), categories.tolist()
