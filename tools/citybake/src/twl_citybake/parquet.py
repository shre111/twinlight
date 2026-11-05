from collections.abc import Iterable, Mapping
from pathlib import Path

import duckdb


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def kv_clause(key: Mapping[str, str]) -> str:
    """The COPY option that stores a cache's key in the Parquet file itself."""
    pairs = ", ".join(f"{name}: {sql_literal(value)}" for name, value in key.items())
    return f"KV_METADATA {{{pairs}}}"


def stored_key(path: Path, names: Iterable[str]) -> dict[str, str]:
    """The named entries of a cached file's metadata; none on an unkeyed file."""
    with duckdb.connect() as con:
        rows = con.execute("SELECT key, value FROM parquet_kv_metadata(?)", [str(path)]).fetchall()
    metadata = {bytes(name).decode(): bytes(value).decode() for name, value in rows}
    return {name: metadata[name] for name in names if name in metadata}
