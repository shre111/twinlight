import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

META = "_table.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_table(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """One `.npy` per column plus `_table.json`; every column has the same first axis."""
    lengths = {_first_axis(name, array) for name, array in columns.items()}
    if len(lengths) > 1:
        raise ValueError(f"columns of {path} have different lengths {sorted(lengths)}")
    _write(path, columns, lengths.pop() if lengths else 0)


def write_ragged(path: Path, columns: Mapping[str, np.ndarray], rows: int) -> None:
    """Like write_table for offset-encoded records (CSR offsets plus flat values): `rows`
    records whose columns have different lengths."""
    for name, array in columns.items():
        _first_axis(name, array)
    _write(path, columns, rows)


def table_digest(path: Path) -> str:
    """sha256 over a table's files in name order, to tell whether a derived layer is stale."""
    digest = hashlib.sha256()
    for file in sorted(path.iterdir()):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def read_table(path: Path) -> dict[str, np.ndarray]:
    meta = json.loads((path / META).read_text(encoding="utf-8"))
    return {
        col["name"]: np.load(path / f"{col['name']}.npy", allow_pickle=False)
        for col in meta["columns"]
    }


def _first_axis(name: str, array: np.ndarray) -> int:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid column name {name!r}")
    if array.ndim == 0:
        raise ValueError(f"column {name!r} is a scalar")
    if array.dtype.hasobject:
        raise ValueError(f"column {name!r} has object dtype")
    return array.shape[0]


def _write(path: Path, columns: Mapping[str, np.ndarray], rows: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for stale in sorted(path.glob("*.npy")):
        if stale.stem not in columns:
            stale.unlink()
    for name, array in columns.items():
        np.save(path / f"{name}.npy", np.ascontiguousarray(array), allow_pickle=False)
    meta = {
        "rows": int(rows),
        "columns": [{"name": name, "dtype": str(array.dtype)} for name, array in columns.items()],
    }
    write_json(path / META, meta)
