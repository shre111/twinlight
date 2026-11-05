import urllib.request
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import numpy.typing as npt

from twl_citybake.config import BBox
from twl_citybake.parquet import kv_clause, sql_literal, stored_key

SOURCE_CODES = {"google": 1, "microsoft": 2}


class VidaRows(NamedTuple):
    source: npt.NDArray[np.uint8]
    confidence: npt.NDArray[np.float32]
    wkb: list[bytes]


def vida_cache_key(etag: str, bbox: BBox) -> dict[str, str]:
    return {"etag": etag, "bbox": repr((bbox.west, bbox.south, bbox.east, bbox.north))}


def fetch_vida(url: str, etag: str, bbox: BBox, dest: Path) -> Path:
    """Google and Microsoft footprints whose bbox meets the city bbox, cached once as Parquet
    (source, confidence, area_m2, wkb) in a total row order so every read sees the same rows. The
    pin and bbox sit in the file's metadata: a matching cache is kept, any other cache raises."""
    key = vida_cache_key(etag, bbox)
    if dest.exists():
        if (cached := stored_key(dest, key)) != key:
            raise ValueError(f"{dest} holds {cached}, not {key}; delete it to fetch again")
        return dest
    if (actual := _etag(url)) != etag:
        raise ValueError(f"{url} has ETag {actual}, pinned {etag}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    w, s, e, n = (repr(float(v)) for v in (bbox.west, bbox.south, bbox.east, bbox.north))
    sources = ", ".join(f"'{name}'" for name in SOURCE_CODES)
    query = f"""
        COPY (
            SELECT bf_source AS source, confidence, area_in_meters AS area_m2, geometry AS wkb
            FROM read_parquet({sql_literal(url)})
            WHERE bf_source IN ({sources})
                AND bbox.xmin <= {e} AND bbox.xmax >= {w} AND bbox.ymin <= {n} AND bbox.ymax >= {s}
            ORDER BY source, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, area_m2, confidence, wkb
        ) TO {sql_literal(str(partial))} (FORMAT parquet, COMPRESSION zstd, {kv_clause(key)})
    """
    try:
        with duckdb.connect() as con:
            con.execute("INSTALL httpfs; LOAD httpfs; SET enable_geoparquet_conversion = false")
            con.execute(query)
        partial.rename(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def read_vida(path: Path) -> VidaRows:
    """The cached rows in file order; Microsoft has no confidence, so it reads as NaN."""
    with duckdb.connect() as con:
        con.execute("SET enable_geoparquet_conversion = false")
        rows = con.execute(
            "SELECT source, coalesce(confidence::FLOAT, 'NaN'::FLOAT) AS confidence, wkb "
            "FROM read_parquet(?, file_row_number = true) ORDER BY file_row_number",
            [str(path)],
        ).fetchnumpy()
    names = np.asarray(rows["source"], dtype=object)
    source = np.zeros(len(names), np.uint8)
    for name, code in SOURCE_CODES.items():
        source[names == name] = code
    if (source == 0).any():
        raise ValueError(f"{path} has sources other than {sorted(SOURCE_CODES)}")
    return VidaRows(
        source=source,
        confidence=np.asarray(rows["confidence"], np.float32),
        wkb=[bytes(b) for b in rows["wkb"]],
    )


def _etag(url: str) -> str | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.headers.get("ETag")
