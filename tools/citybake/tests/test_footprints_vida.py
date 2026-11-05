from pathlib import Path

import numpy as np
import pytest
import shapely
from footprints_fixture import (
    GOOGLE,
    MICROSOFT,
    CacheRow,
    no_network,  # noqa: F401
    square,
    tiny_cache_rows,
    to_lonlat_wkb,
    write_vida_cache,
)
from osm_fixture import BBOX

from twl_citybake.config import BBox
from twl_citybake.footprints.vida import fetch_vida, read_vida, vida_cache_key
from twl_citybake.projection import LocalProjection

PROJ = LocalProjection.for_bbox(BBOX)
SOURCE_CODES = {GOOGLE: 1, MICROSOFT: 2}


def test_read_vida_returns_sources_confidence_and_wkb_in_cache_order(tmp_path: Path) -> None:
    rows = tiny_cache_rows()
    source, confidence, wkb = read_vida(write_vida_cache(tmp_path / "vida.parquet", rows, PROJ))

    assert source.dtype == np.uint8
    assert source.tolist() == [SOURCE_CODES[r.source] for r in rows]
    assert confidence.dtype == np.float32
    assert len(wkb) == len(rows)
    for got, row in zip(wkb, rows, strict=True):
        assert bytes(got) == to_lonlat_wkb(row.geom, PROJ)


def test_microsoft_confidence_is_nan_and_google_confidence_is_kept(tmp_path: Path) -> None:
    rows = tiny_cache_rows()
    _, confidence, _ = read_vida(write_vida_cache(tmp_path / "vida.parquet", rows, PROJ))

    expected = np.asarray([np.nan if r.confidence is None else r.confidence for r in rows])
    np.testing.assert_array_equal(np.isnan(confidence), np.isnan(expected))
    np.testing.assert_allclose(confidence[~np.isnan(expected)], expected[~np.isnan(expected)])
    assert all(np.isnan(c) for c, r in zip(confidence, rows, strict=True) if r.source == MICROSOFT)


def test_wkb_decodes_to_the_longitude_latitude_footprints_that_were_cached(tmp_path: Path) -> None:
    rows = [CacheRow(GOOGLE, 0.9, square(0.0, 0.0))]
    _, _, wkb = read_vida(write_vida_cache(tmp_path / "vida.parquet", rows, PROJ))

    geom = shapely.from_wkb(bytes(wkb[0]))
    lon, lat = geom.centroid.x, geom.centroid.y
    assert BBOX.contains(lon, lat)
    assert lon == pytest.approx(BBOX.centre[0], abs=1e-6)
    assert lat == pytest.approx(BBOX.centre[1], abs=1e-6)


def test_an_empty_cache_reads_as_empty_arrays(tmp_path: Path) -> None:
    source, confidence, wkb = read_vida(write_vida_cache(tmp_path / "empty.parquet", [], PROJ))
    assert len(source) == len(confidence) == len(wkb) == 0
    assert source.dtype == np.uint8
    assert confidence.dtype == np.float32


def test_reading_twice_gives_identical_results(tmp_path: Path) -> None:
    path = write_vida_cache(tmp_path / "vida.parquet", tiny_cache_rows(), PROJ)
    a, b = read_vida(path), read_vida(path)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert [bytes(x) for x in a[2]] == [bytes(x) for x in b[2]]


@pytest.mark.usefixtures("no_network")
def test_fetch_vida_with_an_existing_cache_returns_it_without_any_request(tmp_path: Path) -> None:
    key = vida_cache_key('"etag"', BBOX)
    dest = write_vida_cache(tmp_path / "vida.parquet", tiny_cache_rows(), PROJ, key)
    before = dest.read_bytes()

    result = fetch_vida("https://example.invalid/vida.parquet", '"etag"', BBOX, dest)

    assert result == dest
    assert dest.read_bytes() == before


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize(
    "key",
    [
        vida_cache_key('"other"', BBOX),
        vida_cache_key('"etag"', BBox(BBOX.west, BBOX.south, BBOX.east, BBOX.north + 0.01)),
        None,
    ],
)
def test_fetch_vida_refuses_a_cache_built_from_anything_else(
    tmp_path: Path, key: dict[str, str] | None
) -> None:
    dest = write_vida_cache(tmp_path / "vida.parquet", tiny_cache_rows(), PROJ, key)

    with pytest.raises(ValueError, match="delete it to fetch again"):
        fetch_vida("https://example.invalid/vida.parquet", '"etag"', BBOX, dest)


def test_the_cache_key_is_the_same_for_equal_integer_and_float_bboxes() -> None:
    assert vida_cache_key('"e"', BBox(72, 23, 73, 24)) == vida_cache_key(
        '"e"', BBox(72.0, 23.0, 73.0, 24.0)
    )
