import hashlib
from pathlib import Path

import numpy as np
import pytest

from twl_citybake.config import BBox, DataDirs, load_city
from twl_citybake.projection import LocalProjection
from twl_citybake.sources import fetch


def test_ahmedabad_config_is_pinned() -> None:
    city = load_city("ahmedabad")
    assert city.bbox.contains(72.5714, 23.0225)
    assert city.osm.filename.endswith("-260101.osm.pbf")
    assert len(city.osm.md5) == 32


def test_unknown_city_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown city"):
        load_city("atlantis")


def test_bbox_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        BBox(72.7, 22.9, 72.4, 23.1)


def test_data_dirs_default_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TWL_DATA_DIR", raising=False)
    assert DataDirs.from_env().root == Path("data")
    monkeypatch.setenv("TWL_DATA_DIR", str(tmp_path))
    assert (
        DataDirs.from_env().stage("ahmedabad", "osm") == tmp_path / "cache/citybake/ahmedabad/osm"
    )


def test_projection_origin_is_bbox_centre_and_round_trips() -> None:
    bbox = load_city("ahmedabad").bbox
    proj = LocalProjection.for_bbox(bbox)
    x, y = proj.forward([bbox.centre[0], bbox.east], [bbox.centre[1], bbox.centre[1]])
    assert abs(x[0]) < 1e-6 and abs(y[0]) < 1e-6
    assert 13_500 < x[1] < 14_000
    lon, lat = proj.inverse(x, y)
    np.testing.assert_allclose(lon, [bbox.centre[0], bbox.east], atol=1e-9)
    np.testing.assert_allclose(lat, [bbox.centre[1], bbox.centre[1]], atol=1e-9)


def test_fetch_verifies_checksum(tmp_path: Path) -> None:
    payload = tmp_path / "remote.bin"
    payload.write_bytes(b"twinlight")
    md5 = hashlib.md5(b"twinlight", usedforsecurity=False).hexdigest()
    dest = tmp_path / "raw" / "remote.bin"

    assert fetch(payload.as_uri(), md5, dest).read_bytes() == b"twinlight"
    assert fetch("file:///does/not/exist", md5, dest) == dest
    with pytest.raises(ValueError, match="pinned"):
        fetch(payload.as_uri(), "0" * 32, tmp_path / "other.bin")
    assert not (tmp_path / "other.bin").exists()
