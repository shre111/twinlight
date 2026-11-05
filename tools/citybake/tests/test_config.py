import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from twl_citybake.config import BBox, DataDirs, load_city
from twl_citybake.projection import LocalProjection
from twl_citybake.sources import extract_member, fetch


def test_ahmedabad_config_is_pinned() -> None:
    city = load_city("ahmedabad")
    assert city.bbox.contains(72.5714, 23.0225)
    assert city.osm.filename.endswith("-260101.osm.pbf")
    assert len(city.osm.md5) == 32


def test_unknown_city_is_rejected() -> None:
    for slug in ("atlantis", "../ahmedabad", "Ahmedabad"):
        with pytest.raises(ValueError, match="unknown city"):
            load_city(slug)


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
    assert not (tmp_path / "other.bin.part").exists()


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_member_unpacks_exactly_the_named_member(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"a.tif": b"AAA", "b.tif": b"BBB", "a.txt": b"T"})
    dest = tmp_path / "out" / "a.tif"
    assert extract_member(archive, "a.tif", dest) == dest
    assert dest.read_bytes() == b"AAA"
    assert [p.name for p in dest.parent.iterdir()] == ["a.tif"]


def test_extract_member_leaves_an_existing_dest_untouched(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"a.tif": b"AAA"})
    dest = tmp_path / "out" / "a.tif"
    dest.parent.mkdir()
    dest.write_bytes(b"keep")
    assert extract_member(archive, "a.tif", dest) == dest
    assert dest.read_bytes() == b"keep"


def test_extract_member_missing_member_raises_and_leaves_nothing(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"a.tif": b"AAA"})
    dest = tmp_path / "out" / "nope.tif"
    with pytest.raises(ValueError, match=r"nope\.tif"):
        extract_member(archive, "nope.tif", dest)
    assert not dest.exists()
    assert not dest.with_name("nope.tif.part").exists()


def test_extract_member_with_path_segments_lands_only_at_dest(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"../evil/x.tif": b"X", "deep/dir/y.tif": b"Y"})
    root = tmp_path / "root"
    extract_member(archive, "deep/dir/y.tif", root / "y.tif")
    extract_member(archive, "../evil/x.tif", root / "x.tif")
    assert (root / "y.tif").read_bytes() == b"Y"
    assert (root / "x.tif").read_bytes() == b"X"
    assert sorted(p.name for p in root.rglob("*")) == ["x.tif", "y.tif"]
    assert not (tmp_path / "evil").exists()
