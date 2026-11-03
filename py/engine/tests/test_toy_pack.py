from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city, write_pack

SMALL = 5_000

Column = tuple[str, Any, Callable[[dict[str, int]], int]]


def const(key: str) -> Callable[[dict[str, int]], int]:
    return lambda counts: counts[key]


def plus_one(key: str) -> Callable[[dict[str, int]], int]:
    return lambda counts: counts[key] + 1


F32, U32, U16, U8 = np.float32, np.uint32, np.uint16, np.uint8

LAYOUT: dict[str, tuple[str, list[Column]]] = {
    "buildings.bin": (
        "buildings",
        [
            ("x", F32, const("buildings")),
            ("y", F32, const("buildings")),
            ("width", F32, const("buildings")),
            ("depth", F32, const("buildings")),
            ("height", F32, const("buildings")),
            ("rotation", F32, const("buildings")),
            ("district", U16, const("buildings")),
        ],
    ),
    "residents.bin": (
        "residents",
        [
            ("x", F32, const("residents")),
            ("y", F32, const("residents")),
            ("z", F32, const("residents")),
            ("building", U32, const("residents")),
            ("hex", U32, const("residents")),
            ("segment", U16, const("residents")),
            ("persona_ref", U32, const("residents")),
            ("weight", F32, const("residents")),
        ],
    ),
    "graph.bin": (
        "ties",
        [
            ("offsets", U32, plus_one("residents")),
            ("targets", U32, const("ties")),
            ("kind", U8, const("ties")),
            ("strength", F32, const("ties")),
        ],
    ),
    "stores.bin": (
        "stores",
        [
            ("x", F32, const("stores")),
            ("y", F32, const("stores")),
            ("hex", U32, const("stores")),
            ("radius_m", F32, const("stores")),
            ("district", U16, const("stores")),
        ],
    ),
    "roads.bin": (
        "roads",
        [
            ("offsets", U32, plus_one("roads")),
            ("x", F32, const("roadPoints")),
            ("y", F32, const("roadPoints")),
            ("kind", U8, const("roads")),
        ],
    ),
    "river.bin": (
        "river",
        [
            ("offsets", U32, plus_one("riverPolygons")),
            ("x", F32, const("riverPoints")),
            ("y", F32, const("riverPoints")),
            ("kind", U8, const("riverPolygons")),
        ],
    ),
}


def pad8(n: int) -> int:
    return (n + 7) // 8 * 8


def read_columns(
    data: bytes, columns: list[Column], counts: dict[str, int]
) -> tuple[dict[str, np.ndarray], int]:
    out: dict[str, np.ndarray] = {}
    pos = 0
    for name, dtype, length in columns:
        n = length(counts)
        dt = np.dtype(dtype).newbyteorder("<")
        out[name] = np.frombuffer(data, dtype=dt, count=n, offset=pos)
        pos = pad8(pos + n * dt.itemsize)
    return out, pos


def blake3_hex(data: bytes) -> str:
    import blake3

    return blake3.blake3(data).hexdigest()


def pack_files(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def city() -> ToyCity:
    return build_toy_city(seed=49, residents=SMALL)


@pytest.fixture(scope="module")
def pack(city: ToyCity, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("pack_a")
    return out, write_pack(city, out)


def test_same_seed_writes_byte_identical_packs(
    pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_a, hash_a = pack
    out_b = tmp_path_factory.mktemp("pack_b")
    hash_b = write_pack(build_toy_city(seed=49, residents=SMALL), out_b)
    assert hash_a == hash_b
    files_a, files_b = pack_files(out_a), pack_files(out_b)
    assert files_a.keys() == files_b.keys()
    for name in files_a:
        assert files_a[name] == files_b[name], name


def test_different_seed_writes_a_different_pack_hash(
    pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    other = write_pack(
        build_toy_city(seed=7, residents=SMALL), tmp_path_factory.mktemp("pack_seed7")
    )
    assert other != pack[1]


def test_changing_one_column_changes_the_pack_hash(
    city: ToyCity, pack: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    taller = dataclasses.replace(city.buildings, height=city.buildings.height + np.float32(1))
    changed = dataclasses.replace(city, buildings=taller)
    assert write_pack(changed, tmp_path_factory.mktemp("pack_taller")) != pack[1]


def test_pack_directory_holds_the_spec_files(pack: tuple[Path, str]) -> None:
    names = {p.name for p in pack[0].iterdir()}
    expected = {*LAYOUT, "districts.json", "manifest.json"}
    assert expected <= names


def test_manifest_has_the_spec_keys_and_values(pack: tuple[Path, str]) -> None:
    manifest = json.loads((pack[0] / "manifest.json").read_bytes())
    assert manifest["format"] == "twinlight-pack"
    assert manifest["version"] == 1
    assert manifest["kind"] == "toy"
    assert manifest["seed"] == 49
    assert isinstance(manifest["crs"], str)
    assert isinstance(manifest["attributions"], list)
    for key in ("name", "bbox", "counts", "files"):
        assert key in manifest
    assert len(manifest["bbox"]) == 4
    min_x, min_y, max_x, max_y = manifest["bbox"]
    assert min_x < max_x
    assert min_y < max_y


def test_manifest_bbox_contains_every_building_and_resident(
    pack: tuple[Path, str], city: ToyCity
) -> None:
    min_x, min_y, max_x, max_y = json.loads((pack[0] / "manifest.json").read_bytes())["bbox"]
    for xs, ys in ((city.buildings.x, city.buildings.y), (city.residents.x, city.residents.y)):
        assert float(xs.min()) >= min_x - 1
        assert float(xs.max()) <= max_x + 1
        assert float(ys.min()) >= min_y - 1
        assert float(ys.max()) <= max_y + 1


def test_manifest_counts_match_the_city(pack: tuple[Path, str], city: ToyCity) -> None:
    counts = json.loads((pack[0] / "manifest.json").read_bytes())["counts"]
    assert counts["buildings"] == len(city.buildings.x)
    assert counts["residents"] == SMALL
    assert counts["ties"] == len(city.ties.targets)
    assert counts["stores"] == len(city.stores.x) == 14
    assert counts["roads"] == len(city.roads.kind)
    assert counts["roadPoints"] == len(city.roads.x)
    assert counts["riverPolygons"] == len(city.river.offsets) - 1
    assert counts["riverPoints"] == len(city.river.x)


def test_manifest_bytes_are_canonical_sorted_compact_json(pack: tuple[Path, str]) -> None:
    raw = (pack[0] / "manifest.json").read_bytes()
    obj = json.loads(raw)
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    assert raw == canonical


def test_manifest_files_hashes_match_each_file(pack: tuple[Path, str]) -> None:
    root = pack[0]
    files = json.loads((root / "manifest.json").read_bytes())["files"]
    assert {*LAYOUT, "districts.json"} <= set(files)
    assert "manifest.json" not in files
    for name, digest in files.items():
        assert digest == blake3_hex((root / name).read_bytes()), name


def test_returned_pack_hash_is_blake3_of_manifest_bytes(pack: tuple[Path, str]) -> None:
    out, pack_hash = pack
    assert pack_hash == blake3_hex((out / "manifest.json").read_bytes())
    assert len(pack_hash) == 64


@pytest.mark.parametrize("filename", sorted(LAYOUT))
def test_bin_file_columns_equal_the_city_arrays_and_size_is_padded_total(
    filename: str, pack: tuple[Path, str], city: ToyCity
) -> None:
    root = pack[0]
    counts = json.loads((root / "manifest.json").read_bytes())["counts"]
    group, columns = LAYOUT[filename]
    data = (root / filename).read_bytes()
    parsed, total = read_columns(data, columns, counts)
    assert len(data) == total
    assert len(data) % 8 == 0
    source = getattr(city, group)
    for name, dtype, _ in columns:
        expected = np.asarray(getattr(source, name))
        assert expected.dtype == np.dtype(dtype), name
        np.testing.assert_array_equal(parsed[name], expected, err_msg=f"{filename}:{name}")


def test_bin_columns_start_at_eight_byte_aligned_zero_padded_offsets(
    pack: tuple[Path, str],
) -> None:
    root = pack[0]
    counts = json.loads((root / "manifest.json").read_bytes())["counts"]
    data = (root / "buildings.bin").read_bytes()
    pos = 0
    for _, dtype, length in LAYOUT["buildings.bin"][1]:
        end = pos + length(counts) * np.dtype(dtype).itemsize
        nxt = pad8(end)
        assert data[end:nxt] == bytes(nxt - end)
        pos = nxt
    assert pos == len(data)


def test_districts_json_matches_the_city_districts(pack: tuple[Path, str], city: ToyCity) -> None:
    got = json.loads((pack[0] / "districts.json").read_bytes())
    assert [d["id"] for d in got] == list(range(12))
    for entry, district in zip(got, city.districts, strict=True):
        assert set(entry) >= {"id", "name", "x", "y", "east"}
        assert entry["id"] == district.id
        assert entry["name"] == district.name
        assert entry["x"] == pytest.approx(district.x)
        assert entry["y"] == pytest.approx(district.y)
        assert entry["east"] is district.east


def test_cli_prints_the_same_hash_as_write_pack(
    city: ToyCity, tmp_path_factory: pytest.TempPathFactory
) -> None:
    expected = write_pack(city, tmp_path_factory.mktemp("pack_api"))
    out = tmp_path_factory.mktemp("pack_cli")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "twl_engine.fixtures",
            "--seed",
            "49",
            "--residents",
            str(SMALL),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1].strip() == expected
    assert (out / "manifest.json").is_file()
