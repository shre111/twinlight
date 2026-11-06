import dataclasses
import json
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import pytest

from twl_tape.fixtures import (
    FIXTURE_DIR,
    HAND_BUILT,
    build_fixtures,
    check_fixtures,
    main,
    probes,
)
from twl_tape.format import NEVER, Tape
from twl_tape.reader import read_tape

ENGINE_FIXTURE = "engine_supply_small_seed1"
FIXTURE_NAMES = [*HAND_BUILT, ENGINE_FIXTURE]
SECTIONS = ["residents", "chatter", "deliveries", "stores", "aggregates"]
FOOTER = struct.Struct("<Q32s")
F32_MAX, F32_SUBNORMAL = 3.4028234663852886e38, 1.401298464324817e-45


def tape_bytes(name: str, directory: Path = FIXTURE_DIR) -> bytes:
    return (directory / f"{name}.twlt").read_bytes()


def columns(tape: Tape) -> dict[str, np.ndarray[Any, Any]]:
    return {
        f"{section}.{field.name.rstrip('_')}": getattr(getattr(tape, section), field.name)
        for section in SECTIONS
        for field in dataclasses.fields(getattr(tape, section))
    }


def reject_constant(token: str) -> None:
    raise AssertionError(f"{token} is not strict JSON")


@pytest.fixture
def copied(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE_DIR, tmp_path / "v1"))


def test_the_checked_in_fixtures_pass_the_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_fixtures() == []
    main(["--check"])
    assert capsys.readouterr().out.count("\n") == 1
    assert sorted(path.stem for path in FIXTURE_DIR.glob("*.twlt")) == sorted(FIXTURE_NAMES)


def test_two_builds_in_one_process_are_byte_identical() -> None:
    assert build_fixtures() == build_fixtures()


def flip_a_tape_byte(directory: Path) -> None:
    path = directory / "small.twlt"
    data = bytearray(path.read_bytes())
    data[400] ^= 1
    path.write_bytes(bytes(data))


def make_probes_stale(directory: Path) -> None:
    path = directory / f"{ENGINE_FIXTURE}.probes.json"
    document = json.loads(path.read_text())
    document["columns"]["residents"]["aware_at"]["head"][0] += 1
    path.write_text(json.dumps(document, indent=2) + "\n")


def add_a_tape_without_probes(directory: Path) -> None:
    shutil.copy(directory / "limits.twlt", directory / "extra.twlt")


def add_probes_without_a_tape(directory: Path) -> None:
    shutil.copy(directory / "small.probes.json", directory / "ghost.probes.json")


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (flip_a_tape_byte, "small.twlt differs from a fresh build"),
        (make_probes_stale, f"{ENGINE_FIXTURE}.probes.json is stale"),
        (add_a_tape_without_probes, "extra.twlt has no extra.probes.json"),
        (add_probes_without_a_tape, "ghost.probes.json has no ghost.twlt"),
    ],
)
def test_the_check_fails_on_a_damaged_directory(damage: Any, problem: str, copied: Path) -> None:
    assert check_fixtures(copied) == []
    damage(copied)
    assert any(problem in found for found in check_fixtures(copied))
    with pytest.raises(SystemExit) as exit_info:
        main(["--check"], copied)
    assert problem in str(exit_info.value.code)


def test_writing_restores_a_damaged_directory(copied: Path) -> None:
    flip_a_tape_byte(copied)
    make_probes_stale(copied)
    main([], copied)
    assert check_fixtures(copied) == []
    for name in FIXTURE_NAMES:
        assert tape_bytes(name, copied) == tape_bytes(name)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_probes_agree_with_the_reader_and_the_file(name: str) -> None:
    data = tape_bytes(name)
    tape, document = read_tape(data), probes(data)
    total_length, stored_hash = FOOTER.unpack(data[-FOOTER.size :])

    assert document["file"] == {"bytes": len(data), "blake3": blake3.blake3(data).hexdigest()}
    assert document["footer"] == {"total_length": total_length, "blake3": stored_hash.hex()}
    assert stored_hash == blake3.blake3(data[:total_length]).digest()

    header = dataclasses.asdict(tape.header)
    expected_header = header | {
        "version": 1,
        "seed": str(tape.header.seed),
        "run_id": tape.header.run_id.hex(),
        "city_pack_hash": tape.header.city_pack_hash.hex(),
        "scenario_hash": tape.header.scenario_hash.hex(),
    }
    assert document["header"] == expected_header

    assert [(row["kind"], row["name"]) for row in document["sections"]] == list(
        enumerate([*SECTIONS, "chunks"], start=1)
    )
    for row in document["sections"]:
        compressed = data[row["offset"] : row["offset"] + row["length"]]
        assert row["crc32"] == zlib.crc32(compressed), row["name"]

    probed = {
        f"{section}.{column}": probe
        for section in SECTIONS
        for column, probe in document["columns"][section].items()
    }
    expected = columns(tape)
    assert list(probed) == list(expected)
    for label, values in expected.items():
        probe, rows = probed[label], values.tolist()
        assert probe["length"] == len(rows), label
        assert probe["head"] == rows[:8] and probe["tail"] == rows[-8:], label
        assert probe["crc32"] == zlib.crc32(values.tobytes()), label
        if probe["dtype"] == "f32":
            assert "sum" not in probe, label
        else:
            assert probe["sum"] == str(sum(rows)), label


def test_probe_documents_have_the_agreed_shape_and_key_order() -> None:
    document = probes(tape_bytes("small"))
    assert list(document) == ["file", "footer", "header", "sections", "columns"]
    assert list(document["header"]) == [
        "version",
        "flags",
        "run_id",
        "seed",
        "engine_version",
        "city_pack_hash",
        "scenario_hash",
        "sim_minutes",
        "tick_minutes",
        "residents",
        "stores",
        "chatter_rows",
        "delivery_rows",
        "stockout_pairs",
    ]
    assert list(document["sections"][0]) == [
        "kind",
        "name",
        "offset",
        "length",
        "raw_length",
        "crc32",
    ]
    assert list(document["columns"]) == SECTIONS
    assert list(document["columns"]["chatter"]) == ["from", "to", "at", "kind"]
    stores = document["columns"]["stores"]
    assert list(stores["orders"]) == ["dtype", "length", "crc32", "sum", "head", "tail"]
    assert list(stores["stock"]) == ["dtype", "length", "crc32", "head", "tail"]
    assert [probe["dtype"] for probe in stores.values()] == ["f32", "u16", "u32", "u32"]
    assert document["columns"]["residents"]["first_source"]["dtype"] == "u8"
    assert stores["stock"]["head"][4] == float(np.float32(1.5e-3))


def test_small_covers_every_code_and_keeps_the_documented_orderings() -> None:
    tape = read_tape(tape_bytes("small"))
    residents, deliveries, stores = tape.residents, tape.deliveries, tape.stores
    for with_never in (
        residents.aware_at,
        residents.first_order_at,
        residents.heard_from,
        deliveries.picked_at,
        deliveries.delivered_at,
    ):
        assert NEVER in with_never
    assert sorted(set(residents.first_source.tolist())) == [0, 1, 2, 3]
    assert np.bitwise_or.reduce(residents.outcome_flags) == 7
    assert np.bitwise_or.reduce(deliveries.outcome) == 3
    assert sorted(set(tape.chatter.kind.tolist())) == [1, 2]
    assert np.all(np.diff(tape.chatter.at.astype(np.int64)) >= 0)
    assert np.all(np.diff(deliveries.ordered_at.astype(np.int64)) >= 0)
    assert stores.stockout_offsets.tolist() == [0, 2, 3]
    pairs = stores.stockout_intervals.reshape(-1, 2).tolist()
    assert pairs == [[10, 20], [35, tape.header.sim_minutes], [12, 28]]


def test_limits_contains_the_extremes() -> None:
    tape = read_tape(tape_bytes("limits"))
    header = tape.header
    assert header.seed == 2**64 - 1
    assert len(header.engine_version) == 32
    assert header.ticks % 8 != 0
    for identifier in (header.run_id, header.city_pack_hash, header.scenario_hash):
        assert 0xFF in identifier and 0x00 in identifier
        assert sum(byte != 0 for byte in identifier) >= len(identifier) - 1
    for label, values in columns(tape).items():
        present = set(values.tolist())
        if values.dtype == np.uint16:
            assert 0xFFFF in present, label
        elif values.dtype == np.uint32 and label != "stores.stockout_offsets":
            assert {0xFFFF_FFFE, NEVER} <= present, label
        elif values.dtype == np.float32:
            assert {F32_MAX, F32_SUBNORMAL} <= present, label
            assert np.any((values == 0) & np.signbit(values)), label
            assert np.all(np.isfinite(values)), label
    assert np.all(np.diff(tape.chatter.at.astype(np.int64)) >= 0)
    assert np.all(np.diff(tape.deliveries.ordered_at.astype(np.int64)) >= 0)
    assert probes(tape_bytes("limits"))["header"]["seed"] == "18446744073709551615"


def test_empty_has_zero_length_columns_and_a_single_stockout_offset() -> None:
    data = tape_bytes("empty")
    tape, document = read_tape(data), probes(data)
    assert tape.stores.stockout_offsets.tolist() == [0]
    assert tape.header.ticks == 1
    for label, values in columns(tape).items():
        section = label.split(".")[0]
        expected_length = 1 if section == "aggregates" or label.endswith("offsets") else 0
        assert values.shape == (expected_length,), label
    assert document["columns"]["residents"]["aware_at"] == {
        "dtype": "u32",
        "length": 0,
        "crc32": 0,
        "sum": "0",
        "head": [],
        "tail": [],
    }
    assert document["columns"]["stores"]["stock"] == {
        "dtype": "f32",
        "length": 0,
        "crc32": 0,
        "head": [],
        "tail": [],
    }


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_probe_files_are_strict_json_of_the_probes(name: str) -> None:
    text = (FIXTURE_DIR / f"{name}.probes.json").read_text()
    assert json.loads(text, parse_constant=reject_constant) == probes(tape_bytes(name))
    assert text.endswith("}\n") and "\r" not in text
