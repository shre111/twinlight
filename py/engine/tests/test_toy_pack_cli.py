import json
from pathlib import Path

import pytest
from blake3 import blake3

from twl_engine.fixtures.__main__ import main
from twl_tape import read_tape, tape_hash


def test_cli_prints_the_hash_of_the_manifest_it_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--seed", "7", "--residents", "500", "--stores", "3", "--out", str(tmp_path)])
    printed = capsys.readouterr().out.strip()
    manifest = (tmp_path / "manifest.json").read_bytes()
    assert printed == blake3(manifest).hexdigest()
    parsed = json.loads(manifest)
    assert parsed["seed"] == 7
    assert parsed["counts"]["residents"] == 500
    assert parsed["counts"]["stores"] == 3


def test_cli_with_tape_writes_the_demo_run_of_the_pack_it_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tape_path = tmp_path / "tapes" / "toy.twlt"
    city = ["--residents", "2000", "--stores", "4", "--out", str(tmp_path / "pack")]
    main([*city, "--tape", str(tape_path), "--tape-seed", "3"])
    printed_pack, printed_tape = capsys.readouterr().out.split()
    data = tape_path.read_bytes()
    header = read_tape(data).header
    assert printed_tape == tape_hash(data)
    assert header.city_pack_hash.hex() == printed_pack
    assert (header.seed, header.residents, header.stores) == (3, 2000, 4)
    assert header.delivery_rows > 0
