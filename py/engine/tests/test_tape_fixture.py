from pathlib import Path

import pytest

from twl_engine import goldens
from twl_tape import tape_hash
from twl_tape.fixtures import FIXTURE_DIR

pytestmark = pytest.mark.golden


@pytest.mark.parametrize("run", goldens.TAPE_FIXTURE_RUNS)
def test_the_checked_in_engine_tape_is_the_golden_run(run: str) -> None:
    name = goldens.tape_fixture_name(run)
    data = (FIXTURE_DIR / f"{name}.twlt").read_bytes()
    golden = goldens.read_goldens()["engine"][run]
    assert tape_hash(data) == golden["hash"], (
        f"{name}.twlt is not the [engine.{run}] golden; `just goldens` writes both together"
    )
    assert len(data) == golden["tape_bytes"]


@pytest.mark.parametrize("write", [False, True])
def test_only_write_puts_the_golden_tape_in_the_fixture_directory(
    write: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tape = (FIXTURE_DIR / "small.twlt").read_bytes()
    written: dict[str, bytes] = {}
    monkeypatch.setenv("TWL_GOLDEN_STRICT", "1")
    monkeypatch.setattr(goldens, "GOLDEN_PATH", tmp_path / "hashes.toml")
    monkeypatch.setattr(goldens, "compute_engine_goldens", lambda: ({}, {"engine_x": tape}))
    monkeypatch.setattr(goldens, "write_fixture", written.__setitem__)

    goldens.main(["--write"] if write else [])

    assert written == ({"engine_x": tape} if write else {})
    assert (tmp_path / "hashes.toml").exists() is write
