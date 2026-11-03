import json
import subprocess
import sys
from pathlib import Path

import pytest
from blake3 import blake3

from twl_engine.fixtures.__main__ import main


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


def test_cli_runs_as_a_module(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "twl_engine.fixtures", "--residents", "200", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == blake3((tmp_path / "manifest.json").read_bytes()).hexdigest()
