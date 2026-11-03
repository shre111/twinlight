import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

from twl_engine.fixtures import ToyCity, build_toy_city, write_pack

GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "golden" / "hashes.toml"
STRICT = os.environ.get("TWL_GOLDEN_STRICT") == "1"

pytestmark = pytest.mark.golden


def probes(city: ToyCity) -> dict[str, int | float]:
    return {
        "buildings": len(city.buildings.x),
        "ties": len(city.ties.targets),
        "roads": len(city.roads.kind),
        "road_points": len(city.roads.x),
        "river_points": len(city.river.x),
        "building_height_sum": float(city.buildings.height.astype(np.float64).sum()),
        "resident_x_sum": float(city.residents.x.astype(np.float64).sum()),
        "resident_y_sum": float(city.residents.y.astype(np.float64).sum()),
    }


def test_toy_pack_matches_the_golden(tmp_path: Path) -> None:
    expected = tomllib.loads(GOLDEN.read_text())["toy_pack"]["seed49_residents5000"]
    city = build_toy_city(seed=49, residents=5_000)
    pack_hash = write_pack(city, tmp_path)
    actual = probes(city)
    if STRICT:
        assert pack_hash == expected["hash"], f"toy pack hash is {pack_hash}; probes {actual}"
        return
    for key, value in actual.items():
        if isinstance(value, int):
            assert value == expected[key], key
        else:
            assert value == pytest.approx(expected[key], rel=1e-6), key
