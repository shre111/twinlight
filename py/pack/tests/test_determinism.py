import dataclasses
import os
import subprocess
import sys

import numpy as np

from twl_pack.writer import encode_pack

SCRIPT = """
import sys
from twl_pack.fixture import small_contents
from twl_pack.writer import encode_pack
e = encode_pack(small_contents())
sys.stdout.write(e.pack_hash)
"""


def test_encoding_twice_gives_identical_bytes_and_hash(city, toy):
    for contents in (city, toy):
        a, b = encode_pack(contents), encode_pack(contents)
        assert a.manifest == b.manifest
        assert a.pack_hash == b.pack_hash
        assert a.raw == b.raw
        assert a.objects == b.objects


def test_encoding_does_not_mutate_its_input(city):
    before = {
        f.name: getattr(city.buildings, f.name).copy() for f in dataclasses.fields(city.buildings)
    }
    residents_before = city.residents.building.copy()
    encode_pack(city)
    for name, value in before.items():
        assert np.array_equal(getattr(city.buildings, name), value)
    assert np.array_equal(city.residents.building, residents_before)


def hash_in_subprocess(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout


def test_pack_hash_is_independent_of_pythonhashseed():
    from twl_pack.fixture import small_contents

    here = encode_pack(small_contents()).pack_hash
    assert hash_in_subprocess("0") == here
    assert hash_in_subprocess("987654") == here
