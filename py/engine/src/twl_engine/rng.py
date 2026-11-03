from collections.abc import Sequence

import numpy as np


def spawn_streams(seed: int, names: Sequence[str]) -> dict[str, np.random.Generator]:
    """One independent PCG64 stream per subsystem, children of `SeedSequence(seed)` in order."""
    if len(dict.fromkeys(names)) != len(names):
        raise ValueError(f"stream names must be unique: {list(names)}")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(names, children, strict=True)
    }
