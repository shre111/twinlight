from enum import IntEnum

import numpy as np


class Stream(IntEnum):
    POPULATION = 0
    PERSONAS = 1
    GRAPH = 2
    STORES = 3


def stream(seed: int, which: Stream) -> np.random.Generator:
    """The only source of randomness in citybake: one independent child stream per stage."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(len(Stream))[which])
