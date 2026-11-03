from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_tape import NEVER


@dataclass(eq=False)
class ResidentState:
    """Per-resident struct of arrays; the first six columns are written to the tape as they are."""

    aware_at: NDArray[np.uint32]
    first_order_at: NDArray[np.uint32]
    order_count: NDArray[np.uint16]
    heard_from: NDArray[np.uint32]
    first_source: NDArray[np.uint8]
    outcome_flags: NDArray[np.uint8]
    awareness: NDArray[np.float32]

    @property
    def count(self) -> int:
        return self.aware_at.shape[0]


def new_resident_state(n: int) -> ResidentState:
    """Nobody aware, nobody ordered, awareness A_r = 0 (MODEL.md section 3)."""
    return ResidentState(
        aware_at=np.full(n, NEVER, dtype=np.uint32),
        first_order_at=np.full(n, NEVER, dtype=np.uint32),
        order_count=np.zeros(n, dtype=np.uint16),
        heard_from=np.full(n, NEVER, dtype=np.uint32),
        first_source=np.zeros(n, dtype=np.uint8),
        outcome_flags=np.zeros(n, dtype=np.uint8),
        awareness=np.zeros(n, dtype=np.float32),
    )
