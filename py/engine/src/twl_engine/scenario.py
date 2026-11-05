import dataclasses
import json
from dataclasses import dataclass

from blake3 import blake3

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class Scenario:
    """What is launched and when. Empty for now: P-012 adds the campaign and P-013 the stock."""

    days: int = 30
    tick_minutes: int = 15

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError("a scenario runs for at least one day")
        if self.tick_minutes < 1:
            raise ValueError("tick_minutes must be at least 1")

    @property
    def sim_minutes(self) -> int:
        return self.days * MINUTES_PER_DAY

    def canonical_json(self) -> bytes:
        return _canonical(dataclasses.asdict(self))

    def hash(self) -> bytes:
        return blake3(self.canonical_json()).digest()


@dataclass(frozen=True)
class Params:
    """Per-run model parameters; empty until calibration lands, but hashed into the run id."""

    def canonical_json(self) -> bytes:
        return _canonical(dataclasses.asdict(self))

    def hash(self) -> bytes:
        return blake3(self.canonical_json()).digest()


EMPTY = Scenario()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
