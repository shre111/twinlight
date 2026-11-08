from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ROAD_STREET = 0
ROAD_ARTERIAL = 1
ROAD_BRIDGE = 2

TIE_NEIGHBOUR = 1
TIE_FRIEND = 2

NO_PERSONA = 0xFFFF_FFFF

DEFAULT_TIMEZONE = "Asia/Kolkata"
SATURDAY_SUNDAY = (6, 7)


@dataclass(frozen=True, eq=False)
class Buildings:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    width: NDArray[np.float32]
    depth: NDArray[np.float32]
    height: NDArray[np.float32]
    rotation: NDArray[np.float32]
    district: NDArray[np.uint16]


@dataclass(frozen=True, eq=False)
class Roads:
    offsets: NDArray[np.uint32]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    kind: NDArray[np.uint8]


@dataclass(frozen=True, eq=False)
class River:
    offsets: NDArray[np.uint32]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    kind: NDArray[np.uint8]


@dataclass(frozen=True)
class District:
    id: int
    name: str
    x: float
    y: float
    east: bool


@dataclass(frozen=True, eq=False)
class Residents:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    z: NDArray[np.float32]
    building: NDArray[np.uint32]
    hex: NDArray[np.uint32]
    segment: NDArray[np.uint16]
    persona_ref: NDArray[np.uint32]
    weight: NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class Ties:
    offsets: NDArray[np.uint32]
    targets: NDArray[np.uint32]
    kind: NDArray[np.uint8]
    strength: NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class Stores:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    hex: NDArray[np.uint32]
    radius_m: NDArray[np.float32]
    district: NDArray[np.uint16]


@dataclass(frozen=True, eq=False)
class ToyCity:
    """`timezone` is an IANA name and `weekend_days` are ISO weekdays, 1 Monday to 7 Sunday
    (D-036)."""

    seed: int
    buildings: Buildings
    roads: Roads
    river: River
    districts: tuple[District, ...]
    residents: Residents
    ties: Ties
    stores: Stores
    timezone: str = DEFAULT_TIMEZONE
    weekend_days: tuple[int, ...] = SATURDAY_SUNDAY
