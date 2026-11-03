"""Run tape v1 (ARCHITECTURE.md section 4): header, section dataclasses and constants."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from blake3 import blake3
from numpy.typing import NDArray

TAPE_VERSION = 1
MINUTES_30_DAYS = 43_200
TICK_MINUTES = 15
NEVER = 0xFFFF_FFFF

MAGIC = b"TWLT"
FLAG_COMPLETE = 1
RUN_ID_BYTES = 16
HASH_BYTES = 32
ENGINE_VERSION_BYTES = 32
COLUMN_ALIGN = 8
ZSTD_LEVEL = 3

HEADER_STRUCT = "<4sHH16sQ32s32s32sII"
TABLE_ROW_STRUCT = "<IQQQI"
FOOTER_STRUCT = "<Q32s"


class TapeError(ValueError):
    """The bytes are not a well-formed, intact tape."""


class SectionKind(IntEnum):
    RESIDENTS = 1
    CHATTER = 2
    DELIVERIES = 3
    STORES = 4
    AGGREGATES = 5
    CHUNKS = 6


@dataclass(frozen=True)
class TapeHeader:
    run_id: bytes
    seed: int
    engine_version: str
    city_pack_hash: bytes
    scenario_hash: bytes
    sim_minutes: int
    tick_minutes: int
    flags: int

    @property
    def ticks(self) -> int:
        return self.sim_minutes // self.tick_minutes


@dataclass(frozen=True, eq=False)
class ResidentsSection:
    aware_at: NDArray[np.uint32]
    first_order_at: NDArray[np.uint32]
    order_count: NDArray[np.uint16]
    heard_from: NDArray[np.uint32]
    first_source: NDArray[np.uint8]
    outcome_flags: NDArray[np.uint8]


@dataclass(frozen=True, eq=False)
class ChatterSection:
    from_: NDArray[np.uint32]
    to: NDArray[np.uint32]
    at: NDArray[np.uint32]


@dataclass(frozen=True, eq=False)
class DeliveriesSection:
    store: NDArray[np.uint16]
    resident: NDArray[np.uint32]
    ordered_at: NDArray[np.uint32]
    picked_at: NDArray[np.uint32]
    delivered_at: NDArray[np.uint32]
    outcome: NDArray[np.uint8]


@dataclass(frozen=True, eq=False)
class StoresSection:
    stock: NDArray[np.float32]
    orders: NDArray[np.uint16]
    stockout_offsets: NDArray[np.uint32]
    stockout_intervals: NDArray[np.uint32]


@dataclass(frozen=True, eq=False)
class AggregatesSection:
    aware: NDArray[np.uint32]
    adopted: NDArray[np.uint32]
    orders: NDArray[np.uint32]
    stores_out: NDArray[np.uint16]
    rain_intensity: NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class Tape:
    header: TapeHeader
    residents: ResidentsSection
    chatter: ChatterSection
    deliveries: DeliveriesSection
    stores: StoresSection
    aggregates: AggregatesSection


def derive_run_id(
    city_pack_hash: bytes, scenario_hash: bytes, params_hash: bytes, seed: int
) -> bytes:
    """First 16 bytes of blake3 over the inputs that determine the tape (ARCHITECTURE section 4)."""
    for name, digest in (
        ("city_pack_hash", city_pack_hash),
        ("scenario_hash", scenario_hash),
        ("params_hash", params_hash),
    ):
        if len(digest) != HASH_BYTES:
            raise ValueError(f"{name} must be {HASH_BYTES} bytes, got {len(digest)}")
    material = city_pack_hash + scenario_hash + params_hash + seed.to_bytes(8, "little")
    return blake3(material).digest()[:RUN_ID_BYTES]
