"""Raw (uncompressed) bytes of each section: a u64 row count, then little-endian columns.

Every column is zero-padded to a multiple of 8 bytes as in the pack format (ARCHITECTURE
section 5). The row count comes first because padding makes the row count ambiguous from
the raw length alone, and the header carries no per-section counts.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_tape.format import (
    COLUMN_ALIGN,
    AggregatesSection,
    ChatterSection,
    DeliveriesSection,
    ResidentsSection,
    StoresSection,
    TapeError,
)

U8, U16, U32, U64, F32 = "<u1", "<u2", "<u4", "<u8", "<f4"

Column = tuple[NDArray[Any], str, int]


def encode_residents(section: ResidentsSection) -> bytes:
    n = section.aware_at.shape[0]
    return _encode(
        n,
        [
            (section.aware_at, U32, n),
            (section.first_order_at, U32, n),
            (section.order_count, U16, n),
            (section.heard_from, U32, n),
            (section.first_source, U8, n),
            (section.outcome_flags, U8, n),
        ],
    )


def decode_residents(raw: bytes) -> ResidentsSection:
    cursor = _Cursor(raw)
    n = cursor.count()
    section = ResidentsSection(
        aware_at=cursor.column(U32, n),
        first_order_at=cursor.column(U32, n),
        order_count=cursor.column(U16, n),
        heard_from=cursor.column(U32, n),
        first_source=cursor.column(U8, n),
        outcome_flags=cursor.column(U8, n),
    )
    cursor.finish()
    return section


def encode_chatter(section: ChatterSection) -> bytes:
    m = section.at.shape[0]
    return _encode(m, [(section.from_, U32, m), (section.to, U32, m), (section.at, U32, m)])


def decode_chatter(raw: bytes) -> ChatterSection:
    cursor = _Cursor(raw)
    m = cursor.count()
    section = ChatterSection(
        from_=cursor.column(U32, m), to=cursor.column(U32, m), at=cursor.column(U32, m)
    )
    cursor.finish()
    return section


def encode_deliveries(section: DeliveriesSection) -> bytes:
    d = section.ordered_at.shape[0]
    return _encode(
        d,
        [
            (section.store, U16, d),
            (section.resident, U32, d),
            (section.ordered_at, U32, d),
            (section.picked_at, U32, d),
            (section.delivered_at, U32, d),
            (section.outcome, U8, d),
        ],
    )


def decode_deliveries(raw: bytes) -> DeliveriesSection:
    cursor = _Cursor(raw)
    d = cursor.count()
    section = DeliveriesSection(
        store=cursor.column(U16, d),
        resident=cursor.column(U32, d),
        ordered_at=cursor.column(U32, d),
        picked_at=cursor.column(U32, d),
        delivered_at=cursor.column(U32, d),
        outcome=cursor.column(U8, d),
    )
    cursor.finish()
    return section


def encode_stores(section: StoresSection, ticks: int) -> bytes:
    if section.stockout_offsets.shape[0] == 0:
        raise ValueError("stockout_offsets needs stores + 1 entries")
    stores = section.stockout_offsets.shape[0] - 1
    intervals = 2 * int(section.stockout_offsets[-1])
    return _encode(
        stores,
        [
            (section.stock, F32, stores * ticks),
            (section.orders, U16, stores * ticks),
            (section.stockout_offsets, U32, stores + 1),
            (section.stockout_intervals, U32, intervals),
        ],
    )


def decode_stores(raw: bytes, ticks: int) -> StoresSection:
    cursor = _Cursor(raw)
    stores = cursor.count()
    stock = cursor.column(F32, stores * ticks)
    orders = cursor.column(U16, stores * ticks)
    offsets = cursor.column(U32, stores + 1)
    intervals = cursor.column(U32, 2 * int(offsets[-1]))
    cursor.finish()
    return StoresSection(
        stock=stock, orders=orders, stockout_offsets=offsets, stockout_intervals=intervals
    )


def encode_aggregates(section: AggregatesSection, ticks: int) -> bytes:
    return _encode(
        ticks,
        [
            (section.aware, U32, ticks),
            (section.adopted, U32, ticks),
            (section.orders, U32, ticks),
            (section.stores_out, U16, ticks),
            (section.rain_intensity, F32, ticks),
        ],
    )


def decode_aggregates(raw: bytes, ticks: int) -> AggregatesSection:
    cursor = _Cursor(raw)
    if cursor.count() != ticks:
        raise TapeError("aggregates row count does not match sim_minutes / tick_minutes")
    section = AggregatesSection(
        aware=cursor.column(U32, ticks),
        adopted=cursor.column(U32, ticks),
        orders=cursor.column(U32, ticks),
        stores_out=cursor.column(U16, ticks),
        rain_intensity=cursor.column(F32, ticks),
    )
    cursor.finish()
    return section


def _encode(count: int, columns: list[Column]) -> bytes:
    out = bytearray(np.array([count], dtype=U64).tobytes())
    for values, dtype, length in columns:
        if values.shape != (length,):
            raise ValueError(f"column of shape {values.shape} where ({length},) was expected")
        out += np.ascontiguousarray(values.astype(dtype, casting="safe")).tobytes()
        out += bytes(-len(out) % COLUMN_ALIGN)
    return bytes(out)


class _Cursor:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.pos = 0

    def count(self) -> int:
        return int(self.column(U64, 1)[0])

    def column(self, dtype: str, length: int) -> NDArray[Any]:
        size = np.dtype(dtype).itemsize * length
        end = self.pos + size
        if end > len(self.raw):
            raise TapeError("section is shorter than its columns")
        values = np.frombuffer(self.raw, dtype=dtype, count=length, offset=self.pos).copy()
        self.pos = end + (-end % COLUMN_ALIGN)
        return values

    def finish(self) -> None:
        if self.pos != len(self.raw):
            raise TapeError("section has trailing bytes after its columns")
