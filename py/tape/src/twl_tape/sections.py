"""Raw (uncompressed) bytes of each section: little-endian columns in the documented order.

Every column is zero-padded to a multiple of 8 bytes as in the pack format (ARCHITECTURE
section 5). Column lengths come from the header counts, never from the byte size.
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
    SectionKind,
    StoresSection,
    TapeError,
    TapeHeader,
)

U8, U16, U32, F32 = "<u1", "<u2", "<u4", "<f4"

Column = tuple[NDArray[Any], str, int]


def encode_residents(section: ResidentsSection, header: TapeHeader) -> bytes:
    n = header.residents
    return _encode(
        [
            (section.aware_at, U32, n),
            (section.first_order_at, U32, n),
            (section.order_count, U16, n),
            (section.heard_from, U32, n),
            (section.first_source, U8, n),
            (section.outcome_flags, U8, n),
        ]
    )


def decode_residents(raw: bytes, header: TapeHeader) -> ResidentsSection:
    cursor = _Cursor(raw)
    n = header.residents
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


def encode_chatter(section: ChatterSection, header: TapeHeader) -> bytes:
    m = header.chatter_rows
    return _encode([(section.from_, U32, m), (section.to, U32, m), (section.at, U32, m)])


def decode_chatter(raw: bytes, header: TapeHeader) -> ChatterSection:
    cursor = _Cursor(raw)
    m = header.chatter_rows
    section = ChatterSection(
        from_=cursor.column(U32, m), to=cursor.column(U32, m), at=cursor.column(U32, m)
    )
    cursor.finish()
    return section


def encode_deliveries(section: DeliveriesSection, header: TapeHeader) -> bytes:
    d = header.delivery_rows
    return _encode(
        [
            (section.store, U16, d),
            (section.resident, U32, d),
            (section.ordered_at, U32, d),
            (section.picked_at, U32, d),
            (section.delivered_at, U32, d),
            (section.outcome, U8, d),
        ]
    )


def decode_deliveries(raw: bytes, header: TapeHeader) -> DeliveriesSection:
    cursor = _Cursor(raw)
    d = header.delivery_rows
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


def encode_stores(section: StoresSection, header: TapeHeader) -> bytes:
    series = header.stores * header.ticks
    return _encode(
        [
            (section.stock, F32, series),
            (section.orders, U16, series),
            (section.stockout_offsets, U32, header.stores + 1),
            (section.stockout_intervals, U32, 2 * header.stockout_pairs),
        ]
    )


def decode_stores(raw: bytes, header: TapeHeader) -> StoresSection:
    cursor = _Cursor(raw)
    series = header.stores * header.ticks
    section = StoresSection(
        stock=cursor.column(F32, series),
        orders=cursor.column(U16, series),
        stockout_offsets=cursor.column(U32, header.stores + 1),
        stockout_intervals=cursor.column(U32, 2 * header.stockout_pairs),
    )
    cursor.finish()
    try:
        check_stockout_offsets(section.stockout_offsets, header.stockout_pairs)
    except ValueError as error:
        raise TapeError(str(error)) from error
    return section


def check_stockout_offsets(offsets: NDArray[Any], stockout_pairs: int) -> None:
    """CSR offsets start at 0, never decrease and end at the number of (start, end) pairs."""
    if offsets.shape[0] == 0 or int(offsets[0]) != 0:
        raise ValueError("stockout_offsets must start at 0")
    if np.any(np.diff(offsets.astype(np.int64)) < 0):
        raise ValueError("stockout_offsets must be non-decreasing")
    if int(offsets[-1]) != stockout_pairs:
        raise ValueError("stockout_offsets does not end at stockout_pairs")


def raw_lengths(header: TapeHeader) -> dict[SectionKind, int]:
    """Uncompressed size of every section implied by the header counts (CHUNKS is empty)."""
    n, m, d = header.residents, header.chatter_rows, header.delivery_rows
    series, offsets, intervals = (
        header.stores * header.ticks,
        header.stores + 1,
        2 * header.stockout_pairs,
    )
    ticks = header.ticks
    return {
        SectionKind.RESIDENTS: _padded(4, n) * 3 + _padded(2, n) + _padded(1, n) * 2,
        SectionKind.CHATTER: _padded(4, m) * 3,
        SectionKind.DELIVERIES: _padded(2, d) + _padded(4, d) * 3 + _padded(1, d),
        SectionKind.STORES: _padded(4, series)
        + _padded(2, series)
        + _padded(4, offsets)
        + _padded(4, intervals),
        SectionKind.AGGREGATES: _padded(4, ticks) * 3 + _padded(2, ticks) + _padded(4, ticks),
        SectionKind.CHUNKS: 0,
    }


def _padded(itemsize: int, length: int) -> int:
    size = itemsize * length
    return size + (-size % COLUMN_ALIGN)


def encode_aggregates(section: AggregatesSection, header: TapeHeader) -> bytes:
    ticks = header.ticks
    return _encode(
        [
            (section.aware, U32, ticks),
            (section.adopted, U32, ticks),
            (section.orders, U32, ticks),
            (section.stores_out, U16, ticks),
            (section.rain_intensity, F32, ticks),
        ]
    )


def decode_aggregates(raw: bytes, header: TapeHeader) -> AggregatesSection:
    cursor = _Cursor(raw)
    ticks = header.ticks
    section = AggregatesSection(
        aware=cursor.column(U32, ticks),
        adopted=cursor.column(U32, ticks),
        orders=cursor.column(U32, ticks),
        stores_out=cursor.column(U16, ticks),
        rain_intensity=cursor.column(F32, ticks),
    )
    cursor.finish()
    return section


def _encode(columns: list[Column]) -> bytes:
    out = bytearray()
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

    def column(self, dtype: str, length: int) -> NDArray[Any]:
        size = np.dtype(dtype).itemsize * length
        end = self.pos + size
        if end > len(self.raw):
            raise TapeError("section is shorter than the header counts require")
        values = np.frombuffer(self.raw, dtype=dtype, count=length, offset=self.pos).copy()
        self.pos = end + (-end % COLUMN_ALIGN)
        return values

    def finish(self) -> None:
        if self.pos != len(self.raw):
            raise TapeError("section has trailing bytes after its columns")
