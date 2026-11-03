import dataclasses
import struct

import numpy as np
import pytest
from blake3 import blake3

from twl_tape import (
    NEVER,
    AggregatesSection,
    ChatterSection,
    DeliveriesSection,
    ResidentsSection,
    SectionKind,
    StoresSection,
    Tape,
    TapeError,
    TapeHeader,
    derive_run_id,
    read_tape,
    write_tape,
)
from twl_tape.format import FOOTER_STRUCT, HEADER_STRUCT, TABLE_ROW_STRUCT
from twl_tape.sections import decode_chatter, encode_chatter
from twl_tape.writer import HEADER_SIZE

PACK, SCENARIO, PARAMS = (
    blake3(b"pack").digest(),
    blake3(b"scenario").digest(),
    blake3(b"p").digest(),
)


def small_header(residents: int = 3, stores: int = 2, ticks: int = 4) -> TapeHeader:
    return TapeHeader(
        run_id=derive_run_id(PACK, SCENARIO, PARAMS, 7),
        seed=7,
        engine_version="twl-engine 1",
        city_pack_hash=PACK,
        scenario_hash=SCENARIO,
        sim_minutes=ticks * 15,
        tick_minutes=15,
        residents=residents,
        stores=stores,
        chatter_rows=1,
        delivery_rows=0,
        stockout_pairs=1,
        flags=0,
    )


def small_tape(residents: int = 3, stores: int = 2, ticks: int = 4) -> Tape:
    series = stores * ticks
    return Tape(
        header=small_header(residents, stores, ticks),
        residents=ResidentsSection(
            aware_at=np.full(residents, NEVER, "<u4"),
            first_order_at=np.full(residents, NEVER, "<u4"),
            order_count=np.zeros(residents, "<u2"),
            heard_from=np.full(residents, NEVER, "<u4"),
            first_source=np.zeros(residents, "<u1"),
            outcome_flags=np.zeros(residents, "<u1"),
        ),
        chatter=ChatterSection(
            from_=np.array([0], "<u4"), to=np.array([1], "<u4"), at=np.array([30], "<u4")
        ),
        deliveries=DeliveriesSection(
            store=np.zeros(0, "<u2"),
            resident=np.zeros(0, "<u4"),
            ordered_at=np.zeros(0, "<u4"),
            picked_at=np.zeros(0, "<u4"),
            delivered_at=np.zeros(0, "<u4"),
            outcome=np.zeros(0, "<u1"),
        ),
        stores=StoresSection(
            stock=np.arange(series, dtype="<f4"),
            orders=np.zeros(series, "<u2"),
            stockout_offsets=np.array([0, 0, 1], "<u4"),
            stockout_intervals=np.array([15, 45], "<u4"),
        ),
        aggregates=AggregatesSection(
            aware=np.zeros(ticks, "<u4"),
            adopted=np.zeros(ticks, "<u4"),
            orders=np.zeros(ticks, "<u4"),
            stores_out=np.zeros(ticks, "<u2"),
            rain_intensity=np.zeros(ticks, "<f4"),
        ),
    )


def table_rows(data: bytes) -> list[tuple[int, int, int, int, int]]:
    fixed = struct.calcsize(HEADER_STRUCT)
    size = struct.calcsize(TABLE_ROW_STRUCT)
    return [
        struct.unpack(TABLE_ROW_STRUCT, data[fixed + i * size : fixed + (i + 1) * size])
        for i in range(len(SectionKind))
    ]


def reseal(body: bytearray) -> bytes:
    return bytes(body) + struct.pack(FOOTER_STRUCT, len(body), blake3(bytes(body)).digest())


def test_odd_row_counts_round_trip_through_padding() -> None:
    header = small_header()
    chatter = ChatterSection(
        from_=np.array([3], "<u4"), to=np.array([4], "<u4"), at=np.array([5], "<u4")
    )
    raw = encode_chatter(chatter, header)
    assert len(raw) == 3 * 8
    assert decode_chatter(raw, header).at.tolist() == [5]


def test_header_is_352_bytes_and_every_section_is_aligned() -> None:
    data = write_tape(small_tape())
    rows = table_rows(data)
    assert HEADER_SIZE == 352
    assert [row[0] for row in rows] == [int(kind) for kind in SectionKind]
    assert rows[SectionKind.DELIVERIES - 1][3] == 0
    assert rows[SectionKind.CHUNKS - 1][3] == 0
    assert all(row[2] > 0 for row in rows), "empty sections are still a valid zstd frame"
    assert all(row[1] % 8 == 0 for row in rows)


def test_header_counts_must_match_the_arrays() -> None:
    tape = small_tape()
    wrong = dataclasses.replace(tape.header, chatter_rows=2)
    with pytest.raises(ValueError, match="chatter_rows"):
        write_tape(dataclasses.replace(tape, header=wrong))


def test_section_corruption_fails_the_crc_even_with_a_fresh_footer() -> None:
    data = write_tape(small_tape())
    footer = struct.calcsize(FOOTER_STRUCT)
    body = bytearray(data[:-footer])
    offset = table_rows(data)[SectionKind.STORES - 1][1]
    body[offset + 6] ^= 0xFF
    with pytest.raises(TapeError, match="STORES section crc32"):
        read_tape(reseal(body))


def test_raw_length_is_checked_against_the_decompressed_size() -> None:
    data = write_tape(small_tape())
    footer = struct.calcsize(FOOTER_STRUCT)
    body = bytearray(data[:-footer])
    row_start = struct.calcsize(HEADER_STRUCT) + (SectionKind.CHATTER - 1) * 32
    kind, offset, length, raw_length, crc = struct.unpack(
        TABLE_ROW_STRUCT, body[row_start : row_start + 32]
    )
    body[row_start : row_start + 32] = struct.pack(
        TABLE_ROW_STRUCT, kind, offset, length, raw_length + 8, crc
    )
    with pytest.raises(TapeError, match="CHATTER section decompressed"):
        read_tape(reseal(body))


def test_stockout_intervals_follow_the_csr_offsets() -> None:
    tape = read_tape(write_tape(small_tape()))
    assert tape.header.flags == 1
    assert tape.header.stores == 2 and tape.header.stockout_pairs == 1
    assert tape.stores.stockout_offsets.tolist() == [0, 0, 1]
    assert tape.stores.stockout_intervals.tolist() == [15, 45]
    assert tape.stores.stock.tolist() == list(range(8))
