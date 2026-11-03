import dataclasses
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import blake3
import numpy as np
import pytest
import zstandard
from numpy.typing import NDArray

from twl_tape.format import (
    NEVER,
    TAPE_VERSION,
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
)
from twl_tape.reader import read_tape
from twl_tape.writer import tape_hash, write_tape

HEADER_FIXED = struct.Struct("<4sHH16sQ32s32s32sII")
ROW = struct.Struct("<IQQQI")
TABLE_ROWS = 6
TABLE_START = HEADER_FIXED.size
BODY_START = TABLE_START + TABLE_ROWS * ROW.size
FOOTER_SIZE = 8 + 32

KIND_ORDER = [
    SectionKind.RESIDENTS,
    SectionKind.CHATTER,
    SectionKind.DELIVERIES,
    SectionKind.STORES,
    SectionKind.AGGREGATES,
    SectionKind.CHUNKS,
]

COLUMNS: dict[SectionKind, list[tuple[str, str]]] = {
    SectionKind.RESIDENTS: [
        ("aware_at", "<u4"),
        ("first_order_at", "<u4"),
        ("order_count", "<u2"),
        ("heard_from", "<u4"),
        ("first_source", "u1"),
        ("outcome_flags", "u1"),
    ],
    SectionKind.CHATTER: [("from_", "<u4"), ("to", "<u4"), ("at", "<u4")],
    SectionKind.DELIVERIES: [
        ("store", "<u2"),
        ("resident", "<u4"),
        ("ordered_at", "<u4"),
        ("picked_at", "<u4"),
        ("delivered_at", "<u4"),
        ("outcome", "u1"),
    ],
    SectionKind.STORES: [
        ("stock", "<f4"),
        ("orders", "<u2"),
        ("stockout_offsets", "<u4"),
        ("stockout_intervals", "<u4"),
    ],
    SectionKind.AGGREGATES: [
        ("aware", "<u4"),
        ("adopted", "<u4"),
        ("orders", "<u4"),
        ("stores_out", "<u2"),
        ("rain_intensity", "<f4"),
    ],
}

PACK_HASH = bytes(range(32))
SCENARIO_HASH = bytes(range(100, 132))
PARAMS_HASH = bytes(range(200, 232))


def arr(values: list[float] | list[int], dtype: str) -> NDArray[np.generic]:
    return np.array(values, dtype=dtype)


def make_header(
    sim_minutes: int = 60, seed: int = 7, run_id: bytes | None = None, flags: int = 0
) -> TapeHeader:
    return TapeHeader(
        run_id=run_id or derive_run_id(PACK_HASH, SCENARIO_HASH, PARAMS_HASH, seed),
        seed=seed,
        engine_version="1.0.0-test",
        city_pack_hash=PACK_HASH,
        scenario_hash=SCENARIO_HASH,
        sim_minutes=sim_minutes,
        tick_minutes=15,
        flags=flags,
    )


def small_tape() -> Tape:
    return Tape(
        header=make_header(),
        residents=ResidentsSection(
            aware_at=arr([0, 15, NEVER, 45, NEVER], "<u4"),
            first_order_at=arr([30, NEVER, NEVER, 60, NEVER], "<u4"),
            order_count=arr([2, 0, 0, 1, 0], "<u2"),
            heard_from=arr([NEVER, 0, NEVER, 1, NEVER], "<u4"),
            first_source=arr([1, 2, 0, 2, 0], "u1"),
            outcome_flags=arr([0, 0, 0, 1, 0], "u1"),
        ),
        chatter=ChatterSection(
            from_=arr([0, 1], "<u4"), to=arr([1, 3], "<u4"), at=arr([10, 40], "<u4")
        ),
        deliveries=DeliveriesSection(
            store=arr([1], "<u2"),
            resident=arr([3], "<u4"),
            ordered_at=arr([60], "<u4"),
            picked_at=arr([70], "<u4"),
            delivered_at=arr([85], "<u4"),
            outcome=arr([2], "u1"),
        ),
        stores=StoresSection(
            stock=arr([10.5, 8.0, 0.0, 0.0, 20.0, 19.0, 18.0, 17.25], "<f4"),
            orders=arr([1, 2, 0, 0, 3, 1, 1, 1], "<u2"),
            stockout_offsets=arr([0, 1, 1], "<u4"),
            stockout_intervals=arr([2, 4], "<u4"),
        ),
        aggregates=AggregatesSection(
            aware=arr([1, 2, 2, 3], "<u4"),
            adopted=arr([0, 0, 1, 1], "<u4"),
            orders=arr([0, 0, 2, 3], "<u4"),
            stores_out=arr([0, 0, 1, 1], "<u2"),
            rain_intensity=arr([0.0, 0.5, 1.0, 0.25], "<f4"),
        ),
    )


def empty_tape() -> Tape:
    return Tape(
        header=make_header(sim_minutes=0),
        residents=ResidentsSection(
            aware_at=arr([], "<u4"),
            first_order_at=arr([], "<u4"),
            order_count=arr([], "<u2"),
            heard_from=arr([], "<u4"),
            first_source=arr([], "u1"),
            outcome_flags=arr([], "u1"),
        ),
        chatter=ChatterSection(from_=arr([], "<u4"), to=arr([], "<u4"), at=arr([], "<u4")),
        deliveries=DeliveriesSection(
            store=arr([], "<u2"),
            resident=arr([], "<u4"),
            ordered_at=arr([], "<u4"),
            picked_at=arr([], "<u4"),
            delivered_at=arr([], "<u4"),
            outcome=arr([], "u1"),
        ),
        stores=StoresSection(
            stock=arr([], "<f4"),
            orders=arr([], "<u2"),
            stockout_offsets=arr([0], "<u4"),
            stockout_intervals=arr([], "<u4"),
        ),
        aggregates=AggregatesSection(
            aware=arr([], "<u4"),
            adopted=arr([], "<u4"),
            orders=arr([], "<u4"),
            stores_out=arr([], "<u2"),
            rain_intensity=arr([], "<f4"),
        ),
    )


def section_of(tape: Tape, kind: SectionKind) -> object | None:
    names = {
        SectionKind.RESIDENTS: "residents",
        SectionKind.CHATTER: "chatter",
        SectionKind.DELIVERIES: "deliveries",
        SectionKind.STORES: "stores",
        SectionKind.AGGREGATES: "aggregates",
    }
    return getattr(tape, names[kind]) if kind in names else None


def padded(column: NDArray[np.generic], dtype: str) -> bytes:
    raw = np.ascontiguousarray(column).astype(dtype).tobytes()
    return raw + b"\0" * (-len(raw) % 8)


def expected_raw(tape: Tape, kind: SectionKind) -> bytes:
    section = section_of(tape, kind)
    if section is None:
        return b""
    return b"".join(padded(getattr(section, name), dt) for name, dt in COLUMNS[kind])


@dataclass(frozen=True)
class Row:
    kind: int
    offset: int
    length: int
    raw_length: int
    crc32: int


@dataclass(frozen=True)
class Parsed:
    magic: bytes
    version: int
    flags: int
    run_id: bytes
    seed: int
    engine_version: bytes
    city_pack_hash: bytes
    scenario_hash: bytes
    sim_minutes: int
    tick_minutes: int
    rows: list[Row]
    total_length: int
    footer_hash: bytes


def parse(data: bytes) -> Parsed:
    magic, version, flags, run_id, seed, engine, city, scenario, sim, tick = (
        HEADER_FIXED.unpack_from(data, 0)
    )
    rows = [Row(*ROW.unpack_from(data, TABLE_START + i * ROW.size)) for i in range(TABLE_ROWS)]
    total_length = struct.unpack_from("<Q", data, len(data) - FOOTER_SIZE)[0]
    return Parsed(
        magic,
        version,
        flags,
        run_id,
        seed,
        engine,
        city,
        scenario,
        sim,
        tick,
        rows,
        total_length,
        data[-32:],
    )


def decompress(data: bytes, row: Row) -> bytes:
    chunk = data[row.offset : row.offset + row.length]
    if row.raw_length == 0:
        return zstandard.ZstdDecompressor().decompressobj().decompress(chunk) if chunk else b""
    return zstandard.ZstdDecompressor().decompress(chunk, max_output_size=row.raw_length)


def refooted(body: bytes) -> bytes:
    footer = struct.pack("<Q", len(body)) + blake3.blake3(body).digest()
    return body + footer


def body_of(data: bytes) -> bytes:
    return data[:-FOOTER_SIZE]


def assert_arrays_equal(actual: Any, expected: Any) -> None:
    for field in dataclasses.fields(expected):
        got = getattr(actual, field.name)
        want = getattr(expected, field.name)
        assert got.dtype == want.dtype, field.name
        assert got.shape == want.shape, field.name
        assert np.array_equal(got, want), field.name


@pytest.fixture(scope="module")
def tape() -> Tape:
    return small_tape()


@pytest.fixture(scope="module")
def data(tape: Tape) -> bytes:
    return write_tape(tape)


def test_section_kinds_and_constants_match_the_spec() -> None:
    assert [int(k) for k in KIND_ORDER] == [1, 2, 3, 4, 5, 6]
    assert NEVER == 0xFFFF_FFFF
    assert TAPE_VERSION == 1


def test_small_tape_round_trips_with_every_array_and_dtype_equal(tape: Tape, data: bytes) -> None:
    back = read_tape(data)
    for name in ("residents", "chatter", "deliveries", "stores", "aggregates"):
        assert_arrays_equal(getattr(back, name), getattr(tape, name))


def test_round_trip_header_equals_the_input_header_with_complete_flag(
    tape: Tape, data: bytes
) -> None:
    back = read_tape(data)
    assert back.header == dataclasses.replace(tape.header, flags=tape.header.flags | 1)


def test_empty_tape_round_trips() -> None:
    tape = empty_tape()
    back = read_tape(write_tape(tape))
    for name in ("residents", "chatter", "deliveries", "stores", "aggregates"):
        assert_arrays_equal(getattr(back, name), getattr(tape, name))


def test_file_header_matches_an_independent_parse(tape: Tape, data: bytes) -> None:
    p = parse(data)
    h = tape.header
    assert p.magic == b"TWLT"
    assert p.version == TAPE_VERSION
    assert p.flags & 1 == 1
    assert p.run_id == h.run_id
    assert p.seed == h.seed
    assert p.engine_version == h.engine_version.encode("ascii").ljust(32, b"\0")
    assert p.city_pack_hash == h.city_pack_hash
    assert p.scenario_hash == h.scenario_hash
    assert (p.sim_minutes, p.tick_minutes) == (h.sim_minutes, h.tick_minutes)


def test_section_table_has_six_rows_in_kind_order_with_aligned_offsets(data: bytes) -> None:
    rows = parse(data).rows
    assert [r.kind for r in rows] == [int(k) for k in KIND_ORDER]
    total = parse(data).total_length
    previous_end = BODY_START
    for r in rows:
        assert r.offset % 8 == 0
        assert r.offset >= previous_end
        previous_end = r.offset + r.length
    assert previous_end <= total


def test_section_crc32_covers_the_compressed_bytes(data: bytes) -> None:
    for r in parse(data).rows:
        assert zlib.crc32(data[r.offset : r.offset + r.length]) == r.crc32


def test_raw_lengths_equal_the_padded_column_totals(tape: Tape, data: bytes) -> None:
    for r in parse(data).rows:
        assert r.raw_length == len(expected_raw(tape, SectionKind(r.kind))), r.kind


def test_decompressed_sections_equal_the_padded_little_endian_columns(
    tape: Tape, data: bytes
) -> None:
    for r in parse(data).rows:
        assert decompress(data, r) == expected_raw(tape, SectionKind(r.kind)), r.kind


def test_chunks_section_is_written_empty(data: bytes) -> None:
    chunks = parse(data).rows[5]
    assert chunks.raw_length == 0
    assert decompress(data, chunks) == b""


def test_footer_holds_total_length_and_blake3_of_everything_before_it(data: bytes) -> None:
    p = parse(data)
    assert p.total_length == len(data) - FOOTER_SIZE
    assert p.footer_hash == blake3.blake3(data[: p.total_length]).digest()


def test_tape_hash_is_blake3_hex_of_the_whole_file(data: bytes) -> None:
    assert tape_hash(data) == blake3.blake3(data).hexdigest()


def test_writing_the_same_tape_twice_gives_identical_bytes() -> None:
    assert write_tape(small_tape()) == write_tape(small_tape())


def test_writing_marks_the_file_complete_even_if_the_input_flags_are_clear(tape: Tape) -> None:
    assert tape.header.flags == 0
    assert parse(write_tape(tape)).flags & 1 == 1


def test_changing_one_value_changes_the_tape_hash(tape: Tape, data: bytes) -> None:
    other = dataclasses.replace(
        tape,
        aggregates=dataclasses.replace(tape.aggregates, adopted=arr([0, 0, 1, 2], "<u4")),
    )
    assert tape_hash(write_tape(other)) != tape_hash(data)


def test_corrupting_a_section_byte_raises_tape_error(data: bytes) -> None:
    row = parse(data).rows[0]
    bad = bytearray(data)
    bad[row.offset + row.length // 2] ^= 0xFF
    with pytest.raises(TapeError):
        read_tape(bytes(bad))


def test_section_crc_mismatch_raises_even_when_the_footer_is_recomputed(data: bytes) -> None:
    row = parse(data).rows[0]
    bad = bytearray(body_of(data))
    bad[row.offset + row.length // 2] ^= 0xFF
    with pytest.raises(TapeError):
        read_tape(refooted(bytes(bad)))


def test_wrong_raw_length_raises_even_when_the_footer_is_recomputed(data: bytes) -> None:
    body = bytearray(body_of(data))
    row = parse(data).rows[0]
    at = TABLE_START + 4 + 8 + 8
    struct.pack_into("<Q", body, at, row.raw_length + 8)
    with pytest.raises(TapeError):
        read_tape(refooted(bytes(body)))


def test_wrong_magic_raises_tape_error(data: bytes) -> None:
    body = b"XXXX" + body_of(data)[4:]
    with pytest.raises(TapeError):
        read_tape(refooted(body))


def test_wrong_version_raises_tape_error(data: bytes) -> None:
    body = bytearray(body_of(data))
    struct.pack_into("<H", body, 4, TAPE_VERSION + 1)
    with pytest.raises(TapeError):
        read_tape(refooted(bytes(body)))


def test_wrong_footer_hash_raises_tape_error(data: bytes) -> None:
    bad = bytearray(data)
    bad[-1] ^= 0x01
    with pytest.raises(TapeError):
        read_tape(bytes(bad))


@pytest.mark.parametrize("cut", [0, 10, BODY_START - 1, BODY_START + 5, -1, -20, -FOOTER_SIZE - 1])
def test_truncated_file_raises_tape_error(data: bytes, cut: int) -> None:
    keep = cut if cut >= 0 else len(data) + cut
    with pytest.raises(TapeError):
        read_tape(data[:keep])


def test_dropping_the_footer_raises_tape_error(data: bytes) -> None:
    with pytest.raises(TapeError):
        read_tape(body_of(data))


def test_derive_run_id_is_deterministic_and_sixteen_bytes() -> None:
    a = derive_run_id(PACK_HASH, SCENARIO_HASH, PARAMS_HASH, 5)
    assert a == derive_run_id(PACK_HASH, SCENARIO_HASH, PARAMS_HASH, 5)
    assert isinstance(a, bytes) and len(a) == 16


def test_derive_run_id_is_the_blake3_prefix_of_the_concatenated_inputs() -> None:
    expected = blake3.blake3(
        PACK_HASH + SCENARIO_HASH + PARAMS_HASH + (5).to_bytes(8, "little")
    ).digest()[:16]
    assert derive_run_id(PACK_HASH, SCENARIO_HASH, PARAMS_HASH, 5) == expected


def test_derive_run_id_changes_with_every_input() -> None:
    base = derive_run_id(PACK_HASH, SCENARIO_HASH, PARAMS_HASH, 5)

    def flipped(h: bytes) -> bytes:
        return bytes([h[0] ^ 1]) + h[1:]

    variants = {
        derive_run_id(flipped(PACK_HASH), SCENARIO_HASH, PARAMS_HASH, 5),
        derive_run_id(PACK_HASH, flipped(SCENARIO_HASH), PARAMS_HASH, 5),
        derive_run_id(PACK_HASH, SCENARIO_HASH, flipped(PARAMS_HASH), 5),
        derive_run_id(PACK_HASH, SCENARIO_HASH, PARAMS_HASH, 6),
    }
    assert base not in variants
    assert len(variants) == 4
