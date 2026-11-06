import struct
import zlib

import zstandard
from blake3 import blake3

from twl_tape.format import (
    COLUMN_ALIGN,
    FLAG_COMPLETE,
    FOOTER_STRUCT,
    HEADER_STRUCT,
    MAGIC,
    TABLE_ROW_STRUCT,
    TAPE_VERSION,
    SectionKind,
    Tape,
    TapeError,
    TapeHeader,
)
from twl_tape.sections import (
    decode_aggregates,
    decode_chatter,
    decode_deliveries,
    decode_residents,
    decode_stores,
    raw_lengths,
)
from twl_tape.writer import HEADER_SIZE, SECTION_COUNT

_FIXED_SIZE = struct.calcsize(HEADER_STRUCT)
_ROW_SIZE = struct.calcsize(TABLE_ROW_STRUCT)
_FOOTER_SIZE = struct.calcsize(FOOTER_STRUCT)

TableRow = tuple[int, int, int, int, int]


def read_tape(data: bytes) -> Tape:
    """Decode a complete tape, verifying magic, version, section crc32s and the footer hash."""
    body = _verify_footer(data)
    header = _read_header(body)
    expected = raw_lengths(header)
    raw = {
        kind: _read_section(body, row, expected[kind])
        for kind, row in zip(SectionKind, _read_table(body), strict=True)
    }
    return Tape(
        header=header,
        residents=decode_residents(raw[SectionKind.RESIDENTS], header),
        chatter=decode_chatter(raw[SectionKind.CHATTER], header),
        deliveries=decode_deliveries(raw[SectionKind.DELIVERIES], header),
        stores=decode_stores(raw[SectionKind.STORES], header),
        aggregates=decode_aggregates(raw[SectionKind.AGGREGATES], header),
    )


def read_footer(data: bytes) -> tuple[int, bytes]:
    """The stored `(total_length, blake3 digest)`, verified against the bytes before the footer."""
    if len(data) < HEADER_SIZE + _FOOTER_SIZE:
        raise TapeError("file is too short to be a tape")
    total_length, digest = struct.unpack(FOOTER_STRUCT, data[-_FOOTER_SIZE:])
    if total_length != len(data) - _FOOTER_SIZE:
        raise TapeError("footer total_length does not match the file size")
    if blake3(data[:total_length]).digest() != digest:
        raise TapeError("footer hash does not match the file contents")
    return total_length, digest


def read_section_table(data: bytes) -> list[TableRow]:
    """The stored `(kind, offset, length, raw_length, crc32)` rows, in kind order."""
    return _read_table(_verify_footer(data))


def _verify_footer(data: bytes) -> bytes:
    total_length, _digest = read_footer(data)
    return data[:total_length]


def _read_header(body: bytes) -> TapeHeader:
    (
        magic,
        version,
        flags,
        run_id,
        seed,
        engine_version,
        city_pack_hash,
        scenario_hash,
        sim_minutes,
        tick_minutes,
        residents,
        stores,
        chatter_rows,
        delivery_rows,
        stockout_pairs,
        reserved,
    ) = struct.unpack(HEADER_STRUCT, body[:_FIXED_SIZE])
    if magic != MAGIC:
        raise TapeError("not a tape (bad magic)")
    if version != TAPE_VERSION:
        raise TapeError(f"unsupported tape version {version}")
    if not flags & FLAG_COMPLETE:
        raise TapeError("tape is not marked complete (flags bit0)")
    if reserved != 0:
        raise TapeError("reserved header field must be zero")
    if tick_minutes == 0 or sim_minutes % tick_minutes:
        raise TapeError("sim_minutes must be a positive multiple of tick_minutes")
    try:
        engine_version_text = engine_version.rstrip(b"\0").decode("ascii")
    except UnicodeDecodeError as error:
        raise TapeError("engine_version is not ASCII") from error
    return TapeHeader(
        run_id=run_id,
        seed=seed,
        engine_version=engine_version_text,
        city_pack_hash=city_pack_hash,
        scenario_hash=scenario_hash,
        sim_minutes=sim_minutes,
        tick_minutes=tick_minutes,
        residents=residents,
        stores=stores,
        chatter_rows=chatter_rows,
        delivery_rows=delivery_rows,
        stockout_pairs=stockout_pairs,
        flags=flags,
    )


def _read_table(body: bytes) -> list[TableRow]:
    rows: list[TableRow] = []
    for index, kind in enumerate(SectionKind):
        start = _FIXED_SIZE + index * _ROW_SIZE
        row: TableRow = struct.unpack(TABLE_ROW_STRUCT, body[start : start + _ROW_SIZE])
        if row[0] != int(kind):
            raise TapeError(f"section table entry {index} is kind {row[0]}, expected {int(kind)}")
        rows.append(row)
    if len(rows) != SECTION_COUNT:
        raise TapeError("section table is incomplete")
    _check_layout(rows, len(body))
    return rows


def _check_layout(rows: list[TableRow], total_length: int) -> None:
    previous_end = HEADER_SIZE
    previous_offset = -1
    for kind, offset, length, _raw_length, _crc in rows:
        name = SectionKind(kind).name
        if offset % COLUMN_ALIGN or offset < HEADER_SIZE:
            raise TapeError(f"{name} section offset {offset} is misaligned or inside the header")
        if offset <= previous_offset or offset < previous_end:
            raise TapeError(f"{name} section overlaps or precedes the section before it")
        if offset + length > total_length:
            raise TapeError(f"{name} section runs past the end of the file")
        previous_offset = offset
        previous_end = offset + length
    if previous_end + (-previous_end % COLUMN_ALIGN) != total_length:
        raise TapeError("the last section does not end at total_length")


def _read_section(body: bytes, row: TableRow, expected_raw_length: int) -> bytes:
    kind, offset, length, raw_length, crc = row
    name = SectionKind(kind).name
    if raw_length != expected_raw_length:
        raise TapeError(
            f"{name} section claims {raw_length} raw bytes; the header counts need "
            f"{expected_raw_length}"
        )
    compressed = body[offset : offset + length]
    if zlib.crc32(compressed) != crc:
        raise TapeError(f"{name} section crc32 mismatch")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            compressed, max_output_size=max(raw_length, 1)
        )
    except zstandard.ZstdError as error:
        raise TapeError(f"{name} section is not a valid zstd frame: {error}") from error
    if len(raw) != raw_length:
        raise TapeError(f"{name} section decompressed to {len(raw)} bytes, expected {raw_length}")
    return raw
