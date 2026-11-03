import struct
import zlib

import zstandard
from blake3 import blake3

from twl_tape.format import (
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
    raw = {
        kind: _read_section(body, row)
        for kind, row in zip(SectionKind, _read_table(body), strict=True)
    }
    if raw[SectionKind.CHUNKS]:
        raise TapeError("CHUNKS must be empty in tape version 1")
    ticks = header.ticks
    return Tape(
        header=header,
        residents=decode_residents(raw[SectionKind.RESIDENTS]),
        chatter=decode_chatter(raw[SectionKind.CHATTER]),
        deliveries=decode_deliveries(raw[SectionKind.DELIVERIES]),
        stores=decode_stores(raw[SectionKind.STORES], ticks),
        aggregates=decode_aggregates(raw[SectionKind.AGGREGATES], ticks),
    )


def _verify_footer(data: bytes) -> bytes:
    if len(data) < HEADER_SIZE + _FOOTER_SIZE:
        raise TapeError("file is too short to be a tape")
    total_length, digest = struct.unpack(FOOTER_STRUCT, data[-_FOOTER_SIZE:])
    if total_length != len(data) - _FOOTER_SIZE:
        raise TapeError("footer total_length does not match the file size")
    body = data[:total_length]
    if blake3(body).digest() != digest:
        raise TapeError("footer hash does not match the file contents")
    return body


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
    ) = struct.unpack(HEADER_STRUCT, body[:_FIXED_SIZE])
    if magic != MAGIC:
        raise TapeError("not a tape (bad magic)")
    if version != TAPE_VERSION:
        raise TapeError(f"unsupported tape version {version}")
    if tick_minutes == 0 or sim_minutes % tick_minutes:
        raise TapeError("sim_minutes must be a positive multiple of tick_minutes")
    return TapeHeader(
        run_id=run_id,
        seed=seed,
        engine_version=engine_version.rstrip(b"\0").decode("ascii"),
        city_pack_hash=city_pack_hash,
        scenario_hash=scenario_hash,
        sim_minutes=sim_minutes,
        tick_minutes=tick_minutes,
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
    return rows


def _read_section(body: bytes, row: TableRow) -> bytes:
    kind, offset, length, raw_length, crc = row
    name = SectionKind(kind).name
    if offset % 8 or offset + length > len(body):
        raise TapeError(f"{name} section lies outside the file or is misaligned")
    compressed = body[offset : offset + length]
    if zlib.crc32(compressed) != crc:
        raise TapeError(f"{name} section crc32 mismatch")
    if length == 0:
        if raw_length:
            raise TapeError(f"{name} section is empty but claims {raw_length} raw bytes")
        return b""
    raw = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=raw_length)
    if len(raw) != raw_length:
        raise TapeError(f"{name} section decompressed to {len(raw)} bytes, expected {raw_length}")
    return raw
