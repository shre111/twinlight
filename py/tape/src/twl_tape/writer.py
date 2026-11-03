import struct
import zlib

import zstandard
from blake3 import blake3

from twl_tape.format import (
    COLUMN_ALIGN,
    ENGINE_VERSION_BYTES,
    FLAG_COMPLETE,
    FOOTER_STRUCT,
    HASH_BYTES,
    HEADER_STRUCT,
    MAGIC,
    RUN_ID_BYTES,
    TABLE_ROW_STRUCT,
    TAPE_VERSION,
    ZSTD_LEVEL,
    SectionKind,
    Tape,
    TapeHeader,
)
from twl_tape.sections import (
    encode_aggregates,
    encode_chatter,
    encode_deliveries,
    encode_residents,
    encode_stores,
)

SECTION_COUNT = len(SectionKind)
_PACKED_HEADER = struct.calcsize(HEADER_STRUCT) + SECTION_COUNT * struct.calcsize(TABLE_ROW_STRUCT)
HEADER_SIZE = _PACKED_HEADER + (-_PACKED_HEADER % COLUMN_ALIGN)


def write_tape(tape: Tape) -> bytes:
    """The complete file bytes for a finished run (flags bit0 set)."""
    header = tape.header
    ticks = header.ticks
    raw_sections = [
        encode_residents(tape.residents),
        encode_chatter(tape.chatter),
        encode_deliveries(tape.deliveries),
        encode_stores(tape.stores, ticks),
        encode_aggregates(tape.aggregates, ticks),
        b"",
    ]
    body = bytearray(HEADER_SIZE)
    rows: list[bytes] = []
    for kind, raw in zip(SectionKind, raw_sections, strict=True):
        compressed = compress(raw)
        offset = len(body)
        body += compressed
        body += bytes(-len(body) % COLUMN_ALIGN)
        rows.append(
            struct.pack(
                TABLE_ROW_STRUCT,
                int(kind),
                offset,
                len(compressed),
                len(raw),
                zlib.crc32(compressed),
            )
        )
    body[:HEADER_SIZE] = _pack_header(header, rows)
    footer = struct.pack(FOOTER_STRUCT, len(body), blake3(bytes(body)).digest())
    return bytes(body) + footer


def tape_hash(data: bytes) -> str:
    """blake3 hex of the whole file; this is what the goldens pin."""
    return blake3(data).hexdigest()


def compress(raw: bytes) -> bytes:
    """Deterministic zstd frame; an empty section stays empty (zero bytes)."""
    if not raw:
        return b""
    compressor = zstandard.ZstdCompressor(
        level=ZSTD_LEVEL, write_checksum=False, write_content_size=True, threads=0
    )
    return compressor.compress(raw)


def _pack_header(header: TapeHeader, rows: list[bytes]) -> bytes:
    version = header.engine_version.encode("ascii")
    if len(version) > ENGINE_VERSION_BYTES:
        raise ValueError(f"engine_version longer than {ENGINE_VERSION_BYTES} bytes")
    if len(header.run_id) != RUN_ID_BYTES:
        raise ValueError(f"run_id must be {RUN_ID_BYTES} bytes")
    if len(header.city_pack_hash) != HASH_BYTES or len(header.scenario_hash) != HASH_BYTES:
        raise ValueError(f"city_pack_hash and scenario_hash must be {HASH_BYTES} bytes")
    fixed = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        TAPE_VERSION,
        header.flags | FLAG_COMPLETE,
        header.run_id,
        header.seed,
        version,
        header.city_pack_hash,
        header.scenario_hash,
        header.sim_minutes,
        header.tick_minutes,
    )
    packed = fixed + b"".join(rows)
    return packed + bytes(HEADER_SIZE - len(packed))
