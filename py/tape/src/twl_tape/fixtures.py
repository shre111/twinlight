"""Checked-in tape fixtures and their probe files (`python -m twl_tape.fixtures [--check]`).

`packages/tape` decodes `fixtures/tape/v1/*.twlt` and must reproduce every value in the
`.probes.json` next to it (ARCHITECTURE section 4). The hand-built tapes use integer and
exactly representable float arithmetic only, so no library upgrade can move a byte.
"""

import argparse
import dataclasses
import json
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_tape.format import (
    FLAG_COMPLETE,
    NEVER,
    TAPE_VERSION,
    AggregatesSection,
    ChatterSection,
    DeliveriesSection,
    ResidentsSection,
    SectionKind,
    StoresSection,
    Tape,
    TapeHeader,
)
from twl_tape.reader import read_footer, read_section_table, read_tape
from twl_tape.writer import tape_hash, write_tape

FIXTURE_DIR = Path(__file__).resolve().parents[4] / "fixtures" / "tape" / f"v{TAPE_VERSION}"
TAPE_SUFFIX = ".twlt"
PROBES_SUFFIX = ".probes.json"
PROBE_ROWS = 8
SECTION_NAMES = ("residents", "chatter", "deliveries", "stores", "aggregates")
DTYPE_NAMES = {"u1": "u8", "u2": "u16", "u4": "u32", "f4": "f32"}

U16_MAX = 0xFFFF
U32_BELOW_NEVER = 0xFFFF_FFFE
U32_EXTREMES = (U32_BELOW_NEVER, NEVER)
KNUTH_MULTIPLIER = 2_654_435_761
# Largest finite float32, smallest subnormal and -0.0, from their bit patterns.
F32_EXTREMES = tuple(
    np.array([0x7F7F_FFFF, 0x0000_0001, 0x8000_0000], dtype="<u4").view("<f4").tolist()
)


def small_tape() -> Tape:
    """Counts chosen so that columns need padding; three ticks, coherent by hand."""
    u8, u16, u32, f32 = np.uint8, np.uint16, np.uint32, np.float32
    return Tape(
        header=TapeHeader(
            run_id=bytes(range(1, 17)),
            seed=7,
            engine_version="fixture-small",
            city_pack_hash=bytes(range(32)),
            scenario_hash=bytes(range(32, 64)),
            sim_minutes=45,
            tick_minutes=15,
            residents=7,
            stores=2,
            chatter_rows=5,
            delivery_rows=3,
            stockout_pairs=3,
            flags=FLAG_COMPLETE,
        ),
        residents=ResidentsSection(
            aware_at=np.array([0, 3, 14, NEVER, 29, 31, 44], dtype=u32),
            first_order_at=np.array([5, NEVER, 20, NEVER, 34, NEVER, NEVER], dtype=u32),
            order_count=np.array([1, 0, 1, 0, 1, 0, 0], dtype=u16),
            heard_from=np.array([NEVER, 0, 1, NEVER, 2, NEVER, NEVER], dtype=u32),
            first_source=np.array([1, 2, 2, 0, 2, 3, 1], dtype=u8),
            outcome_flags=np.array([0, 5, 3, 0, 0, 1, 0], dtype=u8),
        ),
        chatter=ChatterSection(
            from_=np.array([0, 1, 0, 2, 4], dtype=u32),
            to=np.array([1, 2, 2, 4, 1], dtype=u32),
            at=np.array([3, 14, 18, 29, 40], dtype=u32),
            kind=np.array([1, 1, 2, 1, 2], dtype=u8),
        ),
        deliveries=DeliveriesSection(
            store=np.array([0, 0, 1], dtype=u16),
            resident=np.array([0, 2, 4], dtype=u32),
            ordered_at=np.array([5, 20, 34], dtype=u32),
            picked_at=np.array([9, 26, NEVER], dtype=u32),
            delivered_at=np.array([41, NEVER, NEVER], dtype=u32),
            outcome=np.array([1, 2, 0], dtype=u8),
        ),
        stores=StoresSection(
            stock=np.array([0.0, 12.5, 0.0, 0.0, 1.5e-3, 0.1], dtype=f32),
            orders=np.array([1, 1, 0, 0, 0, 1], dtype=u16),
            stockout_offsets=np.array([0, 2, 3], dtype=u32),
            stockout_intervals=np.array([10, 20, 35, 45, 12, 28], dtype=u32),
        ),
        aggregates=AggregatesSection(
            aware=np.array([3, 4, 6], dtype=u32),
            adopted=np.array([1, 2, 3], dtype=u32),
            orders=np.array([1, 1, 1], dtype=u32),
            stores_out=np.array([2, 0, 1], dtype=u16),
            rain_intensity=np.array([0.0, 0.3, 2.5e-3], dtype=f32),
        ),
    )


def empty_tape() -> Tape:
    """No residents, stores or events and a single tick: every column but one is empty."""
    return Tape(
        header=TapeHeader(
            run_id=bytes(16),
            seed=0,
            engine_version="",
            city_pack_hash=bytes(32),
            scenario_hash=bytes(32),
            sim_minutes=15,
            tick_minutes=15,
            residents=0,
            stores=0,
            chatter_rows=0,
            delivery_rows=0,
            stockout_pairs=0,
            flags=FLAG_COMPLETE,
        ),
        residents=ResidentsSection(
            aware_at=np.zeros(0, np.uint32),
            first_order_at=np.zeros(0, np.uint32),
            order_count=np.zeros(0, np.uint16),
            heard_from=np.zeros(0, np.uint32),
            first_source=np.zeros(0, np.uint8),
            outcome_flags=np.zeros(0, np.uint8),
        ),
        chatter=ChatterSection(
            from_=np.zeros(0, np.uint32),
            to=np.zeros(0, np.uint32),
            at=np.zeros(0, np.uint32),
            kind=np.zeros(0, np.uint8),
        ),
        deliveries=DeliveriesSection(
            store=np.zeros(0, np.uint16),
            resident=np.zeros(0, np.uint32),
            ordered_at=np.zeros(0, np.uint32),
            picked_at=np.zeros(0, np.uint32),
            delivered_at=np.zeros(0, np.uint32),
            outcome=np.zeros(0, np.uint8),
        ),
        stores=StoresSection(
            stock=np.zeros(0, np.float32),
            orders=np.zeros(0, np.uint16),
            stockout_offsets=np.zeros(1, np.uint32),
            stockout_intervals=np.zeros(0, np.uint32),
        ),
        aggregates=AggregatesSection(
            aware=np.zeros(1, np.uint32),
            adopted=np.zeros(1, np.uint32),
            orders=np.zeros(1, np.uint32),
            stores_out=np.zeros(1, np.uint16),
            rain_intensity=np.zeros(1, np.float32),
        ),
    )


def limits_tape() -> Tape:
    """The extremes of every header field and column dtype. Times deliberately run past
    `sim_minutes`, so a decoder must not range-check values against the header."""
    residents, stores, ticks, chatter_rows, delivery_rows = 300, 3, 67, 301, 257
    series = stores * ticks
    intervals = [0, 1, 2, 15, 16, 1000, 1004, 1005, 7, 8, 9, 1000, *U32_EXTREMES]
    return Tape(
        header=TapeHeader(
            run_id=bytes([0xFF, 0x00, *range(0x10, 0x1E)]),
            seed=2**64 - 1,
            engine_version="fixture-limits-0123456789-abcdef",
            city_pack_hash=bytes([0x00, 0xFF, *range(0x80, 0x9E)]),
            scenario_hash=bytes([*range(0xE1, 0x100), 0x00]),
            sim_minutes=ticks * 15,
            tick_minutes=15,
            residents=residents,
            stores=stores,
            chatter_rows=chatter_rows,
            delivery_rows=delivery_rows,
            stockout_pairs=len(intervals) // 2,
            flags=FLAG_COMPLETE,
        ),
        residents=ResidentsSection(
            aware_at=_planted(_scrambled(residents, 2**32, 0), np.uint32, U32_EXTREMES),
            first_order_at=_planted(_scrambled(residents, 2**32, 1), np.uint32, U32_EXTREMES),
            order_count=_planted(_scrambled(residents, 2**16, 2), np.uint16, (U16_MAX,)),
            heard_from=_planted(_scrambled(residents, residents, 3), np.uint32, U32_EXTREMES),
            first_source=(np.arange(residents) % 4).astype(np.uint8),
            outcome_flags=(np.arange(residents) % 8).astype(np.uint8),
        ),
        chatter=ChatterSection(
            from_=_planted(_scrambled(chatter_rows, residents, 4), np.uint32, U32_EXTREMES),
            to=_planted(_scrambled(chatter_rows, residents, 5), np.uint32, U32_EXTREMES),
            at=_rising(chatter_rows, 3),
            kind=(1 + np.arange(chatter_rows) % 2).astype(np.uint8),
        ),
        deliveries=DeliveriesSection(
            store=_planted(_scrambled(delivery_rows, stores, 6), np.uint16, (U16_MAX,)),
            resident=_planted(_scrambled(delivery_rows, residents, 7), np.uint32, U32_EXTREMES),
            ordered_at=_rising(delivery_rows, 5),
            picked_at=_planted(_scrambled(delivery_rows, 2**32, 8), np.uint32, U32_EXTREMES),
            delivered_at=_planted(_scrambled(delivery_rows, 2**32, 9), np.uint32, U32_EXTREMES),
            outcome=(np.arange(delivery_rows) % 4).astype(np.uint8),
        ),
        stores=StoresSection(
            stock=_planted(_dyadic(series, 10), np.float32, F32_EXTREMES),
            orders=_planted(_scrambled(series, 2**16, 11), np.uint16, (U16_MAX,)),
            stockout_offsets=np.array([0, 4, 4, 7], dtype=np.uint32),
            stockout_intervals=np.array(intervals, dtype=np.uint32),
        ),
        aggregates=AggregatesSection(
            aware=_rising(ticks, 4),
            adopted=_rising(ticks, 2),
            orders=_planted(_scrambled(ticks, 2**32, 12), np.uint32, U32_EXTREMES),
            stores_out=_planted(_scrambled(ticks, stores + 1, 13), np.uint16, (U16_MAX,)),
            rain_intensity=_planted(_dyadic(ticks, 14), np.float32, F32_EXTREMES),
        ),
    )


HAND_BUILT: dict[str, Callable[[], Tape]] = {
    "small": small_tape,
    "empty": empty_tape,
    "limits": limits_tape,
}


def _scrambled(count: int, modulus: int, salt: int) -> NDArray[np.uint64]:
    """Knuth's multiplicative hash of `salt + 0..count-1`: varied bytes from integers alone."""
    index = np.arange(count, dtype=np.uint64) + np.uint64(salt * 1_000_003)
    return index * np.uint64(KNUTH_MULTIPLIER) % np.uint64(modulus)


def _dyadic(count: int, salt: int) -> NDArray[np.float64]:
    """Signed multiples of 1/1024 below 2**13: 24-bit mantissas that float32 holds exactly."""
    return (_scrambled(count, 2**24, salt).astype(np.int64) - 2**23) / 1024


def _rising(count: int, step: int) -> NDArray[np.uint32]:
    """Non-decreasing with ties, ending at the two largest u32 values."""
    values = np.arange(count, dtype=np.uint32) * np.uint32(step) // np.uint32(2)
    values[-2:] = U32_EXTREMES
    return values


def _planted(values: NDArray[Any], dtype: type, extremes: Sequence[float]) -> NDArray[Any]:
    """`values` as `dtype` with the extremes written at evenly spaced rows."""
    planted = values.astype(dtype)
    spacing = planted.shape[0] // (len(extremes) + 1)
    if spacing == 0:
        raise ValueError(f"{planted.shape[0]} rows cannot hold {len(extremes)} extremes")
    for place, extreme in enumerate(extremes, start=1):
        planted[place * spacing] = extreme
    return planted


def build_fixtures() -> dict[str, bytes]:
    """The file bytes of every hand-built fixture, by name."""
    return {name: write_tape(build()) for name, build in HAND_BUILT.items()}


def probes(data: bytes) -> dict[str, Any]:
    """The probe document of a tape: what a decoder in any language must reproduce."""
    tape = read_tape(data)
    header = tape.header
    total_length, digest = read_footer(data)
    return {
        "file": {"bytes": len(data), "blake3": tape_hash(data)},
        "footer": {"total_length": total_length, "blake3": digest.hex()},
        "header": {
            "version": TAPE_VERSION,
            "flags": header.flags,
            "run_id": header.run_id.hex(),
            "seed": str(header.seed),
            "engine_version": header.engine_version,
            "city_pack_hash": header.city_pack_hash.hex(),
            "scenario_hash": header.scenario_hash.hex(),
            "sim_minutes": header.sim_minutes,
            "tick_minutes": header.tick_minutes,
            "residents": header.residents,
            "stores": header.stores,
            "chatter_rows": header.chatter_rows,
            "delivery_rows": header.delivery_rows,
            "stockout_pairs": header.stockout_pairs,
        },
        "sections": [
            {
                "kind": kind,
                "name": SectionKind(kind).name.lower(),
                "offset": offset,
                "length": length,
                "raw_length": raw_length,
                "crc32": crc32,
            }
            for kind, offset, length, raw_length, crc32 in read_section_table(data)
        ],
        "columns": {name: _section_probes(getattr(tape, name)) for name in SECTION_NAMES},
    }


def _section_probes(section: Any) -> dict[str, dict[str, Any]]:
    return {
        field.name.rstrip("_"): _column_probes(getattr(section, field.name))
        for field in dataclasses.fields(section)
    }


def _column_probes(values: NDArray[Any]) -> dict[str, Any]:
    code = f"{values.dtype.kind}{values.dtype.itemsize}"
    length = int(values.shape[0])
    column: dict[str, Any] = {
        "dtype": DTYPE_NAMES[code],
        "length": length,
        "crc32": zlib.crc32(values.astype(f"<{code}").tobytes()),
    }
    if values.dtype.kind == "u":
        column["sum"] = str(int(values.sum(dtype=np.uint64)))
    column["head"] = values[:PROBE_ROWS].tolist()
    column["tail"] = values[max(0, length - PROBE_ROWS) :].tolist()
    return column


def probes_text(data: bytes) -> str:
    """The exact contents of a probe file; NaN and infinity are refused as they are not JSON."""
    return json.dumps(probes(data), indent=2, allow_nan=False) + "\n"


def write_fixture(name: str, data: bytes, directory: Path = FIXTURE_DIR) -> None:
    """Write `<name>.twlt` and the `<name>.probes.json` derived from it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}{TAPE_SUFFIX}").write_bytes(data)
    _probes_path(directory / f"{name}{TAPE_SUFFIX}").write_bytes(probes_text(data).encode())


def write_fixtures(directory: Path = FIXTURE_DIR) -> list[str]:
    """Write the hand-built fixtures and refresh the probes of every other tape present."""
    built = build_fixtures()
    for name, data in built.items():
        write_fixture(name, data, directory)
    others = [
        path for path in _tapes(directory) if path.name.removesuffix(TAPE_SUFFIX) not in built
    ]
    for path in others:
        write_fixture(path.name.removesuffix(TAPE_SUFFIX), path.read_bytes(), directory)
    return [*built, *(path.name.removesuffix(TAPE_SUFFIX) for path in others)]


def check_fixtures(directory: Path = FIXTURE_DIR) -> list[str]:
    """Everything that is out of date in the checked-in fixtures; empty when they are current."""
    problems: list[str] = []
    for name, data in build_fixtures().items():
        path = directory / f"{name}{TAPE_SUFFIX}"
        if not path.is_file():
            problems.append(f"{path.name} is missing")
        elif path.read_bytes() != data:
            problems.append(f"{path.name} differs from a fresh build")
    for path in _tapes(directory):
        problems.extend(_probe_problems(path))
    for path in sorted(directory.glob(f"*{PROBES_SUFFIX}")):
        tape_name = path.name.removesuffix(PROBES_SUFFIX) + TAPE_SUFFIX
        if not (directory / tape_name).is_file():
            problems.append(f"{path.name} has no {tape_name}")
    return problems


def _probe_problems(tape_path: Path) -> list[str]:
    probes_path = _probes_path(tape_path)
    try:
        expected = probes_text(tape_path.read_bytes())
    except ValueError as error:
        return [f"{tape_path.name} cannot be probed: {error}"]
    if not probes_path.is_file():
        return [f"{tape_path.name} has no {probes_path.name}"]
    if probes_path.read_bytes().decode() != expected:
        return [f"{probes_path.name} is stale: it is not the probes of {tape_path.name}"]
    return []


def _tapes(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"*{TAPE_SUFFIX}"))


def _probes_path(tape_path: Path) -> Path:
    return tape_path.with_name(tape_path.name.removesuffix(TAPE_SUFFIX) + PROBES_SUFFIX)


def main(argv: Sequence[str] | None = None, directory: Path = FIXTURE_DIR) -> None:
    parser = argparse.ArgumentParser(prog="python -m twl_tape.fixtures")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the checked-in fixtures and probes instead of writing them",
    )
    args = parser.parse_args(argv)
    if not args.check:
        names = write_fixtures(directory)
        print(f"wrote {len(names)} tape fixtures to {directory}: {', '.join(names)}")
        return
    problems = check_fixtures(directory)
    if problems:
        lines = [f"tape fixtures in {directory} are not current:", *(f"  {p}" for p in problems)]
        lines.append(
            "regenerate with `uv run python -m twl_tape.fixtures` (engine tapes: `just goldens`)"
        )
        raise SystemExit("\n".join(lines))
    print(f"{len(_tapes(directory))} tape fixtures in {directory} are current")


if __name__ == "__main__":
    main()
