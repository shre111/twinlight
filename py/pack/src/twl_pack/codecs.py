"""Pure column codecs for city pack v2 (ARCHITECTURE.md section 5)."""

from typing import Any

import numpy as np
import zstandard
from numpy.typing import ArrayLike, DTypeLike, NDArray

from twl_pack.format import COLUMN_ALIGN, FLOOR_HEIGHT_M, MAX_TILE_LEVEL, ZSTD_LEVEL, PackError

_U16_MAX = 0xFFFF
_U32_MAX = 0xFFFFFFFF
_PLANE_WIDTHS = (2, 4)
_MAX_EXACT_FLOAT = 2**53


def as_uint(
    values: ArrayLike, dtype: DTypeLike, column: str, *, low: int = 0, high: int
) -> NDArray[Any]:
    """Narrow to `dtype` after proving every value is a whole number in `low..high`."""
    array = np.asarray(values)
    kind = array.dtype.kind
    if kind not in "iuf":
        raise PackError(f"{column} must be numeric, got {array.dtype}")
    if array.size == 0:
        return array.astype(dtype)
    if kind == "f":
        if not np.isfinite(array).all():
            raise PackError(f"{column} has NaN or infinity")
        if (array != np.floor(array)).any():
            raise PackError(f"{column} must hold whole numbers")
        if np.abs(array).max() > _MAX_EXACT_FLOAT:
            raise PackError(f"{column} holds floats too large to be exact integers")
    smallest, largest = int(array.min()), int(array.max())
    if smallest < low or largest > high:
        raise PackError(f"{column} runs {smallest}..{largest}, outside {low}..{high}")
    return array.astype(dtype, copy=False)


def quantise(values: ArrayLike, low: float, high: float) -> NDArray[np.uint16]:
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise PackError(f"quantise needs finite low < high, got {low} and {high}")
    array = np.asarray(values, dtype=np.float64)
    if not ((array >= low) & (array <= high)).all():
        raise PackError(f"value outside [{low}, {high}], or NaN, cannot be quantised")
    return np.rint((array - low) / (high - low) * _U16_MAX).astype("<u2")


def dequantise(q: ArrayLike, low: float, high: float) -> NDArray[np.float64]:
    return low + np.asarray(q, dtype=np.float64) / _U16_MAX * (high - low)


def _spread_bits(v: NDArray[Any]) -> NDArray[Any]:
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    return (v | (v << 1)) & 0x55555555


def _compact_bits(v: NDArray[Any]) -> NDArray[Any]:
    v = v & 0x55555555
    v = (v | (v >> 1)) & 0x33333333
    v = (v | (v >> 2)) & 0x0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF
    return (v | (v >> 8)) & 0x0000FFFF


def morton_encode(qx: ArrayLike, qy: ArrayLike) -> NDArray[np.uint32]:
    x = as_uint(qx, np.uint64, "morton qx", high=_U16_MAX)
    y = as_uint(qy, np.uint64, "morton qy", high=_U16_MAX)
    return (_spread_bits(x) | (_spread_bits(y) << 1)).astype("<u4")


def morton_decode(code: ArrayLike) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    stored = as_uint(code, np.uint64, "morton code", high=_U32_MAX)
    return _compact_bits(stored).astype("<u2"), _compact_bits(stored >> 1).astype("<u2")


def to_planes(values: ArrayLike, width: int) -> bytes:
    _check_plane_width(width)
    stored = as_uint(values, f"<u{width}", "plane column", high=2 ** (8 * width) - 1)
    return np.ascontiguousarray(stored).view(np.uint8).reshape(-1, width).T.tobytes()


def from_planes(data: bytes, count: int, width: int) -> NDArray[Any]:
    _check_plane_width(width)
    if len(data) != count * width:
        raise PackError(f"{count} values of {width} bytes need {count * width}, got {len(data)}")
    planes = np.frombuffer(data, dtype=np.uint8).reshape(width, count)
    return np.frombuffer(planes.T.tobytes(), dtype=f"<u{width}")


def _check_plane_width(width: int) -> None:
    if width not in _PLANE_WIDTHS:
        raise PackError(f"plane width must be one of {_PLANE_WIDTHS}, got {width}")


def delta_wrap_encode(q: ArrayLike) -> NDArray[np.uint16]:
    stored = as_uint(q, np.int64, "road position", high=_U16_MAX)
    delta = np.diff(stored, prepend=0)
    wrapped = ((delta + 32768) % 65536) - 32768
    return ((wrapped * 2) ^ (wrapped >> 63)).astype("<u2")


def delta_wrap_decode(z: ArrayLike) -> NDArray[np.uint16]:
    stored = as_uint(z, np.int64, "road delta", high=_U16_MAX)
    delta = (stored >> 1) ^ -(stored & 1)
    return (np.cumsum(delta) % 65536).astype("<u2")


def size_encode(metres: ArrayLike) -> tuple[NDArray[np.uint8], int]:
    array = np.asarray(metres, dtype=np.float64)
    if not (np.isfinite(array).all() and (array >= 0).all()):
        raise PackError("a size must be finite and not negative")
    rounded = np.rint(np.sqrt(array) * 12)
    clipped = int(np.count_nonzero(rounded > 255))
    return np.minimum(rounded, 255).astype(np.uint8), clipped


def size_decode(code: ArrayLike) -> NDArray[np.float64]:
    root = np.asarray(code, dtype=np.float64) / 12
    return root * root


def rotation_encode(radians: ArrayLike) -> NDArray[np.uint8]:
    array = np.asarray(radians, dtype=np.float64)
    if not np.isfinite(array).all():
        raise PackError("a rotation must be finite")
    return np.rint(np.mod(array, np.pi) / np.pi * 255).astype(np.uint8)


def rotation_decode(code: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(code, dtype=np.float64) / 255 * np.pi


def floor_z(floor: ArrayLike) -> NDArray[np.float64]:
    return (np.asarray(floor, dtype=np.float64) + 0.5) * FLOOR_HEIGHT_M


def tile_starts(code: ArrayLike, level: int) -> NDArray[np.uint32]:
    if not 0 <= level <= MAX_TILE_LEVEL:
        raise PackError(f"tile_level must be in 0..{MAX_TILE_LEVEL}, got {level}")
    stored = as_uint(code, np.uint64, "morton code", high=_U32_MAX)
    if (stored[1:] < stored[:-1]).any():
        raise PackError("morton codes must ascend")
    first_code = np.arange(4**level + 1, dtype=np.uint64) << (32 - 2 * level)
    return np.searchsorted(stored, first_code, side="left").astype("<u4")


def pad(raw: bytes) -> bytes:
    return raw + bytes(-len(raw) % COLUMN_ALIGN)


def compress(raw: bytes) -> bytes:
    compressor = zstandard.ZstdCompressor(
        level=ZSTD_LEVEL, write_checksum=False, write_content_size=True, threads=0
    )
    return compressor.compress(raw)


def content_size(data: bytes) -> int:
    """The uncompressed size a frame declares, read from its header without decoding it."""
    try:
        size = zstandard.frame_content_size(data)
    except zstandard.ZstdError as error:
        raise PackError(f"not a zstd frame: {error}") from error
    if size < 0:
        raise PackError("zstd frame has no content size")
    return size


def decompress(data: bytes) -> bytes:
    content_size(data)
    try:
        return zstandard.ZstdDecompressor().decompress(data, allow_extra_data=False)
    except zstandard.ZstdError as error:
        raise PackError(f"bad zstd frame: {error}") from error
