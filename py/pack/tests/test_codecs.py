import math

import numpy as np
import pytest
import zstandard
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from twl_pack import codecs
from twl_pack.format import DEFAULT_TILE_LEVEL, ZSTD_LEVEL, PackError

U16 = hnp.arrays(np.uint16, st.integers(0, 64))
U32 = hnp.arrays(np.uint32, st.integers(0, 64))
FAST = settings(max_examples=60, deadline=None)


def test_morton_hand_values():
    qx = np.array([1, 0, 0xFFFF, 0, 0xFFFF], dtype=np.uint16)
    qy = np.array([0, 1, 0, 0xFFFF, 0xFFFF], dtype=np.uint16)
    code = codecs.morton_encode(qx, qy)
    assert code.dtype == np.uint32
    assert code.tolist() == [1, 2, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF]


def test_morton_decode_dtypes_and_values():
    qx, qy = codecs.morton_decode(np.array([0xAAAAAAAA, 0x55555555], dtype=np.uint32))
    assert qx.dtype == qy.dtype == np.uint16
    assert qx.tolist() == [0, 0xFFFF]
    assert qy.tolist() == [0xFFFF, 0]


@FAST
@given(U16, st.data())
def test_morton_round_trip(qx, data):
    qy = data.draw(hnp.arrays(np.uint16, qx.shape))
    dx, dy = codecs.morton_decode(codecs.morton_encode(qx, qy))
    assert np.array_equal(dx, qx)
    assert np.array_equal(dy, qy)


def test_planes_width_2_hand_bytes():
    assert codecs.to_planes(np.array([0x0102, 0x0304], dtype=np.uint16), 2) == bytes(
        [0x02, 0x04, 0x01, 0x03]
    )


def test_planes_width_4_hand_bytes():
    out = codecs.to_planes(np.array([0x04030201, 0x08070605], dtype=np.uint32), 4)
    assert out == bytes([1, 5, 2, 6, 3, 7, 4, 8])


def test_from_planes_hand_values_and_dtypes():
    v2 = codecs.from_planes(bytes([0x02, 0x04, 0x01, 0x03]), 2, 2)
    assert v2.dtype == np.dtype("<u2")
    assert v2.tolist() == [0x0102, 0x0304]
    v4 = codecs.from_planes(bytes([1, 5, 2, 6, 3, 7, 4, 8]), 2, 4)
    assert v4.dtype == np.dtype("<u4")
    assert v4.tolist() == [0x04030201, 0x08070605]


@FAST
@given(U16)
def test_planes_round_trip_width_2(values):
    out = codecs.from_planes(codecs.to_planes(values, 2), len(values), 2)
    assert np.array_equal(out, values)


@FAST
@given(U32)
def test_planes_round_trip_width_4(values):
    out = codecs.from_planes(codecs.to_planes(values, 4), len(values), 4)
    assert np.array_equal(out, values)


def test_road_wrap_hand_values():
    # deltas: 65535 -> wrap -1 -> zigzag 1; -65535 -> wrap 1 -> zigzag 2;
    # 40000 -> wrap -25536 -> zigzag 51071
    z = codecs.delta_wrap_encode(np.array([65535, 0, 40000]))
    assert z.dtype == np.uint16
    assert z.tolist() == [1, 2, 51071]


def test_road_wrap_decode_hand_values():
    q = codecs.delta_wrap_decode(np.array([1, 2, 51071], dtype=np.uint16))
    assert q.dtype == np.uint16
    assert q.tolist() == [65535, 0, 40000]


def test_road_wrap_small_steps_stay_small():
    assert codecs.delta_wrap_encode(np.array([10, 12, 11])).tolist() == [20, 4, 1]


def test_road_wrap_extremes():
    # 32768 as the first delta wraps to -32768, zigzag 65535
    assert codecs.delta_wrap_encode(np.array([32768])).tolist() == [65535]
    assert codecs.delta_wrap_encode(np.array([32767])).tolist() == [65534]


@FAST
@given(U16)
def test_road_wrap_round_trip(q):
    assert np.array_equal(codecs.delta_wrap_decode(codecs.delta_wrap_encode(q)), q)


@pytest.mark.parametrize(
    ("metres", "code"), [(0.0, 0), (4.0, 24), (100.0, 120), (451.5625, 255), (1.0, 12)]
)
def test_size_encode_hand_values(metres, code):
    out, clipped = codecs.size_encode(np.array([metres]))
    assert out.dtype == np.uint8
    assert out.tolist() == [code]
    assert clipped == 0


def test_size_encode_clamps_and_counts():
    out, clipped = codecs.size_encode(np.array([500.0, 4.0, 451.6, 1000.0]))
    assert out.tolist() == [255, 24, 255, 255]
    assert clipped == 2


def test_size_encode_exactly_at_limit_is_not_clipped():
    # sqrt(451.5625) * 12 = 255 exactly
    _, clipped = codecs.size_encode(np.array([451.5625]))
    assert clipped == 0


def test_size_decode_hand_values():
    out = codecs.size_decode(np.array([255, 24, 0], dtype=np.uint8))
    assert out.dtype == np.float64
    assert out.tolist() == [451.5625, 4.0, 0.0]


@given(st.integers(0, 255))
def test_size_code_round_trip(code):
    out, clipped = codecs.size_encode(codecs.size_decode(np.array([code], dtype=np.uint8)))
    assert out.tolist() == [code]
    assert clipped == 0


def test_rotation_pi_and_zero_share_a_code():
    # mod(pi, pi) = 0, so a box turned by pi is the box at 0
    out = codecs.rotation_encode(np.array([0.0, math.pi, 2 * math.pi]))
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 0]


def test_rotation_negative_wraps_into_range():
    # -pi/4 -> 3pi/4 -> 191.25; pi/4 -> 63.75; pi/3 -> 85
    out = codecs.rotation_encode(np.array([-math.pi / 4, math.pi / 4, math.pi / 3]))
    assert out.tolist() == [191, 64, 85]


def test_rotation_large_negative_stays_in_range():
    out = codecs.rotation_encode(np.array([-100.0, 100.0, -1e-18]))
    assert out.dtype == np.uint8
    assert out.max() <= 255


def test_rotation_decode_hand_values():
    out = codecs.rotation_decode(np.array([0, 255], dtype=np.uint8))
    assert out[0] == 0.0
    assert out[1] == pytest.approx(math.pi)


@given(st.integers(0, 254))
def test_rotation_code_round_trip(code):
    back = codecs.rotation_encode(codecs.rotation_decode(np.array([code], dtype=np.uint8)))
    assert back.tolist() == [code]


def test_floor_z():
    assert codecs.floor_z(np.array([0, 1, 255]))[0] == pytest.approx(1.6)
    assert codecs.floor_z(np.array([0, 1, 255])).tolist() == pytest.approx([1.6, 4.8, 817.6])
    assert codecs.floor_z(0) == pytest.approx(1.6)


def test_quantise_hand_values():
    q = codecs.quantise(np.array([0.0, 5.0, 10.0]), 0.0, 10.0)
    assert q.dtype == np.uint16
    assert q.tolist() == [0, 32768, 65535]  # 32767.5 rounds to even


def test_quantise_negative_range():
    assert codecs.quantise(np.array([-5.0, 5.0]), -5.0, 5.0).tolist() == [0, 65535]


@pytest.mark.parametrize("values", [[-0.001], [10.001], [0.0, 11.0]])
def test_quantise_refuses_values_outside_range(values):
    with pytest.raises(PackError):
        codecs.quantise(np.array(values), 0.0, 10.0)


@pytest.mark.parametrize(("low", "high"), [(1.0, 1.0), (2.0, 1.0)])
def test_quantise_refuses_empty_range(low, high):
    with pytest.raises(PackError):
        codecs.quantise(np.array([1.0]), low, high)


def test_dequantise_hand_values():
    out = codecs.dequantise(np.array([0, 65535], dtype=np.uint16), -5.0, 5.0)
    assert out.dtype == np.float64
    assert out.tolist() == [-5.0, 5.0]


@FAST
@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=32),
    st.floats(-1e5, 1e5),
    st.floats(0.5, 1e5),
)
def test_quantise_error_is_half_a_step(unit, low, span):
    high = low + span
    values = np.clip(low + np.array(unit) * span, low, high)
    back = codecs.dequantise(codecs.quantise(values, low, high), low, high)
    assert np.all(np.abs(back - values) <= span / 65535 / 2 * (1 + 1e-9) + 1e-9)


def test_tile_starts_level_0_is_total_count():
    code = np.array([0, 5, 0xFFFFFFFF], dtype=np.uint32)
    out = codecs.tile_starts(code, 0)
    assert out.dtype == np.uint32
    assert out.tolist() == [0, 3]


CODES = np.array([0x00000001, 0x40000000, 0x40000005, 0xC0000000], dtype=np.uint32)


def test_tile_starts_level_1_hand_values():
    # tiles (code >> 30): 0, 1, 1, 3; tile 2 is empty
    assert codecs.tile_starts(CODES, 1).tolist() == [0, 1, 3, 3, 4]


def test_tile_starts_level_2_hand_values_with_empty_tiles():
    # tiles (code >> 28): 0, 4, 4, 12
    expected = [0] + [1] * 4 + [3] * 8 + [4] * 4
    assert len(expected) == 17
    assert codecs.tile_starts(CODES, 2).tolist() == expected


def test_tile_starts_coarser_level_reads_every_fourth_entry():
    fine = codecs.tile_starts(CODES, 2)
    assert codecs.tile_starts(CODES, 1).tolist() == fine[::4].tolist()
    assert codecs.tile_starts(CODES, 0).tolist() == fine[::16].tolist()


def test_tile_starts_empty_input():
    assert codecs.tile_starts(np.zeros(0, dtype=np.uint32), 2).tolist() == [0] * 17


@FAST
@given(st.lists(st.integers(0, 2**32 - 1), max_size=40), st.integers(0, 4))
def test_tile_starts_match_brute_force(codes, level):
    code = np.array(sorted(codes), dtype=np.uint32)
    out = codecs.tile_starts(code, level)
    tile = code.astype(np.uint64) >> (32 - 2 * level)
    expected = [int((tile < i).sum()) for i in range(4**level + 1)]
    assert out.tolist() == expected


@pytest.mark.parametrize(("size", "padded"), [(0, 0), (1, 8), (7, 8), (8, 8), (9, 16)])
def test_pad_zero_fills_to_multiple_of_eight(size, padded):
    out = codecs.pad(b"\x07" * size)
    assert len(out) == padded
    assert out[:size] == b"\x07" * size
    assert out[size:] == bytes(padded - size)


def test_compress_is_one_frame_with_content_size_and_no_checksum():
    raw = bytes(range(256)) * 8
    frame = codecs.compress(raw)
    params = zstandard.get_frame_parameters(frame)
    assert params.content_size == len(raw)
    assert not params.has_checksum
    assert zstandard.frame_content_size(frame) == len(raw)
    assert codecs.decompress(frame) == raw


def test_compress_matches_direct_level_19_call():
    raw = b"twinlight" * 500
    direct = zstandard.ZstdCompressor(
        level=19, write_checksum=False, write_content_size=True, threads=0
    ).compress(raw)
    assert codecs.compress(raw) == direct


def test_decompress_refuses_frame_without_content_size():
    frame = zstandard.ZstdCompressor(level=3, write_content_size=False).compress(b"x" * 100)
    assert zstandard.frame_content_size(frame) == -1
    with pytest.raises(PackError):
        codecs.decompress(frame)


@pytest.mark.parametrize("data", [b"", b"not zstd at all", b"\x28\xb5\x2f\xfd\x00"])
def test_decompress_refuses_garbage(data):
    with pytest.raises(PackError):
        codecs.decompress(data)


def test_tile_starts_refuses_codes_out_of_order():
    with pytest.raises(PackError, match="ascend"):
        codecs.tile_starts(np.array([2, 1]), 1)


def test_decompress_refuses_bytes_after_the_frame():
    with pytest.raises(PackError):
        codecs.decompress(codecs.compress(b"pack") + b"trailing")


def test_compression_level_and_default_tile_level_are_pinned():
    # Small inputs compress alike at 18 and 19, and every test names its tile level, so pin both:
    # the level changes `sizes` and so `pack_hash`, the tile level changes the table.
    assert ZSTD_LEVEL == 19
    assert DEFAULT_TILE_LEVEL == 6
