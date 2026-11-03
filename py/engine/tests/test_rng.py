import numpy as np
import pytest

from twl_engine.rng import spawn_streams


def test_streams_follow_seed_sequence_children_in_order() -> None:
    streams = spawn_streams(7, ["a", "b", "c"])
    children = np.random.SeedSequence(7).spawn(3)
    for name, child in zip(["a", "b", "c"], children, strict=True):
        expected = np.random.Generator(np.random.PCG64(child)).random(4)
        np.testing.assert_array_equal(streams[name].random(4), expected)


def test_streams_preserve_name_order() -> None:
    assert list(spawn_streams(1, ["city", "residents", "stores", "ties"])) == [
        "city",
        "residents",
        "stores",
        "ties",
    ]


def test_same_seed_gives_same_draws() -> None:
    first = spawn_streams(49, ["city", "ties"])
    second = spawn_streams(49, ["city", "ties"])
    for name in ("city", "ties"):
        np.testing.assert_array_equal(first[name].random(16), second[name].random(16))


def test_streams_are_independent_of_each_other() -> None:
    streams = spawn_streams(49, ["city", "ties"])
    assert not np.array_equal(streams["city"].random(16), streams["ties"].random(16))


def test_drawing_from_one_stream_leaves_others_untouched() -> None:
    untouched = spawn_streams(3, ["a", "b"])["b"].random(8)
    streams = spawn_streams(3, ["a", "b"])
    streams["a"].random(1_000)
    np.testing.assert_array_equal(streams["b"].random(8), untouched)


def test_different_seeds_differ() -> None:
    assert not np.array_equal(
        spawn_streams(1, ["a"])["a"].random(8), spawn_streams(2, ["a"])["a"].random(8)
    )


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        spawn_streams(1, ["a", "a"])
