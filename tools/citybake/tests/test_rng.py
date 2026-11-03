import numpy as np

from twl_citybake.rng import Stream, stream


def test_same_seed_and_stream_give_identical_draws() -> None:
    a = stream(20260101, Stream.POPULATION).random(64)
    b = stream(20260101, Stream.POPULATION).random(64)
    assert np.array_equal(a, b)


def test_different_streams_give_different_draws() -> None:
    draws = [stream(7, which).random(16).tobytes() for which in Stream]
    assert len(set(draws)) == len(Stream)


def test_different_seeds_give_different_draws() -> None:
    a = stream(1, Stream.PERSONAS).random(16)
    b = stream(2, Stream.PERSONAS).random(16)
    assert not np.array_equal(a, b)


def test_stream_matches_spawned_seed_sequence() -> None:
    spawned = np.random.SeedSequence(99).spawn(len(Stream))
    expected = np.random.default_rng(spawned[Stream.GRAPH]).random(8)
    assert np.array_equal(stream(99, Stream.GRAPH).random(8), expected)


def test_stream_ids_are_stable() -> None:
    ids = [int(s) for s in (Stream.POPULATION, Stream.PERSONAS, Stream.GRAPH, Stream.STORES)]
    assert ids == [0, 1, 2, 3]


def test_streams_are_independent_of_draw_order() -> None:
    first = stream(5, Stream.STORES).random(4)
    stream(5, Stream.POPULATION).random(1000)
    assert np.array_equal(stream(5, Stream.STORES).random(4), first)
