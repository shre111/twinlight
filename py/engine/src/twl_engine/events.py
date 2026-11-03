"""Logistics event queue: a binary min-heap keyed by (minute, seq) (MODEL.md section 1, D-015)."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

MINUTE, SEQ, KIND, A, B = 0, 1, 2, 3, 4
FIELDS = 5
INITIAL_CAPACITY = 1024


@dataclass(frozen=True, eq=False)
class Events:
    """Events popped from the queue, in (minute, seq) order."""

    minute: NDArray[np.int64]
    seq: NDArray[np.int64]
    kind: NDArray[np.int64]
    a: NDArray[np.int64]
    b: NDArray[np.int64]

    def __len__(self) -> int:
        return self.minute.shape[0]


class EventQueue:
    def __init__(self, capacity: int = INITIAL_CAPACITY) -> None:
        self.heap = np.zeros((max(capacity, 1), FIELDS), dtype=np.int64)
        self.count = 0
        self.next_seq = 0

    def __len__(self) -> int:
        return self.count

    def push(self, minute: int, kind: int, a: int = 0, b: int = 0) -> int:
        """Queue an event; returns its seq, which breaks ties between events at one minute."""
        if minute < 0:
            raise ValueError("event minutes are never negative")
        if self.count == self.heap.shape[0]:
            self._grow()
        seq = self.next_seq
        self.next_seq += 1
        heap_push(self.heap, self.count, minute, seq, kind, a, b)
        self.count += 1
        return seq

    def peek_minute(self) -> int | None:
        return int(self.heap[0, MINUTE]) if self.count else None

    def pop_until(self, minute: int) -> Events:
        """Remove and return every event with `event.minute <= minute`, ordered by (minute, seq)."""
        out = np.empty((self.count, FIELDS), dtype=np.int64)
        popped = heap_pop_until(self.heap, self.count, minute, out)
        self.count -= popped
        taken = np.ascontiguousarray(out[:popped].T)
        return Events(
            minute=taken[MINUTE], seq=taken[SEQ], kind=taken[KIND], a=taken[A], b=taken[B]
        )

    def _grow(self) -> None:
        bigger = np.zeros((self.heap.shape[0] * 2, FIELDS), dtype=np.int64)
        bigger[: self.count] = self.heap[: self.count]
        self.heap = bigger


@njit(cache=True)
def _before(heap: NDArray[np.int64], i: int, j: int) -> bool:
    if heap[i, MINUTE] != heap[j, MINUTE]:
        return heap[i, MINUTE] < heap[j, MINUTE]
    return heap[i, SEQ] < heap[j, SEQ]


@njit(cache=True)
def _swap(heap: NDArray[np.int64], i: int, j: int) -> None:
    for f in range(FIELDS):
        tmp = heap[i, f]
        heap[i, f] = heap[j, f]
        heap[j, f] = tmp


@njit(cache=True)
def heap_push(
    heap: NDArray[np.int64], count: int, minute: int, seq: int, kind: int, a: int, b: int
) -> None:
    i = count
    heap[i, MINUTE] = minute
    heap[i, SEQ] = seq
    heap[i, KIND] = kind
    heap[i, A] = a
    heap[i, B] = b
    while i > 0:
        parent = (i - 1) // 2
        if not _before(heap, i, parent):
            break
        _swap(heap, i, parent)
        i = parent


@njit(cache=True)
def heap_pop_until(heap: NDArray[np.int64], count: int, bound: int, out: NDArray[np.int64]) -> int:
    popped = 0
    while count > 0 and heap[0, MINUTE] <= bound:
        for f in range(FIELDS):
            out[popped, f] = heap[0, f]
        popped += 1
        count -= 1
        if count > 0:
            for f in range(FIELDS):
                heap[0, f] = heap[count, f]
            _sift_down(heap, count)
    return popped


@njit(cache=True)
def _sift_down(heap: NDArray[np.int64], count: int) -> None:
    i = 0
    while True:
        left = 2 * i + 1
        if left >= count:
            return
        right = left + 1
        smallest = left
        if right < count and _before(heap, right, left):
            smallest = right
        if not _before(heap, smallest, i):
            return
        _swap(heap, i, smallest)
        i = smallest
