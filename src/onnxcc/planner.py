"""Offline placement of tensors in an arena: dynamic storage allocation with known lifetimes.

Minimizing the arena is NP-hard, but several cheap orderings get close, and the answer can be
checked against a lower bound (the most bytes ever live at once). All strategies run and the best
result is kept, because planning happens once on a developer machine, not on the device.

This module deliberately knows nothing about graphs. Its only input is a list of intervals, which
is what lets the property tests fuzz it directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Buffer:
    key: str
    size: int
    first: int  # step that writes it
    last: int  # last step that reads it, inclusive

    def conflicts(self, other: Buffer) -> bool:
        return self.first <= other.last and other.first <= self.last


@dataclass
class Plan:
    offsets: dict[str, int]
    size: int
    strategy: str


def _first_fit(ordered: Iterable[Buffer]) -> dict[str, int]:
    """Give each buffer the lowest gap among already-placed buffers it conflicts with."""
    placed: list[tuple[int, Buffer]] = []
    offsets: dict[str, int] = {}
    for buffer in ordered:
        offset = 0
        busy = sorted(
            (item for item in placed if item[1].conflicts(buffer)), key=lambda item: item[0]
        )
        for start, other in busy:
            if offset + buffer.size <= start:
                break
            offset = max(offset, start + other.size)
        offsets[buffer.key] = offset
        placed.append((offset, buffer))
    return offsets


def _by_size(buffers: Sequence[Buffer]) -> list[Buffer]:
    return sorted(buffers, key=lambda b: (-b.size, b.first, b.key))


def _by_start(buffers: Sequence[Buffer]) -> list[Buffer]:
    return sorted(buffers, key=lambda b: (b.first, -b.size, b.key))


def _by_lifetime(buffers: Sequence[Buffer]) -> list[Buffer]:
    return sorted(buffers, key=lambda b: (b.first - b.last, -b.size, b.key))


def _by_area(buffers: Sequence[Buffer]) -> list[Buffer]:
    return sorted(buffers, key=lambda b: (-b.size * (b.last - b.first + 1), b.key))


def _by_conflict_load(buffers: Sequence[Buffer]) -> list[Buffer]:
    """Hardest first: buffers overlapping the most bytes of other tensors."""
    load = {b.key: sum(o.size for o in buffers if o.conflicts(b)) for b in buffers}
    return sorted(buffers, key=lambda b: (-load[b.key], -b.size, b.key))


ORDERINGS: dict[str, Callable[[Sequence[Buffer]], list[Buffer]]] = {
    "greedy-by-size": _by_size,
    "first-fit-by-start": _by_start,
    "greedy-by-lifetime": _by_lifetime,
    "greedy-by-area": _by_area,
    "greedy-by-conflict": _by_conflict_load,
}


def extent(buffers: Sequence[Buffer], offsets: dict[str, int]) -> int:
    return max((offsets[b.key] + b.size for b in buffers), default=0)


def peak_live(buffers: Sequence[Buffer]) -> int:
    """Lower bound on any arena: the largest total size alive at one step."""
    events = sorted({b.first for b in buffers})
    return max((sum(b.size for b in buffers if b.first <= t <= b.last) for t in events), default=0)


def naive(buffers: Sequence[Buffer]) -> Plan:
    offsets: dict[str, int] = {}
    cursor = 0
    for buffer in buffers:
        offsets[buffer.key] = cursor
        cursor += buffer.size
    return Plan(offsets, cursor, "naive")


def best_plan(buffers: Sequence[Buffer]) -> Plan:
    best: Plan | None = None
    for name, ordering in ORDERINGS.items():
        offsets = _first_fit(ordering(buffers))
        size = extent(buffers, offsets)
        if best is None or size < best.size:
            best = Plan(offsets, size, name)
    assert best is not None
    return best


def make_plan(buffers: Sequence[Buffer], strategy: str = "best") -> Plan:
    if strategy == "best":
        return best_plan(buffers)
    if strategy == "naive":
        return naive(buffers)
    offsets = _first_fit(ORDERINGS[strategy](buffers))
    return Plan(offsets, extent(buffers, offsets), strategy)
