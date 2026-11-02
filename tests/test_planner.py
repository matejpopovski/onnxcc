"""Property tests for the arena planner, checked only by memcheck.check_intervals."""

from __future__ import annotations

import random

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from onnxcc.memcheck import PlanViolation, check_intervals
from onnxcc.planner import ORDERINGS, Buffer, Plan, make_plan, naive, peak_live


def lifetimes(buffers: list[Buffer]) -> dict[str, tuple[int, int, int]]:
    return {b.key: (b.size, b.first, b.last) for b in buffers}


@st.composite
def buffer_sets(draw: st.DrawFn, max_size: int = 60) -> list[Buffer]:
    count = draw(st.integers(1, max_size))
    buffers = []
    for index in range(count):
        first = draw(st.integers(0, 40))
        last = first + draw(st.integers(0, 15))
        size = draw(st.integers(1, 5000))
        buffers.append(Buffer(f"b{index}", size, first, last))
    return buffers


@pytest.mark.parametrize("strategy", ["best", "naive", *ORDERINGS])
@settings(max_examples=300)
@given(buffers=buffer_sets())
def test_no_two_live_buffers_overlap(strategy: str, buffers: list[Buffer]) -> None:
    plan = make_plan(buffers, strategy)
    check_intervals(lifetimes(buffers), plan.offsets, plan.size)


@settings(max_examples=300)
@given(buffers=buffer_sets())
def test_plan_lies_between_lower_bound_and_naive(buffers: list[Buffer]) -> None:
    plan = make_plan(buffers)
    assert peak_live(buffers) <= plan.size <= naive(buffers).size


@given(buffers=buffer_sets(max_size=30))
def test_planning_is_deterministic_and_order_independent(buffers: list[Buffer]) -> None:
    shuffled = list(buffers)
    random.Random(0).shuffle(shuffled)
    assert make_plan(buffers).size == make_plan(shuffled).size


@settings(max_examples=300)
@given(buffers=buffer_sets(max_size=40), seed=st.integers(0, 2**32 - 1))
def test_checker_rejects_a_corrupted_plan(buffers: list[Buffer], seed: int) -> None:
    """A checker that accepts everything would make the tests above meaningless."""
    plan = make_plan(buffers)
    conflicting = [
        (left, right)
        for index, left in enumerate(buffers)
        for right in buffers[index + 1 :]
        if left.conflicts(right)
    ]
    assume(conflicting)
    left, right = random.Random(seed).choice(conflicting)
    corrupted = dict(plan.offsets)
    corrupted[left.key] = corrupted[right.key]
    size = max(corrupted[b.key] + b.size for b in buffers)
    with pytest.raises(PlanViolation):
        check_intervals(lifetimes(buffers), corrupted, size)


def test_chain_needs_only_two_buffers_at_a_time() -> None:
    buffers = [Buffer(f"b{i}", 100, i, i + 1) for i in range(20)]
    assert make_plan(buffers).size == 200


def test_empty_plan() -> None:
    assert make_plan([]) == Plan({}, 0, "greedy-by-size")


def test_checker_rejects_out_of_bounds_and_missing_buffers() -> None:
    with pytest.raises(PlanViolation):
        check_intervals({"a": (10, 0, 1)}, {"a": 5}, 10)
    with pytest.raises(PlanViolation):
        check_intervals({"a": (10, 0, 1)}, {}, 10)
