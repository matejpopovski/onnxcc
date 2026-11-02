"""The verifier: independence, teeth, and its verdict on real programs."""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

import onnxcc.memcheck as memcheck
from onnxcc.compiler import compile_model
from onnxcc.memcheck import PlanViolation, verify

from .support import build, make_model, ort_run

node = helper.make_node


def program(lines: list[str], arena: str = "static float F[100];") -> str:
    return "\n".join([arena, "void f(void)", "{", *lines, "}"]) + "\n"


Region = tuple[str, int, int, str]


def pointer(region: Region) -> str:
    return region[0] if region[0] not in "FQ" else f"{region[0]} + {region[1]}"


def access(region: Region) -> str:
    symbol, start, end, value = region
    return f"{symbol}[{start}:{end}]:{value}"


def step(index: int, op: str, write: Region, read: Region) -> str:
    code = f"ox_{op.lower()}_f32({pointer(write)}, {pointer(read)}, 10, 0.0f, 0.0f);"
    return f"/*@ {index} {op} w={access(write)} r={access(read)} */\n    {code}"


INPUT: Region = ("in0", 0, 10, "v0")
FIRST: Region = ("F", 0, 10, "v1")
OUTPUT: Region = ("out0", 0, 10, "v3")
GOOD_FIRST = step(0, "Abs", FIRST, INPUT)


def test_verifier_imports_nothing_from_the_compiler() -> None:
    tree = ast.parse(Path(memcheck.__file__).read_text())
    imported = set()
    for item in ast.walk(tree):
        if isinstance(item, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in item.names}
        elif isinstance(item, ast.ImportFrom):
            assert item.level == 0, "relative import in memcheck"
            imported.add((item.module or "").split(".")[0])
    assert imported <= {"__future__", "dataclasses", "re"}


def test_accepts_a_valid_chain() -> None:
    second: Region = ("F", 10, 20, "v2")
    verify(program([GOOD_FIRST, step(1, "Neg", second, FIRST), step(2, "Neg", OUTPUT, second)]))


def test_rejects_read_of_clobbered_value() -> None:
    clobber: Region = ("F", 5, 15, "v2")
    source = program([GOOD_FIRST, step(1, "Neg", clobber, INPUT), step(2, "Neg", OUTPUT, FIRST)])
    with pytest.raises(PlanViolation, match="different value"):
        verify(source)


def test_rejects_read_of_an_alias_after_its_source_was_overwritten() -> None:
    source = program(
        [
            GOOD_FIRST,
            "/*@ alias v9 v1 10 10 Reshape */",
            step(1, "Neg", ("F", 0, 10, "v2"), INPUT),
            step(2, "Neg", OUTPUT, ("F", 0, 10, "v9")),
        ]
    )
    with pytest.raises(PlanViolation, match="different value"):
        verify(source)


def test_rejects_arena_overrun() -> None:
    with pytest.raises(PlanViolation, match="exceeds"):
        verify(program([step(0, "Abs", ("F", 95, 105, "v1"), INPUT)]))


def test_rejects_in_place_use_of_a_kernel_that_needs_separate_buffers() -> None:
    source = program([GOOD_FIRST, step(1, "Conv", ("F", 0, 10, "v2"), FIRST)])
    with pytest.raises(PlanViolation, match="illegal overlap"):
        verify(source)


def test_rejects_partial_overlap_even_for_elementwise_kernels() -> None:
    source = program([GOOD_FIRST, step(1, "Abs", ("F", 5, 15, "v2"), FIRST)])
    with pytest.raises(PlanViolation, match="illegal overlap"):
        verify(source)


def test_rejects_an_annotation_that_disagrees_with_the_code() -> None:
    honest = step(0, "Abs", FIRST, INPUT)
    with pytest.raises(PlanViolation, match="annotation"):
        verify(program([honest.replace("F + 0", "F + 40")]))


def test_rejects_output_that_is_not_the_first_argument() -> None:
    swapped = "/*@ 0 Abs w=F[0:10]:v1 r=F[10:20]:v0 */\n    ox_abs_f32(F + 10, F + 0, 10);"
    with pytest.raises(PlanViolation):
        verify(program([swapped]))


def test_rejects_a_value_written_twice() -> None:
    with pytest.raises(PlanViolation, match="twice"):
        verify(program([GOOD_FIRST, step(1, "Neg", ("F", 20, 30, "v1"), INPUT)]))


def test_rejects_alias_that_changes_size() -> None:
    source = program([GOOD_FIRST, "/*@ alias v9 v1 10 12 Reshape */"])
    with pytest.raises(PlanViolation, match="element count"):
        verify(source)


def test_rejects_files_without_annotations() -> None:
    with pytest.raises(PlanViolation, match="no annotated steps"):
        verify("int main(void) { return 0; }\n")


def residual_model() -> tuple:
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (1, 2, 8, 8)).astype(np.float32)
    w1 = rng.uniform(-1, 1, (2, 2, 3, 3)).astype(np.float32)
    w2 = rng.uniform(-1, 1, (2, 2, 3, 3)).astype(np.float32)
    pads = [1, 1, 1, 1]
    nodes = [
        node("Conv", ["x", "w1"], ["a"], kernel_shape=[3, 3], pads=pads),
        node("Conv", ["a", "w2"], ["b"], kernel_shape=[3, 3], pads=pads),
        node("Sigmoid", ["b"], ["c"]),
        node("Conv", ["c", "w1"], ["d"], kernel_shape=[3, 3], pads=pads),
        node("Add", ["a", "d"], ["e"]),
        node("Mul", ["e", "c"], ["y"]),
    ]
    return make_model(nodes, {"x": x}, {"w1": w1, "w2": w2}, ["y"]), {"x": x}


def _clobber(source: str, rng: random.Random) -> str | None:
    """Move a tensor on top of another one that is still needed, in code and annotations."""
    parsed = memcheck.parse(source)
    steps = parsed.steps
    candidates = []
    for position, step in enumerate(steps):
        if step.write.symbol != "F":
            continue
        for later in steps[position + 1 :]:
            for read in later.reads:
                born_before = any(
                    s.write.value == read.value and s is not step for s in steps[:position]
                )
                if (
                    read.symbol == "F"
                    and born_before
                    and step.write.end - step.write.start == read.end - read.start
                ):
                    candidates.append((step, read))
    if not candidates:
        return None
    step, read = rng.choice(candidates)
    old = f"w=F[{step.write.start}:{step.write.end}]:{step.write.value}"
    new = f"w=F[{read.start}:{read.end}]:{step.write.value}"
    pattern = re.compile(
        re.escape(old) + r"(.*?)\*/\n(\s*)(ox_\w+\()F \+ " + str(step.write.start) + r"\b"
    )
    return pattern.sub(
        lambda m: f"{new}{m.group(1)}*/\n{m.group(2)}{m.group(3)}F + {read.start}", source, count=1
    )


def test_mutated_plans_are_rejected_and_really_are_wrong(tmp_path: Path) -> None:
    model, feed = residual_model()
    result = compile_model(model, name="mutant", inplace=False)
    verify(result.source)
    expected = ort_run(model, feed)[0]
    rng = random.Random(0)
    rejected = 0
    for _ in range(5):
        mutated = _clobber(result.source, rng)
        assert mutated is not None and mutated != result.source
        with pytest.raises(PlanViolation):
            verify(mutated)
        rejected += 1
        result_copy = result
        original = result_copy.source
        result_copy.source = mutated
        directory = tmp_path / str(rejected)
        directory.mkdir()
        output = build(result_copy, directory, ["-std=c99", "-O1"]).run([feed["x"][None]])[0][0]
        result_copy.source = original
        assert not np.allclose(output, expected, rtol=1e-3, atol=1e-3)
    assert rejected == 5
