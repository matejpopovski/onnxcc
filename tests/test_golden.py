"""Every supported operator, compiled to C and diffed against onnxruntime."""

from pathlib import Path

import pytest

from onnxcc.frontend import MAX_OPSET, MIN_OPSET, SUPPORTED_OPS

from .cases_float import float_cases, node, noise
from .cases_quant import quant_cases
from .support import Case, check_case

FLOAT_CASES = float_cases()
QUANT_CASES = quant_cases()


@pytest.mark.parametrize("case", FLOAT_CASES, ids=lambda case: case.id)
def test_float_operator(case: Case, tmp_path: Path) -> None:
    check_case(case, tmp_path)


@pytest.mark.parametrize("case", QUANT_CASES, ids=lambda case: case.id)
def test_int8_operator(case: Case, tmp_path: Path) -> None:
    check_case(case, tmp_path)


@pytest.mark.parametrize("opset", range(MIN_OPSET, MAX_OPSET + 1))
def test_a_typical_network_at_every_accepted_opset(opset: int, tmp_path: Path) -> None:
    x = noise("opset", 1, 2, 8, 8)
    pads = [1, 1, 1, 1]
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], pads=pads),
        node("BatchNormalization", ["c", "s", "b", "m", "v"], ["n"]),
        node("Relu", ["n"], ["r"]),
        node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
        node("GlobalAveragePool", ["p"], ["g"]),
        node("Flatten", ["g"], ["f"]),
        node("Gemm", ["f", "d"], ["l"]),
        node("Softmax", ["l"], ["y"]),
    ]
    statistics = {name: noise(f"opset{name}", 3) ** 2 + 0.5 for name in ("s", "b", "m", "v")}
    weights = {"w": noise("opsetw", 3, 2, 3, 3), "d": noise("opsetd", 3, 4), **statistics}
    check_case(Case(f"opset{opset}", nodes, {"x": x}, weights, opset=opset), tmp_path)


def test_every_supported_operator_has_a_golden_case() -> None:
    covered: set[str] = set()
    for case in [*FLOAT_CASES, *QUANT_CASES]:
        covered |= case.ops
    assert SUPPORTED_OPS - covered == set()
