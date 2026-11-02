"""Every supported operator, compiled to C and diffed against onnxruntime."""

from pathlib import Path

import pytest

from onnxcc.frontend import SUPPORTED_OPS

from .cases_float import float_cases
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


def test_every_supported_operator_has_a_golden_case() -> None:
    covered: set[str] = set()
    for case in [*FLOAT_CASES, *QUANT_CASES]:
        covered |= case.ops
    assert SUPPORTED_OPS - covered == set()
