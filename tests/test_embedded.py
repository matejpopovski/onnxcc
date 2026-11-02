"""Cross-compilation checks. Skipped where the toolchain is absent, which includes CI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from onnxcc.compiler import compile_model
from onnxcc.toolchain import STRICT_FLAGS, freestanding_flags

from .cases_float import float_cases
from .cases_quant import quant_cases
from .support import make_model

GCC = "arm-none-eabi-gcc"
CPU = ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
needs_toolchain = pytest.mark.skipif(shutil.which(GCC) is None, reason=f"{GCC} not installed")


@needs_toolchain
@pytest.mark.parametrize("case", [*float_cases(), *quant_cases()], ids=lambda case: case.id)
def test_every_golden_model_cross_compiles_cleanly(case, tmp_path: Path) -> None:
    source = tmp_path / "m.c"
    source.write_text(compile_model(case.model(), name="m").source)
    flags = [*STRICT_FLAGS, *CPU, *freestanding_flags(GCC), "-Os"]
    subprocess.run([GCC, *flags, "-c", str(source), "-o", str(tmp_path / "m.o")], check=True)
    listing = subprocess.run(
        ["arm-none-eabi-nm", "-u", str(tmp_path / "m.o")],
        capture_output=True,
        text=True,
        check=True,
    )
    # 64-bit products and float division are inline on Cortex-M4F, so nothing is left to link.
    assert listing.stdout.split() == []


@needs_toolchain
def test_no_double_precision_reaches_the_fpu(tmp_path: Path) -> None:
    """Cortex-M4F has a single-precision FPU. A stray double would call into soft-float."""
    node = helper.make_node
    x = np.zeros((1, 10), np.float32)
    nodes = [
        node("Softmax", ["x"], ["a"]),
        node("Tanh", ["a"], ["b"]),
        node("Sigmoid", ["b"], ["y"]),
    ]
    source = tmp_path / "m.c"
    source.write_text(compile_model(make_model(nodes, {"x": x}, {}, ["y"]), name="m").source)
    flags = [*STRICT_FLAGS, "-Wfloat-conversion", *CPU, *freestanding_flags(GCC), "-Os", "-S"]
    assembly = tmp_path / "m.s"
    subprocess.run([GCC, *flags, str(source), "-o", str(assembly)], check=True)
    assert "__aeabi_d" not in assembly.read_text()
    assert ".f64" not in assembly.read_text()
