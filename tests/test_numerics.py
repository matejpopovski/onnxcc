"""Libm-free math kernels measured against double-precision numpy on a dense grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from onnxcc.compiler import compile_model
from onnxcc.errors import CompileError
from onnxcc.fixedpoint import add_multipliers, multiplier
from onnxcc.toolchain import STRICT_FLAGS

from .support import build, make_model

node = helper.make_node
F = np.float32


def evaluate(op: str, x: np.ndarray, directory: Path) -> np.ndarray:
    model = make_model([node(op, ["x"], ["y"])], {"x": x}, {}, ["y"])
    binary = build(compile_model(model, name="math"), directory, [*STRICT_FLAGS, "-O2"])
    return binary.run([x[None]])[0][0]


@pytest.mark.parametrize(
    ("op", "low", "high", "function", "limit"),
    [
        ("Exp", -80.0, 80.0, np.exp, 3e-7),
        ("Sigmoid", -20.0, 20.0, lambda v: 1.0 / (1.0 + np.exp(-v)), 3e-7),
        ("Tanh", -9.0, 9.0, np.tanh, 3e-7),
        ("Tanh", -0.7, 0.7, np.tanh, 3e-7),
    ],
)
def test_relative_error(op, low, high, function, limit, tmp_path: Path) -> None:
    x = np.linspace(low, high, 100001, dtype=F)
    result = evaluate(op, x, tmp_path).astype(np.float64)
    truth = function(x.astype(np.float64))
    assert np.max(np.abs(result - truth) / np.abs(truth).clip(1e-30)) < limit


def test_exp_saturates_like_ieee_arithmetic(tmp_path: Path) -> None:
    x = np.array([1000.0, 89.0, -1000.0, -104.0], F)
    result = evaluate("Exp", x, tmp_path)
    assert np.isinf(result[0]) and np.isinf(result[1])
    assert result[2] == 0.0 and result[3] == 0.0


@pytest.mark.parametrize(
    "real", [1e-9, 3.7e-6, 0.0004, 0.25, 0.5, 0.75, 0.9999999, 1.0, 1.5, 200.0]
)
def test_multiplier_reconstructs_the_ratio(real: float) -> None:
    mult, shift = multiplier(real)
    assert 2**30 <= mult < 2**31
    assert abs(mult * 2.0 ** (-31 - shift) - real) <= real * 2.0**-30


def test_multiplier_edges() -> None:
    assert multiplier(0.0) == (0, 0)
    assert multiplier(1e-30) == (0, 0)
    with pytest.raises(CompileError):
        multiplier(float("nan"))
    with pytest.raises(CompileError):
        multiplier(1e12)


def test_add_multipliers_fit_32_bit_arithmetic() -> None:
    for a, b in [(0.5, 0.25), (2.0, 0.01), (1e-4, 1e-5), (300.0, 2.0)]:
        ma, mb, bits = add_multipliers(a, b)
        assert 255 * (ma + mb) + (1 << (bits - 1)) < 2**31
        assert abs(ma / 2.0**bits - a) <= 2.0**-bits
    with pytest.raises(CompileError):
        add_multipliers(1e7, 1.0)
