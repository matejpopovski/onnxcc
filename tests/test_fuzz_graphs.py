"""Random graphs: memcheck on the emitted C, onnxruntime on the numbers, ASan on the rest."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from hypothesis import given, settings
from hypothesis import strategies as st
from onnx import helper

from onnxcc.compiler import compile_model, quantized_graph
from onnxcc.export import to_onnx
from onnxcc.memcheck import verify
from onnxcc.quantize import QuantizeOptions
from onnxcc.toolchain import STRICT_FLAGS

from .support import (
    assert_close,
    build,
    fuzz_budget,
    make_model,
    ort_run,
    run_isolated,
    run_planned,
)

node = helper.make_node
F = np.float32
UNARY = ["Relu", "Sigmoid", "Tanh", "Abs", "Neg", "LeakyRelu", "HardSwish", "Softmax"]
BINARY = ["Add", "Sub", "Mul", "Min", "Max"]


@dataclass
class Builder:
    rng: np.random.Generator
    nodes: list[onnx.NodeProto] = field(default_factory=list)
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    tensors: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def fresh(self, prefix: str = "t") -> str:
        return f"{prefix}{len(self.nodes)}_{len(self.initializers)}"

    def const(self, array: np.ndarray) -> str:
        name = self.fresh("c")
        self.initializers[name] = array
        return name

    def add(self, op: str, inputs: list[str], shape: tuple[int, ...], **attrs: object) -> None:
        output = self.fresh()
        self.nodes.append(node(op, inputs, [output], **attrs))
        self.tensors.append((output, shape))


def apply_op(builder: Builder, kind: int, pick_a: int, pick_b: int) -> None:
    """Grow the graph by one op whose legality follows from the shape of the operands it picks."""
    # Prefer recent tensors, which grows deep graphs instead of many one-op branches.
    tensors = builder.tensors
    name, shape = tensors[-1 - pick_a % min(len(tensors), 5)]
    other_name, other_shape = tensors[-1 - pick_b % min(len(tensors), 8)]
    rng = builder.rng
    size = int(np.prod(shape))
    match kind % 12:
        case 0:
            op = UNARY[(pick_a + pick_b) % len(UNARY)]
            builder.add(op, [name], shape)
        case 1:
            if shape == other_shape:
                builder.add(BINARY[pick_b % len(BINARY)], [name, other_name], shape)
            else:
                constant = rng.uniform(-1, 1, shape[-1:]).astype(F)
                builder.add(BINARY[pick_b % len(BINARY)], [name, builder.const(constant)], shape)
        case 2 if len(shape) == 4 and shape[2] == shape[3]:
            channels = 1 + pick_b % 3
            w = rng.uniform(-1, 1, (channels, shape[1], 3, 3)).astype(F)
            result = (shape[0], channels, shape[2], shape[3])
            builder.add(
                "Conv", [name, builder.const(w)], result, kernel_shape=[3, 3], pads=[1, 1, 1, 1]
            )
        case 3 if len(shape) == 4 and shape[2] >= 2 and shape[3] >= 2:
            op = ("MaxPool", "AveragePool")[pick_b % 2]
            result = (shape[0], shape[1], shape[2] // 2, shape[3] // 2)
            builder.add(op, [name], result, kernel_shape=[2, 2], strides=[2, 2])
        case 4:
            builder.add("Flatten", [name], (1, size))
        case 5 if len(shape) == 2:
            columns = 1 + pick_b % 4
            w = rng.uniform(-1, 1, (shape[1], columns)).astype(F)
            builder.add("Gemm", [name, builder.const(w)], (shape[0], columns))
        case 6 if len(shape) == 4:
            builder.add(
                "Transpose", [name], (shape[0], shape[2], shape[3], shape[1]), perm=[0, 2, 3, 1]
            )
        case 7 if (
            len(shape) == 4
            and len(other_shape) == 4
            and shape[2:] == other_shape[2:]
            and shape[0] == other_shape[0]
        ):
            result = (shape[0], shape[1] + other_shape[1], shape[2], shape[3])
            builder.add("Concat", [name, other_name], result, axis=1)
        case 8 if len(shape) == 4:
            builder.add("GlobalAveragePool", [name], (*shape[:2], 1, 1))
        case 9:
            rank = len(shape)
            pads = np.array([0] * (2 * rank - 1) + [1], np.int64)
            result = (*shape[:-1], shape[-1] + 1)
            builder.add("Pad", [name, builder.const(pads)], result)
        case 10 if shape[-1] >= 2:
            first, second = builder.fresh("s"), builder.fresh("u")
            builder.nodes.append(node("Split", [name], [first, second], axis=-1, num_outputs=2))
            half = (shape[-1] + 1) // 2
            builder.tensors.append((first, (*shape[:-1], half)))
            builder.tensors.append((second, (*shape[:-1], shape[-1] - half)))
        case 11:
            builder.add("Reshape", [name, builder.const(np.array([1, -1], np.int64))], (1, size))
        case _:
            builder.add(UNARY[(pick_a + pick_b) % len(UNARY)], [name], shape)


def random_model(seed: int, ops: list[tuple[int, int, int]], extra_outputs: list[int]) -> tuple:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (1, 2, 8, 8)).astype(F)
    builder = Builder(rng, tensors=[("x", x.shape)])
    for kind, pick_a, pick_b in ops:
        apply_op(builder, kind, pick_a, pick_b)
    produced = [name for name, _ in builder.tensors if name != "x"]
    outputs = [produced[-1]]
    outputs += sorted({produced[index % len(produced)] for index in extra_outputs} - set(outputs))
    model = make_model(builder.nodes, {"x": x}, builder.initializers, outputs, opset=18)
    return model, {"x": x}


op_lists = st.lists(
    st.tuples(st.sampled_from(range(12)), st.sampled_from(range(64)), st.sampled_from(range(64))),
    min_size=4,
    max_size=28,
)


@settings(max_examples=fuzz_budget(40))
@given(seed=st.integers(0, 2**31), ops=op_lists, extra=st.lists(st.integers(0, 100), max_size=3))
def test_random_graphs_match_onnxruntime(seed: int, ops: list, extra: list[int]) -> None:
    model, feed = random_model(seed, ops, extra)
    expected = ort_run(model, feed)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        planned = run_planned(model, feed, path)
        assert_close(planned, expected, rtol=3e-3, atol=3e-4)
        isolated = run_isolated(model, feed, path)
        if isolated is not None:
            assert_close(planned, isolated, rtol=1e-4, atol=1e-5)


@settings(max_examples=fuzz_budget(25))
@given(seed=st.integers(0, 2**31), ops=op_lists)
def test_random_graphs_survive_quantization(seed: int, ops: list) -> None:
    model, feed = random_model(seed, ops, [])
    rng = np.random.default_rng(seed + 1)
    samples = [rng.uniform(-1, 1, (16, 1, 2, 8, 8)).astype(F)]
    options = QuantizeOptions(method="minmax", weight_limit=63)
    result = compile_model(model, name="quantized", calibration=samples, quantize=options)
    verify(result.source)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        binary = build(result, path, [*STRICT_FLAGS, "-O2"])
        compiled = [out[0] for out in binary.run([feed["x"][None]])]
    reference = to_onnx(quantized_graph(model, samples, options))
    expected = ort_run(reference, feed)
    for got, want in zip(compiled, expected, strict=True):
        assert got.shape == want.shape
        spread = max(float(np.ptp(want)), 1e-3)
        assert np.abs(got - want).max() <= 0.25 * spread + 1e-3
