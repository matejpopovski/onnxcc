"""Shared machinery: build ONNX by hand, run it in onnxruntime and through generated C."""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from onnxcc.compiler import CompileResult, compile_model
from onnxcc.memcheck import verify
from onnxcc.toolchain import SANITIZE_FLAGS, STRICT_FLAGS, Binary, harness_source, link_harness


def fuzz_budget(examples: int) -> int:
    """Example count for property tests. CI uses it as written, campaigns scale it up."""
    return examples * int(os.environ.get("ONNXCC_FUZZ_SCALE", "1"))


def make_model(
    nodes: Sequence[onnx.NodeProto],
    inputs: Mapping[str, np.ndarray],
    initializers: Mapping[str, np.ndarray],
    outputs: Sequence[str],
    opset: int = 17,
    microsoft: bool = False,
) -> onnx.ModelProto:
    def info(name: str, array: np.ndarray) -> onnx.ValueInfoProto:
        element = helper.np_dtype_to_tensor_dtype(array.dtype)
        return helper.make_tensor_value_info(name, element, list(array.shape))

    graph = helper.make_graph(
        list(nodes),
        "case",
        [info(name, array) for name, array in inputs.items()],
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        [numpy_helper.from_array(array, name) for name, array in initializers.items()],
    )
    imports = [helper.make_opsetid("", opset)]
    if microsoft:
        imports.append(helper.make_opsetid("com.microsoft", 1))
    return helper.make_model(graph, opset_imports=imports, ir_version=8)


def ort_run(model: onnx.ModelProto, feed: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, dict(feed))


@functools.lru_cache(maxsize=1)
def sanitizers_work() -> bool:
    """Probe the active compiler. gcc on macOS arm64, for one, ships no sanitizer runtime."""
    from onnxcc.toolchain import host_cc

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "probe.c"
        binary = Path(directory) / "probe"
        source.write_text("int main(void) { return 0; }\n")
        built = subprocess.run(
            [host_cc(), *SANITIZE_FLAGS, str(source), "-o", str(binary)], capture_output=True
        )
        working = built.returncode == 0 and subprocess.run([binary]).returncode == 0
    if not working and os.environ.get("ONNXCC_REQUIRE_SANITIZERS"):
        raise RuntimeError("sanitizers required but unavailable")
    return working


def build(result: CompileResult, directory: Path, flags: Sequence[str]) -> Binary:
    harness = harness_source(result.c_name, result.prototype, result.inputs, result.outputs, False)
    path = link_harness(result.source, harness, directory, result.c_name, flags)
    return Binary(path, result.inputs, result.outputs)


def run_planned(
    model: onnx.ModelProto,
    feed: Mapping[str, np.ndarray],
    directory: Path,
    **options: Any,
) -> list[np.ndarray]:
    """Arena build at -O2 with strict warnings, after the independent memory check."""
    result = compile_model(model, name="planned", **options)
    if any(line.lstrip().startswith("/*@ ") for line in result.source.splitlines()):
        verify(result.source)
    binary = build(result, directory, [*STRICT_FLAGS, "-O2"])
    return [out[0] for out in binary.run([value[None] for value in feed.values()])]


def run_isolated(
    model: onnx.ModelProto,
    feed: Mapping[str, np.ndarray],
    directory: Path,
    **options: Any,
) -> list[np.ndarray] | None:
    """One global array per tensor, so ASan sees every tensor boundary, not just the arena's."""
    if not sanitizers_work():
        return None
    result = compile_model(model, name="isolated", isolated=True, **options)
    binary = build(result, directory, [*STRICT_FLAGS, *SANITIZE_FLAGS, "-O1"])
    return [out[0] for out in binary.run([value[None] for value in feed.values()])]


def assert_close(
    actual: Sequence[np.ndarray], expected: Sequence[np.ndarray], rtol: float, atol: float
) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == want.shape
        assert got.dtype == want.dtype or want.dtype.kind in "iu" or got.dtype == np.float32
        if want.dtype.kind in "iu":
            assert np.abs(got.astype(np.int64) - want.astype(np.int64)).max(initial=0) <= atol
        else:
            np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@dataclass
class Case:
    """One model plus its inputs. `nodes` may hold several ops when a pattern needs them."""

    id: str
    nodes: list[onnx.NodeProto]
    inputs: dict[str, np.ndarray]
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=lambda: ["y"])
    opset: int = 17
    microsoft: bool = False
    rtol: float = 2e-4
    atol: float = 2e-5

    def model(self) -> onnx.ModelProto:
        return make_model(
            self.nodes, self.inputs, self.initializers, self.outputs, self.opset, self.microsoft
        )

    @property
    def ops(self) -> set[str]:
        return {node.op_type for node in self.nodes}


def check_case(case: Case, directory: Path) -> None:
    model = case.model()
    expected = ort_run(model, case.inputs)
    planned = run_planned(model, case.inputs, directory)
    assert_close(planned, expected, case.rtol, case.atol)
    isolated = run_isolated(model, case.inputs, directory)
    if isolated is not None:
        assert_close(isolated, expected, case.rtol, case.atol)
        for left, right in zip(planned, isolated, strict=True):
            # Same kernels, same inputs. Only -O2 versus -O1 float contraction may differ.
            np.testing.assert_allclose(
                left.astype(float), right.astype(float), rtol=1e-5, atol=1e-6
            )
