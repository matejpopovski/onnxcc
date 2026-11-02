"""Diagnostics and normalization. A compiler that rejects clearly is half of being trustworthy."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from onnxcc.compiler import compile_model
from onnxcc.errors import CompileError, UnsupportedError
from onnxcc.frontend import import_model
from onnxcc.ir import F32, Node
from onnxcc.passes import canonicalize

from .support import make_model

node = helper.make_node
F = np.float32
X = np.zeros((1, 2, 4, 4), F)


def relu_model(opset: int = 17) -> object:
    return make_model([node("Relu", ["x"], ["y"])], {"x": X}, {}, ["y"], opset=opset)


def test_lists_every_unsupported_operator_at_once() -> None:
    model = make_model([node("Sin", ["x"], ["a"]), node("Cos", ["a"], ["y"])], {"x": X}, {}, ["y"])
    with pytest.raises(UnsupportedError, match="Cos, Sin"):
        compile_model(model)


def test_rejects_old_opset_and_unknown_domain() -> None:
    with pytest.raises(UnsupportedError, match="opset 12"):
        compile_model(relu_model(opset=12))
    model = relu_model()
    model.opset_import.append(helper.make_opsetid("com.acme", 1))
    with pytest.raises(UnsupportedError, match=r"com\.acme"):
        compile_model(model)


def test_dynamic_batch_axis_defaults_to_one() -> None:
    model = relu_model()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    assert import_model(model).values["x"].shape == (1, 2, 4, 4)


def test_other_dynamic_axes_need_an_override() -> None:
    model = relu_model()
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
    with pytest.raises(CompileError, match="--input-shape"):
        compile_model(model)
    result = compile_model(model, input_shapes={"x": [1, 2, 3, 3]})
    assert result.inputs[0].shape == (1, 2, 3, 3)


def test_input_shape_override_for_unknown_name_is_an_error() -> None:
    with pytest.raises(CompileError, match="unknown inputs"):
        compile_model(relu_model(), input_shapes={"nope": [1]})


def test_runtime_reshape_target_is_rejected() -> None:
    model = make_model(
        [node("Reshape", ["x", "t"], ["y"])],
        {"x": X, "t": np.array([1, -1], np.int64)},
        {},
        ["y"],
    )
    with pytest.raises(UnsupportedError):
        compile_model(model)


def test_training_mode_batch_norm_is_rejected() -> None:
    params = {name: np.ones(2, F) for name in ("s", "b", "m", "v")}
    bn = node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], training_mode=1)
    with pytest.raises(UnsupportedError, match="training"):
        compile_model(make_model([bn], {"x": X}, params, ["y"], opset=15))


def test_windows_outside_the_input_are_rejected() -> None:
    bad = node("MaxPool", ["x"], ["y"], kernel_shape=[1, 1], strides=[2, 2], ceil_mode=1)
    with pytest.raises(UnsupportedError, match="covers no input"):
        compile_model(make_model([bad], {"x": np.zeros((1, 1, 4, 4), F)}, {}, ["y"]))


def test_ceil_mode_with_count_include_pad_is_rejected() -> None:
    bad = node("AveragePool", ["x"], ["y"], kernel_shape=[2, 2], ceil_mode=1, count_include_pad=1)
    with pytest.raises(UnsupportedError):
        compile_model(make_model([bad], {"x": X}, {}, ["y"]))


def test_shape_mismatch_names_the_node() -> None:
    bad = node("Add", ["x", "z"], ["y"], name="the_add")
    with pytest.raises(CompileError, match="the_add"):
        compile_model(make_model([bad], {"x": X, "z": np.zeros(3, F)}, {}, ["y"]))


def test_auto_pad_is_resolved_to_explicit_pads() -> None:
    conv = node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[4, 4], strides=[2, 2], auto_pad="SAME_LOWER"
    )
    model = make_model(
        [conv], {"x": np.zeros((1, 1, 7, 7), F)}, {"w": np.zeros((1, 1, 4, 4), F)}, ["y"]
    )
    graph = import_model(model)
    canonicalize(graph)
    assert graph.nodes[0].attrs["pads"] == [2, 2, 1, 1]
    assert "auto_pad" not in graph.nodes[0].attrs


def ops(graph: object) -> list[str]:
    return [item.op for item in graph.nodes]  # type: ignore[attr-defined]


def test_lowering_removes_reshape_family_and_relu() -> None:
    nodes = [
        node("Flatten", ["x"], ["a"]),
        node("Identity", ["a"], ["b"]),
        node("Relu", ["b"], ["y"]),
    ]
    graph = import_model(make_model(nodes, {"x": X}, {}, ["y"]))
    canonicalize(graph)
    assert ops(graph) == ["Reshape", "Reshape", "Clip"]


def test_dead_nodes_are_removed() -> None:
    nodes = [node("Abs", ["x"], ["dead"]), node("Relu", ["x"], ["y"])]
    graph = import_model(make_model(nodes, {"x": X}, {}, ["y"]))
    canonicalize(graph)
    assert ops(graph) == ["Clip"]


def batch_norm_chain() -> tuple:
    rng = np.random.default_rng(3)
    w = rng.uniform(-1, 1, (2, 2, 3, 3)).astype(F)
    params = {name: rng.uniform(0.5, 1.5, 2).astype(F) for name in ("s", "b", "m", "v")}
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3]),
        node("BatchNormalization", ["c", "s", "b", "m", "v"], ["n"]),
        node("Relu", ["n"], ["y"]),
    ]
    return make_model(
        nodes, {"x": rng.uniform(-1, 1, X.shape).astype(F)}, {"w": w, **params}, ["y"]
    )


def test_batch_norm_merges_into_conv_and_relu_fuses() -> None:
    result = compile_model(batch_norm_chain())
    assert ops(result.graph) == ["Conv"]
    assert result.graph.nodes[0].attrs["clamp"] == (0.0, None)


def test_shared_conv_output_blocks_batch_norm_folding() -> None:
    model = batch_norm_chain()
    model.graph.output.append(helper.make_empty_tensor_value_info("c"))
    graph = import_model(model)
    canonicalize(graph)
    assert "Mul" in ops(graph) and "BatchNormalization" not in ops(graph)


def test_weights_shared_by_two_convs_are_not_mutated_by_folding() -> None:
    w = np.ones((2, 2, 1, 1), F)
    params = {name: np.array([2.0, 2.0], F) for name in ("s", "v")}
    params |= {name: np.zeros(2, F) for name in ("b", "m")}
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[1, 1]),
        node("BatchNormalization", ["c", "s", "b", "m", "v"], ["n"]),
        node("Conv", ["x", "w"], ["d"], kernel_shape=[1, 1]),
        node("Add", ["n", "d"], ["y"]),
    ]
    graph = import_model(make_model(nodes, {"x": X}, {"w": w, **params}, ["y"]))
    canonicalize(graph)
    assert np.array_equal(graph.values["w"].data, w)


def test_ir_helpers() -> None:
    graph = import_model(relu_model())
    name = graph.add_const("k", np.ones(2, F))
    assert graph.values[name].dtype == F32 and name.startswith("k__")
    assert graph.const("") is None
    assert graph.const("x") is None
    graph.nodes.append(Node("Abs", ["y"], ["z"]))
    assert graph.sole_consumer("y") is None  # "y" is a graph output
