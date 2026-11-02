"""ONNX protobuf to IR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from .errors import CompileError, UnsupportedError
from .ir import F32, I8, Graph, Node, Value

MIN_OPSET = 13
MAX_OPSET = 23

# Evaluated at compile time. Reaching code generation with one of these is an error.
COMPILE_TIME_OPS = frozenset(
    {"Cast", "Constant", "ConstantOfShape", "Gather", "Range", "Shape", "Size"}
)

RUNTIME_OPS = frozenset(
    {
        "Abs",
        "Add",
        "AveragePool",
        "BatchNormalization",
        "Clip",
        "Concat",
        "Conv",
        "DequantizeLinear",
        "Div",
        "Dropout",
        "Exp",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "LeakyRelu",
        "MatMul",
        "Max",
        "MaxPool",
        "Min",
        "Mul",
        "Neg",
        "PRelu",
        "Pad",
        "QLinearConv",
        "QLinearMatMul",
        "QuantizeLinear",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Squeeze",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
    }
)

MICROSOFT_OPS = frozenset(
    {
        "QGemm",
        "QLinearAdd",
        "QLinearAveragePool",
        "QLinearConcat",
        "QLinearGlobalAveragePool",
        "QLinearMul",
    }
)

SUPPORTED_OPS = COMPILE_TIME_OPS | RUNTIME_OPS | MICROSOFT_OPS


def _attribute(attr: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attr)
    if attr.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(value)
    if attr.type == onnx.AttributeProto.STRING:
        return value.decode()
    if attr.type in (onnx.AttributeProto.INTS, onnx.AttributeProto.FLOATS):
        return list(value)
    if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
        raise UnsupportedError(f"attribute {attr.name!r} holds a subgraph; control flow is out")
    return value


def _shape_of(
    info: onnx.ValueInfoProto, overrides: Mapping[str, Sequence[int]]
) -> tuple[int, ...] | None:
    if info.name in overrides:
        return tuple(int(dim) for dim in overrides[info.name])
    if not info.type.tensor_type.HasField("shape"):
        return None
    shape = []
    for axis, dim in enumerate(info.type.tensor_type.shape.dim):
        if dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif axis == 0:
            # A dynamic leading axis is a batch axis in practice, and one sample is what a
            # statically planned arena can hold.
            shape.append(1)
        else:
            raise CompileError(
                f"input {info.name!r} has dynamic axis {axis}; "
                f"pass --input-shape {info.name}=... to fix it"
            )
    return tuple(shape)


def _check_supported(model: onnx.ModelProto) -> int:
    opset = None
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            opset = entry.version
        elif entry.domain != "com.microsoft":
            raise UnsupportedError(f"operator domain {entry.domain!r} is not supported")
    if opset is None or not MIN_OPSET <= opset <= MAX_OPSET:
        raise UnsupportedError(
            f"opset {opset} is outside the supported range {MIN_OPSET}..{MAX_OPSET}; "
            "onnx.version_converter can upgrade older models"
        )
    missing = set()
    for node in model.graph.node:
        allowed = MICROSOFT_OPS if node.domain == "com.microsoft" else SUPPORTED_OPS
        if node.domain not in ("", "ai.onnx", "com.microsoft") or node.op_type not in allowed:
            missing.add(node.op_type)
    if missing:
        raise UnsupportedError(f"unsupported operators: {', '.join(sorted(missing))}")
    return int(opset)


def import_model(
    model: onnx.ModelProto | str | Path,
    input_shapes: Mapping[str, Sequence[int]] | None = None,
    name: str | None = None,
) -> Graph:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    overrides = input_shapes or {}
    opset = _check_supported(model)
    onnx_graph = model.graph

    values: dict[str, Value] = {}
    for tensor in onnx_graph.initializer:
        array = numpy_helper.to_array(tensor)
        values[tensor.name] = Value(tensor.name, array.dtype, tuple(array.shape), array)
    if onnx_graph.sparse_initializer:
        raise UnsupportedError("sparse initializers are not supported")

    inputs = []
    for info in onnx_graph.input:
        if info.name in values:
            continue
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type))
        if dtype not in (F32, I8):
            raise UnsupportedError(f"input {info.name!r} has type {dtype}; need float32 or int8")
        shape = _shape_of(info, overrides)
        if shape is None:
            raise CompileError(f"input {info.name!r} has no shape; pass --input-shape")
        values[info.name] = Value(info.name, dtype, shape)
        inputs.append(info.name)
    unknown = set(overrides) - set(inputs)
    if unknown:
        raise CompileError(f"--input-shape names unknown inputs: {', '.join(sorted(unknown))}")

    nodes = []
    for index, proto in enumerate(onnx_graph.node):
        attrs = {attr.name: _attribute(attr) for attr in proto.attribute}
        for output in proto.output:
            if output:
                values.setdefault(output, Value(output))
        nodes.append(
            Node(
                proto.op_type,
                list(proto.input),
                [item for item in proto.output if item],
                attrs,
                proto.name or f"{proto.op_type}_{index}",
            )
        )
    for node in nodes:
        for item in node.inputs:
            if item and item not in values:
                raise CompileError(f"node {node.name!r} reads undefined value {item!r}")
    outputs = [info.name for info in onnx_graph.output]
    return Graph(name or onnx_graph.name or "model", values, nodes, inputs, outputs, opset)
