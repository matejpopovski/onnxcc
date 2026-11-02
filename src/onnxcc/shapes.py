"""Forward shape and type inference, merged with constant propagation.

The two are one sweep because each depends on the other: a Reshape target is often computed from
a Shape node, and a Shape node folds only once its input's shape is known. Rules also rewrite
convenience attributes (`auto_pad`, defaulted strides) into explicit form, so later stages see one
canonical spelling. Running the sweep twice is harmless.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .errors import CompileError, UnsupportedError
from .folding import FOLDERS, axes_from, resolve_reshape, slice_ranges
from .ir import F32, I8, I64, Graph, Node

Rule = Callable[[Graph, Node], None]
Shape = tuple[int, ...]


def _shape(graph: Graph, name: str) -> Shape:
    shape = graph.values[name].shape
    if shape is None:
        raise CompileError(f"shape of {name!r} is unknown")
    return shape


def _set(graph: Graph, node: Node, shape: Shape, dtype: Any, index: int = 0) -> None:
    if any(dim <= 0 for dim in shape):
        raise CompileError(f"node {node.name!r} yields an empty tensor of shape {shape}")
    value = graph.values[node.outputs[index]]
    value.shape, value.dtype = tuple(shape), dtype


def _like_input(graph: Graph, node: Node) -> None:
    source = graph.values[node.inputs[0]]
    _set(graph, node, _shape(graph, source.name), source.dtype)


def broadcast(left: Shape, right: Shape, where: str) -> Shape:
    try:
        return tuple(int(dim) for dim in np.broadcast_shapes(left, right))
    except ValueError:
        raise CompileError(f"{where}: shapes {left} and {right} do not broadcast") from None


def _binary(graph: Graph, node: Node) -> None:
    if len(node.inputs) == 1:
        _like_input(graph, node)
        return
    shape: Shape = ()
    for item in node.inputs:
        shape = broadcast(shape, _shape(graph, item), node.name)
    _set(graph, node, shape, graph.values[node.inputs[0]].dtype)


def _window(node: Node, in_shape: Shape, kernel: list[int]) -> list[int]:
    """Resolve strides, dilations and pads, store them, and return the output spatial shape."""
    spatial = in_shape[2:]
    if len(spatial) not in (1, 2) or len(kernel) != len(spatial):
        raise UnsupportedError(f"{node.name}: only 1-D and 2-D spatial tensors are supported")
    attrs = node.attrs
    strides = [int(item) for item in attrs.get("strides", [1] * len(spatial))]
    dilations = [int(item) for item in attrs.get("dilations", [1] * len(spatial))]
    pads = [int(item) for item in attrs.get("pads", [0] * (2 * len(spatial)))]
    auto_pad = attrs.pop("auto_pad", "NOTSET")
    ceil_mode = int(attrs.get("ceil_mode", 0))
    outputs = []
    for axis, dim in enumerate(spatial):
        extent = (kernel[axis] - 1) * dilations[axis] + 1
        stride = strides[axis]
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out = (dim + stride - 1) // stride
            total = max((out - 1) * stride + extent - dim, 0)
            small, large = total // 2, total - total // 2
            before, after = (small, large) if auto_pad == "SAME_UPPER" else (large, small)
            pads[axis], pads[axis + len(spatial)] = before, after
        elif auto_pad == "VALID":
            pads[axis], pads[axis + len(spatial)] = 0, 0
        elif auto_pad != "NOTSET":
            raise UnsupportedError(f"{node.name}: unknown auto_pad {auto_pad!r}")
        padded = dim + pads[axis] + pads[axis + len(spatial)] - extent
        if padded < 0:
            raise CompileError(f"{node.name}: window larger than padded input")
        out = (padded + stride - 1) // stride + 1 if ceil_mode else padded // stride + 1
        if pads[axis] >= extent or pads[axis + len(spatial)] >= extent:
            raise UnsupportedError(f"{node.name}: padding must be smaller than the window")
        taps = range(0, extent, dilations[axis])
        for position in range(out):
            first = position * stride - pads[axis]
            if not any(0 <= first + tap < dim for tap in taps):
                raise UnsupportedError(f"{node.name}: a window covers no input element")
        outputs.append(out)
    attrs.update(strides=strides, dilations=dilations, pads=pads, kernel_shape=kernel)
    return outputs


def _conv_like(graph: Graph, node: Node, x_name: str, w_name: str, dtype: Any) -> None:
    x_shape = _shape(graph, x_name)
    w_shape = _shape(graph, w_name)
    if len(x_shape) != len(w_shape):
        raise CompileError(f"{node.name}: weight rank {len(w_shape)} != input rank {len(x_shape)}")
    kernel = [int(item) for item in node.attrs.get("kernel_shape", w_shape[2:])]
    group = int(node.attrs.get("group", 1))
    if x_shape[1] != w_shape[1] * group or w_shape[0] % group:
        raise CompileError(f"{node.name}: channels and group disagree")
    node.attrs["group"] = group
    spatial = _window(node, x_shape, kernel)
    _set(graph, node, (x_shape[0], w_shape[0], *spatial), dtype)


def _conv(graph: Graph, node: Node) -> None:
    _conv_like(graph, node, node.inputs[0], node.inputs[1], graph.values[node.inputs[0]].dtype)


def _qlinear_conv(graph: Graph, node: Node) -> None:
    _conv_like(graph, node, node.inputs[0], node.inputs[3], I8)


def _pool(graph: Graph, node: Node) -> None:
    x_shape = _shape(graph, node.inputs[0])
    if node.attrs.get("storage_order", 0) or len(node.outputs) > 1:
        raise UnsupportedError(f"{node.name}: MaxPool indices output is not supported")
    if node.attrs.get("ceil_mode", 0) and node.attrs.get("count_include_pad", 0):
        raise UnsupportedError(f"{node.name}: ceil_mode with count_include_pad is not supported")
    if node.attrs.get("channels_last", 0):
        raise UnsupportedError(f"{node.name}: channels_last is not supported")
    kernel = [int(item) for item in node.attrs["kernel_shape"]]
    spatial = _window(node, x_shape, kernel)
    _set(graph, node, (*x_shape[:2], *spatial), graph.values[node.inputs[0]].dtype)


def _global_pool(graph: Graph, node: Node) -> None:
    x_shape = _shape(graph, node.inputs[0])
    if len(x_shape) < 3:
        raise CompileError(f"{node.name}: GlobalAveragePool needs a spatial tensor")
    if node.attrs.get("channels_last", 0):
        raise UnsupportedError(f"{node.name}: channels_last is not supported")
    _set(
        graph, node, (*x_shape[:2], *([1] * (len(x_shape) - 2))), graph.values[node.inputs[0]].dtype
    )


def _gemm(graph: Graph, node: Node) -> None:
    a, b = (_shape(graph, item) for item in node.inputs[:2])
    if len(a) != 2 or len(b) != 2:
        raise UnsupportedError(f"{node.name}: Gemm needs rank-2 inputs")
    if node.attrs.get("transA", 0):
        a = a[::-1]
    if node.attrs.get("transB", 0):
        b = b[::-1]
    if a[1] != b[0]:
        raise CompileError(f"{node.name}: cannot multiply {a} by {b}")
    _set(graph, node, (a[0], b[1]), graph.values[node.inputs[0]].dtype)


def _matmul(graph: Graph, node: Node) -> None:
    a, b = _shape(graph, node.inputs[0]), _shape(graph, node.inputs[1])
    if len(b) > 2 or len(a) < 1:
        raise UnsupportedError(f"{node.name}: MatMul with a batched right operand is unsupported")
    left = a if len(a) > 1 else (1, *a)
    right = b if len(b) > 1 else (*b, 1)
    if left[-1] != right[0]:
        raise CompileError(f"{node.name}: cannot multiply {a} by {b}")
    shape = list(left[:-1])
    if len(a) == 1:
        shape.pop()
    if len(b) > 1:
        shape.append(right[1])
    _set(graph, node, tuple(shape), graph.values[node.inputs[0]].dtype)


def _qlinear_matmul(graph: Graph, node: Node) -> None:
    a, b = _shape(graph, node.inputs[0]), _shape(graph, node.inputs[3])
    if len(a) != 2 or len(b) != 2 or a[1] != b[0]:
        raise UnsupportedError(f"{node.name}: QLinearMatMul needs two conformable matrices")
    _set(graph, node, (a[0], b[1]), I8)


def _qgemm(graph: Graph, node: Node) -> None:
    a, b = _shape(graph, node.inputs[0]), _shape(graph, node.inputs[3])
    if node.attrs.get("transB", 0):
        b = b[::-1]
    if node.attrs.get("transA", 0) or len(a) != 2 or a[1] != b[0]:
        raise UnsupportedError(f"{node.name}: unsupported QGemm form")
    _set(graph, node, (a[0], b[1]), I8)


def _softmax(graph: Graph, node: Node) -> None:
    _like_input(graph, node)
    axis = int(node.attrs.get("axis", -1))
    rank = len(_shape(graph, node.inputs[0]))
    if not -rank <= axis < rank:
        raise CompileError(f"{node.name}: axis {axis} out of range")
    node.attrs["axis"] = axis % rank


def _reshape(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    target = graph.const(node.inputs[1])
    if target is None:
        raise UnsupportedError(f"{node.name}: Reshape target must be known at compile time")
    resolved = resolve_reshape(shape, target, bool(node.attrs.get("allowzero", 0)))
    if int(np.prod(resolved, dtype=np.int64)) != int(np.prod(shape, dtype=np.int64)):
        raise CompileError(f"{node.name}: cannot reshape {shape} to {resolved}")
    _set(graph, node, resolved, graph.values[node.inputs[0]].dtype)


def _flatten(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    axis = int(node.attrs.get("axis", 1))
    axis += len(shape) if axis < 0 else 0
    head = int(np.prod(shape[:axis], dtype=np.int64))
    tail = int(np.prod(shape[axis:], dtype=np.int64))
    _set(graph, node, (head, tail), graph.values[node.inputs[0]].dtype)


def _args(graph: Graph, node: Node) -> list[Any]:
    return [graph.const(item) for item in node.inputs]


def _squeeze(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    axes = axes_from(node, _args(graph, node), 1)
    drop = (
        {axis % len(shape) for axis in axes}
        if axes
        else {axis for axis, dim in enumerate(shape) if dim == 1}
    )
    if any(shape[axis] != 1 for axis in drop):
        raise CompileError(f"{node.name}: cannot squeeze a dimension larger than 1")
    _set(
        graph,
        node,
        tuple(dim for axis, dim in enumerate(shape) if axis not in drop),
        graph.values[node.inputs[0]].dtype,
    )


def _unsqueeze(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    axes = axes_from(node, _args(graph, node), 1)
    if axes is None:
        raise UnsupportedError(f"{node.name}: Unsqueeze axes must be constant")
    rank = len(shape) + len(axes)
    positions = {axis % rank for axis in axes}
    dims = iter(shape)
    result = tuple(1 if axis in positions else next(dims) for axis in range(rank))
    _set(graph, node, result, graph.values[node.inputs[0]].dtype)


def _transpose(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    perm = [int(axis) for axis in node.attrs.get("perm", range(len(shape) - 1, -1, -1))]
    if sorted(perm) != list(range(len(shape))):
        raise CompileError(f"{node.name}: invalid permutation {perm}")
    node.attrs["perm"] = perm
    _set(graph, node, tuple(shape[axis] for axis in perm), graph.values[node.inputs[0]].dtype)


def _concat_shape(graph: Graph, node: Node, names: list[str], dtype: Any) -> None:
    shapes = [_shape(graph, item) for item in names]
    rank = len(shapes[0])
    axis = int(node.attrs["axis"]) % rank
    node.attrs["axis"] = axis
    for shape in shapes[1:]:
        if shape[:axis] != shapes[0][:axis] or shape[axis + 1 :] != shapes[0][axis + 1 :]:
            raise CompileError(f"{node.name}: mismatched Concat shapes {shapes}")
    total = sum(shape[axis] for shape in shapes)
    _set(graph, node, (*shapes[0][:axis], total, *shapes[0][axis + 1 :]), dtype)


def _concat(graph: Graph, node: Node) -> None:
    _concat_shape(graph, node, node.inputs, graph.values[node.inputs[0]].dtype)


def _qlinear_concat(graph: Graph, node: Node) -> None:
    _concat_shape(graph, node, node.inputs[2::3], I8)


def _const_list(graph: Graph, node: Node, position: int) -> Any:
    if position >= len(node.inputs) or not node.inputs[position]:
        return None
    array = graph.const(node.inputs[position])
    if array is None:
        raise UnsupportedError(f"{node.name}: input {position} must be constant")
    return array


def _slice(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    starts, ends = _const_list(graph, node, 1), _const_list(graph, node, 2)
    axes, steps = _const_list(graph, node, 3), _const_list(graph, node, 4)
    ranges = slice_ranges(shape, starts, ends, axes, steps)
    _set(graph, node, tuple(item[1] for item in ranges), graph.values[node.inputs[0]].dtype)


def _pad(graph: Graph, node: Node) -> None:
    if node.attrs.get("mode", "constant") != "constant":
        raise UnsupportedError(f"{node.name}: only constant padding is supported")
    shape = _shape(graph, node.inputs[0])
    pads = [int(item) for item in _const_list(graph, node, 1)]
    axes = _const_list(graph, node, 3)
    rank = len(shape)
    if axes is not None:
        full = [0] * (2 * rank)
        count = len(axes)
        for position, axis in enumerate(int(item) % rank for item in axes):
            full[axis], full[axis + rank] = pads[position], pads[position + count]
        pads = full
    if min(pads) < 0:
        raise UnsupportedError(f"{node.name}: negative pads are not supported")
    node.attrs["resolved_pads"] = pads
    result = tuple(dim + pads[axis] + pads[axis + rank] for axis, dim in enumerate(shape))
    _set(graph, node, result, graph.values[node.inputs[0]].dtype)


def _reduce(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    axes = axes_from(node, _args(graph, node), 1)
    if axes is None and len(node.inputs) > 1 and node.inputs[1]:
        raise UnsupportedError(f"{node.name}: axes must be constant")
    if not axes:
        axes = [] if node.attrs.get("noop_with_empty_axes", 0) else list(range(len(shape)))
    axes = sorted({axis % len(shape) for axis in axes})
    keep = int(node.attrs.get("keepdims", 1))
    node.attrs["resolved_axes"] = axes
    result = [1 if axis in axes else dim for axis, dim in enumerate(shape)]
    if not keep:
        result = [dim for axis, dim in enumerate(result) if axis not in axes]
    _set(graph, node, tuple(result), graph.values[node.inputs[0]].dtype)


def _split(graph: Graph, node: Node) -> None:
    shape = _shape(graph, node.inputs[0])
    axis = int(node.attrs.get("axis", 0)) % len(shape)
    node.attrs["axis"] = axis
    sizes = _const_list(graph, node, 1)
    if sizes is None:
        sizes = node.attrs.get("split")
    if sizes is None:
        parts = len(node.outputs)
        chunk = (shape[axis] + parts - 1) // parts
        sizes = [chunk] * (parts - 1) + [shape[axis] - chunk * (parts - 1)]
    sizes = [int(item) for item in sizes]
    if sum(sizes) != shape[axis] or len(sizes) != len(node.outputs):
        raise CompileError(f"{node.name}: split sizes {sizes} do not match axis {shape[axis]}")
    node.attrs["resolved_split"] = sizes
    for index, size in enumerate(sizes):
        piece = (*shape[:axis], size, *shape[axis + 1 :])
        _set(graph, node, piece, graph.values[node.inputs[0]].dtype, index)


def _batch_norm(graph: Graph, node: Node) -> None:
    if node.attrs.get("training_mode", 0):
        raise UnsupportedError(f"{node.name}: BatchNormalization in training mode")
    if len(node.outputs) > 1:
        raise UnsupportedError(f"{node.name}: BatchNormalization statistics outputs are unused")
    _like_input(graph, node)


def _dropout(graph: Graph, node: Node) -> None:
    mode = _const_list(graph, node, 2)
    if mode is not None and bool(mode.reshape(-1)[0]):
        raise UnsupportedError(f"{node.name}: Dropout in training mode")
    if len(node.outputs) > 1:
        raise UnsupportedError(f"{node.name}: Dropout mask output is not supported")
    _like_input(graph, node)


def _cast(graph: Graph, node: Node) -> None:
    source = graph.values[node.inputs[0]]
    if onnx.helper.tensor_dtype_to_np_dtype(node.attrs["to"]) != source.dtype:
        raise UnsupportedError(f"{node.name}: runtime Cast is limited to a no-op")
    _like_input(graph, node)


def _clip(graph: Graph, node: Node) -> None:
    for position in (1, 2):
        _const_list(graph, node, position)
    _like_input(graph, node)


def _quantize(graph: Graph, node: Node) -> None:
    if len(node.inputs) < 3 or not node.inputs[2]:
        raise UnsupportedError(f"{node.name}: QuantizeLinear without a zero point yields uint8")
    zero_point = graph.const(node.inputs[2])
    scale = graph.const(node.inputs[1])
    if zero_point is None or scale is None or zero_point.dtype != I8 or scale.size != 1:
        raise UnsupportedError(f"{node.name}: runtime QuantizeLinear must be per-tensor int8")
    _set(graph, node, _shape(graph, node.inputs[0]), I8)


def _dequantize(graph: Graph, node: Node) -> None:
    scale = graph.const(node.inputs[1])
    if graph.values[node.inputs[0]].dtype != I8 or scale is None or scale.size != 1:
        raise UnsupportedError(f"{node.name}: runtime DequantizeLinear must be per-tensor int8")
    _set(graph, node, _shape(graph, node.inputs[0]), F32)


def _qlinear_binary(graph: Graph, node: Node) -> None:
    shape = broadcast(_shape(graph, node.inputs[0]), _shape(graph, node.inputs[3]), node.name)
    _set(graph, node, shape, I8)


RULES: dict[str, Rule] = {
    "Abs": _like_input,
    "Add": _binary,
    "AveragePool": _pool,
    "BatchNormalization": _batch_norm,
    "Cast": _cast,
    "Clip": _clip,
    "Concat": _concat,
    "Conv": _conv,
    "DequantizeLinear": _dequantize,
    "Div": _binary,
    "Dropout": _dropout,
    "Exp": _like_input,
    "Flatten": _flatten,
    "Gemm": _gemm,
    "GlobalAveragePool": _global_pool,
    "HardSigmoid": _like_input,
    "HardSwish": _like_input,
    "Identity": _like_input,
    "LeakyRelu": _like_input,
    "MatMul": _matmul,
    "Max": _binary,
    "MaxPool": _pool,
    "Min": _binary,
    "Mul": _binary,
    "Neg": _like_input,
    "PRelu": _binary,
    "Pad": _pad,
    "QGemm": _qgemm,
    "QLinearAdd": _qlinear_binary,
    "QLinearAveragePool": _pool,
    "QLinearConcat": _qlinear_concat,
    "QLinearConv": _qlinear_conv,
    "QLinearGlobalAveragePool": _global_pool,
    "QLinearMatMul": _qlinear_matmul,
    "QLinearMul": _qlinear_binary,
    "QuantizeLinear": _quantize,
    "ReduceMean": _reduce,
    "ReduceSum": _reduce,
    "Relu": _like_input,
    "Reshape": _reshape,
    "Sigmoid": _like_input,
    "Slice": _slice,
    "Softmax": _softmax,
    "Split": _split,
    "Squeeze": _squeeze,
    "Sub": _binary,
    "Tanh": _like_input,
    "Transpose": _transpose,
    "Unsqueeze": _unsqueeze,
}


def _literal_constant(node: Node) -> Any:
    attrs = node.attrs
    if "value" in attrs:
        return attrs["value"]
    if "value_float" in attrs:
        return np.array(attrs["value_float"], np.float32)
    if "value_floats" in attrs:
        return np.array(attrs["value_floats"], np.float32)
    if "value_int" in attrs:
        return np.array(attrs["value_int"], np.int64)
    if "value_ints" in attrs:
        return np.array(attrs["value_ints"], np.int64)
    raise UnsupportedError(f"{node.name}: unsupported Constant payload {sorted(attrs)}")


def _fold(graph: Graph, node: Node) -> bool:
    """Replace `node` by constants if possible. Returns True when the node is gone."""
    result = None
    if node.op == "Constant":
        result = _literal_constant(node)
    elif node.op in ("Shape", "Size"):
        shape = graph.values[node.inputs[0]].shape
        if shape is not None:
            start = node.attrs.get("start", 0)
            end = node.attrs.get("end", len(shape))
            result = (
                np.array(shape[start:end], I64)
                if node.op == "Shape"
                else np.array(np.prod(shape, dtype=np.int64), I64)
            )
    elif node.op in FOLDERS and "materialize" not in node.attrs:
        args = [graph.const(item) if item else None for item in node.inputs]
        required = [item for item, arg in zip(node.inputs, args, strict=True) if item]
        if all(graph.values[item].is_const for item in required):
            result = FOLDERS[node.op](node, args)
    if result is None:
        return False
    value = graph.values[node.outputs[0]]
    value.data, value.dtype, value.shape = result, result.dtype, tuple(result.shape)
    return True


def infer(graph: Graph) -> None:
    kept = []
    for node in graph.nodes:
        if _fold(graph, node):
            continue
        rule = RULES.get(node.op)
        if rule is None:
            raise CompileError(f"{node.name}: {node.op} must fold, but its inputs are not constant")
        rule(graph, node)
        kept.append(node)
    graph.nodes = kept
    for name in graph.outputs:
        if graph.values[name].shape is None:
            raise CompileError(f"output {name!r} is never produced")
