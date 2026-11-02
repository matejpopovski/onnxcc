"""Numpy evaluators for operators whose inputs are all known at compile time.

PyTorch's exporter routes `x.view(x.size(0), -1)` through Shape, Gather, Unsqueeze and Concat, so
folding is what makes the Reshape target static. The set is limited to what shape arithmetic and
quantization wrappers need; anything else with constant inputs simply runs at inference time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .errors import UnsupportedError
from .ir import Node

Array = np.ndarray[Any, Any]
Evaluator = Callable[[Node, list[Array | None]], Array]


def axes_from(node: Node, args: list[Array | None], position: int) -> list[int] | None:
    """`axes` moved from an attribute to an input in later opsets; accept both."""
    if "axes" in node.attrs:
        return [int(axis) for axis in node.attrs["axes"]]
    if len(args) > position and args[position] is not None:
        return [int(axis) for axis in np.asarray(args[position]).reshape(-1)]
    return None


def resolve_reshape(shape: tuple[int, ...], target: Array, allow_zero: bool) -> tuple[int, ...]:
    dims = [int(item) for item in target]
    for index, dim in enumerate(dims):
        if dim == 0 and not allow_zero:
            dims[index] = shape[index]
    if -1 in dims:
        known = int(np.prod([dim for dim in dims if dim != -1], dtype=np.int64))
        dims[dims.index(-1)] = int(np.prod(shape, dtype=np.int64)) // known
    return tuple(dims)


def slice_ranges(
    shape: tuple[int, ...],
    starts: Array,
    ends: Array,
    axes: Array | None,
    steps: Array | None,
) -> list[tuple[int, int, int]]:
    """Per-axis (start, count, step) after applying ONNX clamping rules."""
    ranges = [(0, dim, 1) for dim in shape]
    axis_list = [int(a) for a in (axes if axes is not None else range(len(starts)))]
    step_list = [int(s) for s in (steps if steps is not None else [1] * len(starts))]
    for start, end, axis, step in zip(starts, ends, axis_list, step_list, strict=True):
        dim = shape[axis]
        python_slice = slice(int(start), int(end), step)
        begin, stop, stride = python_slice.indices(dim)
        ranges[axis] = (begin, max(0, len(range(begin, stop, stride))), stride)
    return ranges


def _gather(node: Node, args: list[Array | None]) -> Array:
    data, indices = args
    assert data is not None and indices is not None
    return np.take(data, indices.astype(np.int64), axis=int(node.attrs.get("axis", 0)))


def _reshape(node: Node, args: list[Array | None]) -> Array:
    data, target = args[0], args[1]
    assert data is not None and target is not None
    allow_zero = bool(node.attrs.get("allowzero", 0))
    return data.reshape(resolve_reshape(data.shape, target, allow_zero))


def _flatten(node: Node, args: list[Array | None]) -> Array:
    data = args[0]
    assert data is not None
    axis = int(node.attrs.get("axis", 1))
    axis += data.ndim if axis < 0 else 0
    head = int(np.prod(data.shape[:axis], dtype=np.int64))
    return data.reshape(head, -1)


def _squeeze(node: Node, args: list[Array | None]) -> Array:
    data = args[0]
    assert data is not None
    axes = axes_from(node, args, 1)
    return np.squeeze(data, None if axes is None else tuple(axes))


def _unsqueeze(node: Node, args: list[Array | None]) -> Array:
    data = args[0]
    assert data is not None
    axes = axes_from(node, args, 1)
    assert axes is not None
    return np.expand_dims(data, tuple(axes))


def _transpose(node: Node, args: list[Array | None]) -> Array:
    data = args[0]
    assert data is not None
    return np.transpose(data, node.attrs.get("perm"))


def _cast(node: Node, args: list[Array | None]) -> Array:
    data = args[0]
    assert data is not None
    return data.astype(onnx.helper.tensor_dtype_to_np_dtype(node.attrs["to"]))


def _constant_of_shape(node: Node, args: list[Array | None]) -> Array:
    target = args[0]
    assert target is not None
    fill = node.attrs.get("value", np.zeros(1, np.float32))
    return np.full([int(d) for d in target], fill.reshape(-1)[0], dtype=fill.dtype)


def _range(node: Node, args: list[Array | None]) -> Array:
    start, limit, delta = args
    assert start is not None and limit is not None and delta is not None
    return np.arange(start.item(), limit.item(), delta.item(), dtype=start.dtype)


def _div(node: Node, args: list[Array | None]) -> Array:
    left, right = args
    assert left is not None and right is not None
    if np.issubdtype(left.dtype, np.integer):
        return np.trunc(left / right).astype(left.dtype)
    return left / right


def _slice(node: Node, args: list[Array | None]) -> Array:
    data, starts, ends = args[:3]
    assert data is not None and starts is not None and ends is not None
    axes = args[3] if len(args) > 3 else None
    steps = args[4] if len(args) > 4 else None
    ranges = slice_ranges(data.shape, starts, ends, axes, steps)
    index = tuple(
        np.arange(start, start + count * step, step, dtype=np.int64)
        for start, count, step in ranges
    )
    result: Array = data[np.ix_(*index)]
    return result


def _axis_shape(array: Array, axis: int, ndim: int) -> tuple[int, ...]:
    shape = [1] * ndim
    shape[axis] = array.size
    return tuple(shape)


def _quantize(node: Node, args: list[Array | None]) -> Array:
    data, scale = args[0], args[1]
    zero_point = args[2] if len(args) > 2 else None
    assert data is not None and scale is not None
    if zero_point is None:
        raise UnsupportedError(f"{node.name}: QuantizeLinear needs an explicit zero point")
    if scale.size > 1:
        axis = int(node.attrs.get("axis", 1)) % data.ndim
        scale = scale.reshape(_axis_shape(scale, axis, data.ndim))
        zero_point = zero_point.reshape(_axis_shape(zero_point, axis, data.ndim))
    info = np.iinfo(zero_point.dtype)
    rounded = np.rint(data / scale) + zero_point.astype(np.float32)
    result: Array = np.clip(rounded, info.min, info.max).astype(zero_point.dtype)
    return result


def _dequantize(node: Node, args: list[Array | None]) -> Array:
    data, scale = args[0], args[1]
    zero_point = args[2] if len(args) > 2 else None
    assert data is not None and scale is not None
    if zero_point is None:
        zero_point = np.zeros(1, data.dtype)
    if scale.size > 1:
        axis = int(node.attrs.get("axis", 1)) % data.ndim
        scale = scale.reshape(_axis_shape(scale, axis, data.ndim))
        zero_point = zero_point.reshape(_axis_shape(zero_point, axis, data.ndim))
    return ((data.astype(np.int32) - zero_point.astype(np.int32)) * scale).astype(np.float32)


def _binary(function: Callable[[Array, Array], Array]) -> Evaluator:
    def evaluate(node: Node, args: list[Array | None]) -> Array:
        left, right = args
        assert left is not None and right is not None
        return function(left, right).astype(left.dtype)

    return evaluate


def _concat(node: Node, args: list[Array | None]) -> Array:
    arrays = [item for item in args if item is not None]
    return np.concatenate(arrays, axis=int(node.attrs["axis"]))


FOLDERS: dict[str, Evaluator] = {
    "Add": _binary(np.add),
    "Cast": _cast,
    "Concat": _concat,
    "ConstantOfShape": _constant_of_shape,
    "DequantizeLinear": _dequantize,
    "Div": _div,
    "Flatten": _flatten,
    "Gather": _gather,
    "Identity": lambda node, args: np.asarray(args[0]),
    "Max": _binary(np.maximum),
    "Min": _binary(np.minimum),
    "Mul": _binary(np.multiply),
    "Neg": lambda node, args: -np.asarray(args[0]),
    "QuantizeLinear": _quantize,
    "Range": _range,
    "Reshape": _reshape,
    "Slice": _slice,
    "Squeeze": _squeeze,
    "Sub": _binary(np.subtract),
    "Transpose": _transpose,
    "Unsqueeze": _unsqueeze,
}
