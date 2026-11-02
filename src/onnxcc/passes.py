"""Graph rewrites between import and memory planning.

`canonicalize` shrinks the operator set the backend must handle: every reshape-like op becomes
Reshape, Gemm and MatMul collapse to one bias-carrying matrix product, 1-D windows become 2-D, and
BatchNormalization disappears. Each lowering spends a little graph complexity to delete a kernel,
which is the right trade when the output has to fit in flash.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .errors import UnsupportedError
from .ir import F32, I64, Graph, Node
from .shapes import infer


def _shape_const(graph: Graph, shape: tuple[int, ...]) -> str:
    return graph.add_const("shape", np.array(shape, I64))


def _reshape(graph: Graph, node: Node, source: str, output: str) -> Node:
    target = graph.values[output].shape
    assert target is not None
    return Node("Reshape", [source, _shape_const(graph, target)], [output], {}, node.name)


def _lower_reshapelike(graph: Graph, node: Node) -> list[Node]:
    return [_reshape(graph, node, node.inputs[0], node.outputs[0])]


def _scale(graph: Graph, node: Node, name: str, factor: float) -> tuple[list[Node], str]:
    if factor == 1.0:
        return [], name
    data = graph.const(name)
    if data is not None:
        return [], graph.add_const("scaled", (data * factor).astype(F32))
    value = graph.values[name]
    out = graph.new_value("scaled", value.dtype, value.shape)
    factor_const = graph.add_const("factor", np.array(factor, F32))
    return [Node("Mul", [name, factor_const], [out], {}, node.name)], out


def _transposed(graph: Graph, node: Node, name: str) -> tuple[list[Node], str]:
    data = graph.const(name)
    if data is not None:
        return [], graph.add_const("transposed", data.T)
    value = graph.values[name]
    assert value.shape is not None
    out = graph.new_value("transposed", value.dtype, value.shape[::-1])
    return [Node("Transpose", [name], [out], {"perm": [1, 0]}, node.name)], out


def _lower_gemm(graph: Graph, node: Node) -> list[Node]:
    """Rewrite to `A @ B + C` with no transposes, and C either a 1-D constant or absent."""
    attrs = node.attrs
    a, b = node.inputs[0], node.inputs[1]
    c = node.inputs[2] if len(node.inputs) > 2 else ""
    nodes: list[Node] = []
    if attrs.get("transA", 0):
        extra, a = _transposed(graph, node, a)
        nodes += extra
    if attrs.get("transB", 0):
        extra, b = _transposed(graph, node, b)
        nodes += extra
    extra, b = _scale(graph, node, b, float(attrs.get("alpha", 1.0)))
    nodes += extra
    output = node.outputs[0]
    out_value = graph.values[output]
    assert out_value.shape is not None
    bias = ""
    tail: list[Node] = []
    if c:
        extra, c = _scale(graph, node, c, float(attrs.get("beta", 1.0)))
        nodes += extra
        data = graph.const(c)
        columns = out_value.shape[1]
        if data is not None:
            if data.shape == (1, columns):
                data = data[0]
            elif data.size == 1:
                data = np.full(columns, data.reshape(-1)[0], data.dtype)
        if data is not None and data.shape == (columns,):
            bias = graph.add_const("bias", data)
        else:
            product = graph.new_value("product", out_value.dtype, out_value.shape)
            tail = [Node("Add", [product, c], [output], {}, node.name)]
            output = product
    inputs = [a, b, bias] if bias else [a, b]
    nodes.append(Node("Gemm", inputs, [output], {}, node.name))
    return nodes + tail


def _lower_matmul(graph: Graph, node: Node) -> list[Node]:
    a, b = node.inputs
    a_value, b_value = graph.values[a], graph.values[b]
    assert a_value.shape is not None and b_value.shape is not None
    out_shape = graph.values[node.outputs[0]].shape
    columns = b_value.shape[1] if len(b_value.shape) > 1 else 1
    rows = a_value.size // b_value.shape[0]
    nodes: list[Node] = []
    if len(a_value.shape) != 2:
        flat = graph.new_value("rows", a_value.dtype, (rows, b_value.shape[0]))
        nodes.append(_reshape(graph, node, a, flat))
        a = flat
    if len(b_value.shape) != 2:
        tall = graph.new_value("columns", b_value.dtype, (b_value.shape[0], 1))
        nodes.append(_reshape(graph, node, b, tall))
        b = tall
    if out_shape == (rows, columns):
        return [*nodes, Node("Gemm", [a, b], list(node.outputs), {}, node.name)]
    product = graph.new_value("product", a_value.dtype, (rows, columns))
    nodes.append(Node("Gemm", [a, b], [product], {}, node.name))
    return [*nodes, _reshape(graph, node, product, node.outputs[0])]


def _lower_relu(graph: Graph, node: Node) -> list[Node]:
    low = graph.add_const("zero", np.array(0.0, F32))
    return [Node("Clip", [node.inputs[0], low], list(node.outputs), {}, node.name)]


def _lower_batch_norm(graph: Graph, node: Node) -> list[Node]:
    """A BatchNormalization left unfused is `x * gain + offset` with per-channel constants."""
    x, scale, bias, mean, variance = node.inputs[:5]
    value = graph.values[x]
    assert value.shape is not None
    constants = [graph.const(item) for item in (scale, bias, mean, variance)]
    if any(item is None for item in constants):
        raise UnsupportedError(f"{node.name}: BatchNormalization parameters must be constant")
    gamma, beta, mu, var = (np.asarray(item, F32) for item in constants)
    gain = gamma / np.sqrt(var + float(node.attrs.get("epsilon", 1e-5)))
    offset = beta - mu * gain
    broadcast_shape = (-1, *[1] * (len(value.shape) - 2))
    gain_name = graph.add_const("gain", gain.reshape(broadcast_shape))
    offset_name = graph.add_const("offset", offset.reshape(broadcast_shape))
    scaled = graph.new_value("scaled", value.dtype, value.shape)
    return [
        Node("Mul", [x, gain_name], [scaled], {}, node.name),
        Node("Add", [scaled, offset_name], list(node.outputs), {}, node.name),
    ]


def _lower_split(graph: Graph, node: Node) -> list[Node]:
    axis, sizes = node.attrs["axis"], node.attrs["resolved_split"]
    nodes = []
    start = 0
    for output, size in zip(node.outputs, sizes, strict=True):
        constants = [
            graph.add_const("slice", np.array([item], I64)) for item in (start, start + size, axis)
        ]
        nodes.append(Node("Slice", [node.inputs[0], *constants], [output], {}, node.name))
        start += size
    return nodes


def _lower_variadic(graph: Graph, node: Node) -> list[Node]:
    if len(node.inputs) == 1:
        return [_reshape(graph, node, node.inputs[0], node.outputs[0])]
    nodes = []
    current = node.inputs[0]
    for position, item in enumerate(node.inputs[1:], start=1):
        last = position == len(node.inputs) - 1
        output = node.outputs[0] if last else graph.new_value("partial")
        nodes.append(Node(node.op, [current, item], [output], {}, node.name))
        current = output
    return nodes


def _lower_reduce_mean(graph: Graph, node: Node) -> list[Node]:
    value = graph.values[node.inputs[0]]
    assert value.shape is not None
    if len(value.shape) != 4 or node.attrs["resolved_axes"] != [2, 3]:
        return [node]
    if int(node.attrs.get("keepdims", 1)):
        return [Node("GlobalAveragePool", node.inputs[:1], list(node.outputs), {}, node.name)]
    kept = graph.new_value("pooled", value.dtype, (*value.shape[:2], 1, 1))
    pool = Node("GlobalAveragePool", node.inputs[:1], [kept], {}, node.name)
    return [pool, _reshape(graph, node, kept, node.outputs[0])]


def _lift_1d(graph: Graph, node: Node) -> list[Node]:
    """Route rank-3 windows through the 2-D kernels. Reshapes alias, so this costs no memory."""
    value = graph.values[node.inputs[0]]
    assert value.shape is not None
    if len(value.shape) != 3:
        return [node]
    n, c, length = value.shape
    out_value = graph.values[node.outputs[0]]
    assert out_value.shape is not None
    lifted = graph.new_value("lifted", value.dtype, (n, c, 1, length))
    out_n, out_c, out_length = out_value.shape
    product = graph.new_value("lifted", value.dtype, (out_n, out_c, 1, out_length))
    attrs: dict[str, Any] = dict(node.attrs)
    attrs["strides"] = [1, attrs["strides"][0]]
    attrs["dilations"] = [1, attrs["dilations"][0]]
    attrs["kernel_shape"] = [1, attrs["kernel_shape"][0]]
    attrs["pads"] = [0, attrs["pads"][0], 0, attrs["pads"][1]]
    inputs = [lifted, *node.inputs[1:]]
    if node.op == "Conv":
        weights = graph.const(node.inputs[1])
        if weights is None:
            raise UnsupportedError(f"{node.name}: Conv weights must be constant")
        inputs[1] = graph.add_const("lifted_w", weights[:, :, None, :])
    return [
        _reshape(graph, node, node.inputs[0], lifted),
        Node(node.op, inputs, [product], attrs, node.name),
        _reshape(graph, node, product, node.outputs[0]),
    ]


def _lower_global_pool(graph: Graph, node: Node) -> list[Node]:
    value = graph.values[node.inputs[0]]
    assert value.shape is not None
    if len(value.shape) == 4:
        return [node]
    if len(value.shape) != 3:
        raise UnsupportedError(f"{node.name}: GlobalAveragePool supports 1-D and 2-D spatial data")
    n, c, length = value.shape
    lifted = graph.new_value("lifted", value.dtype, (n, c, 1, length))
    pooled = graph.new_value("lifted", value.dtype, (n, c, 1, 1))
    return [
        _reshape(graph, node, node.inputs[0], lifted),
        Node(node.op, [lifted], [pooled], {}, node.name),
        _reshape(graph, node, pooled, node.outputs[0]),
    ]


LOWERINGS: dict[str, Callable[[Graph, Node], list[Node]]] = {
    "AveragePool": _lift_1d,
    "BatchNormalization": _lower_batch_norm,
    "Cast": _lower_reshapelike,
    "Conv": _lift_1d,
    "Dropout": _lower_reshapelike,
    "Flatten": _lower_reshapelike,
    "Gemm": _lower_gemm,
    "GlobalAveragePool": _lower_global_pool,
    "Identity": _lower_reshapelike,
    "MatMul": _lower_matmul,
    "Max": _lower_variadic,
    "MaxPool": _lift_1d,
    "Min": _lower_variadic,
    "ReduceMean": _lower_reduce_mean,
    "Relu": _lower_relu,
    "Split": _lower_split,
    "Squeeze": _lower_reshapelike,
    "Unsqueeze": _lower_reshapelike,
}


def _isolate_outputs(graph: Graph) -> None:
    """Give every graph output a distinct runtime producer so it maps to one `out` argument."""
    seen: set[str] = set()
    for index, name in enumerate(graph.outputs):
        value = graph.values[name]
        if name not in seen and not value.is_const and name not in graph.inputs:
            seen.add(name)
            continue
        fresh = graph.new_value("output", value.dtype, value.shape)
        assert value.shape is not None
        graph.nodes.append(
            Node(
                "Reshape",
                [name, _shape_const(graph, value.shape)],
                [fresh],
                {"materialize": 1},
                "output_copy",
            )
        )
        graph.outputs[index] = fresh
        seen.add(fresh)


def fold_batch_norm(graph: Graph) -> None:
    """Merge BatchNormalization into the preceding Conv or dense layer's weights and bias."""
    for node in list(graph.nodes):
        if node.op != "BatchNormalization":
            continue
        producer = graph.producer(node.inputs[0])
        if (
            producer is None
            or producer.op not in ("Conv", "Gemm")
            or graph.sole_consumer(node.inputs[0]) is not node
        ):
            continue
        constants = [graph.const(item) for item in node.inputs[1:5]]
        weights = graph.const(producer.inputs[1])
        if weights is None or any(item is None for item in constants):
            continue
        if producer.op == "Gemm" and producer.attrs:
            continue
        gamma, beta, mu, var = (np.asarray(item, F32) for item in constants)
        gain = gamma / np.sqrt(var + float(node.attrs.get("epsilon", 1e-5)))
        if producer.op == "Conv":
            shape = (-1, *[1] * (weights.ndim - 1))
            new_weights = weights * gain.reshape(shape)
        else:
            new_weights = weights * gain
        bias = graph.const(producer.inputs[2]) if len(producer.inputs) > 2 else None
        old_bias = np.zeros_like(gain) if bias is None else np.asarray(bias, F32)
        new_bias = (old_bias - mu) * gain + beta
        producer.inputs = [
            producer.inputs[0],
            graph.add_const("bn_weights", new_weights.astype(F32)),
            graph.add_const("bn_bias", new_bias.astype(F32)),
        ]
        graph.replace_uses(node.outputs[0], producer.outputs[0])
        if node.outputs[0] in graph.outputs:
            graph.outputs[graph.outputs.index(node.outputs[0])] = producer.outputs[0]
        graph.nodes.remove(node)


def canonicalize(graph: Graph) -> None:
    infer(graph)
    # Gemm must already be free of transposes before its BatchNormalization merges.
    graph.nodes = [
        replacement
        for node in graph.nodes
        for replacement in (_lower_gemm(graph, node) if node.op == "Gemm" else [node])
    ]
    infer(graph)
    fold_batch_norm(graph)
    _isolate_outputs(graph)
    graph.nodes = [
        replacement
        for node in graph.nodes
        for replacement in (
            LOWERINGS[node.op](graph, node)
            if node.op in LOWERINGS and node.op != "Gemm"
            else [node]
        )
    ]
    infer(graph)
    graph.eliminate_dead_code()


def clip_bounds(graph: Graph, node: Node) -> tuple[float | None, float | None]:
    low = graph.const(node.inputs[1]) if len(node.inputs) > 1 else None
    high = graph.const(node.inputs[2]) if len(node.inputs) > 2 else None
    lower = None if low is None else float(low)
    upper = None if high is None else float(high)
    if lower is not None and upper is not None and lower > upper:
        lower = upper  # ONNX: with min > max every element becomes max
    return lower, upper


def fuse_activations(graph: Graph) -> None:
    """Fold a Clip into the producing float Conv or Gemm so the clamp runs inside its kernel."""
    for node in list(graph.nodes):
        if node.op != "Clip" or graph.values[node.inputs[0]].dtype != F32:
            continue
        producer = graph.producer(node.inputs[0])
        if (
            producer is None
            or producer.op not in ("Conv", "Gemm")
            or "clamp" in producer.attrs
            or graph.sole_consumer(node.inputs[0]) is not node
        ):
            continue
        producer.attrs["clamp"] = clip_bounds(graph, node)
        producer.outputs = list(node.outputs)
        graph.nodes.remove(node)
