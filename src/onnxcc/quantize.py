"""Post-training int8 quantization.

Activations use asymmetric per-tensor parameters and weights use symmetric per-channel parameters,
as TFLite and CMSIS-NN do. The result is plain ONNX (QLinearConv and friends), which means
onnxruntime can execute the same quantized graph and act as an independent reference for the C
kernels.

Calibration runs the float model through its own generated C with an observer hook, so the ranges
come from the same kernels that would ship, and the only thing it asks of the host is a C compiler.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CompileError, UnsupportedError
from .ir import F32, I8, I32, Graph, Node
from .passes import clip_bounds


@dataclass(frozen=True)
class QuantizeOptions:
    method: str = "percentile"  # or "minmax"
    percentile: float = 99.99
    per_channel: bool = True
    int8_io: bool = False
    # 127 uses the full int8 range. onnxruntime's AVX2 kernels sum pairs of u8*s8 products in 16
    # bits and can saturate at full range, so differential tests against it use 63.
    weight_limit: int = 127


@dataclass(frozen=True)
class QParams:
    scale: float
    zero_point: int


def activation_params(low: float, high: float) -> QParams:
    """Asymmetric range. Zero stays exactly representable, because padding means real zero."""
    low, high = min(low, 0.0), max(high, 0.0)
    if high == low:
        return QParams(1.0, 0)
    scale = float(np.float32((high - low) / 255.0))
    zero_point = int(np.clip(round(-128 - low / scale), -128, 127))
    return QParams(scale, zero_point)


def collect_ranges(
    graph: Graph, samples: Sequence[np.ndarray[Any, Any]], options: QuantizeOptions
) -> dict[str, tuple[float, float]]:
    from .compiler import emit_c
    from .toolchain import Binary, harness_source, link_harness

    result = emit_c(graph, "calibration", observe=True)
    harness = harness_source(
        result.c_name, result.prototype, result.inputs, result.outputs, observe=True
    )
    lows: dict[int, float] = {}
    highs: dict[int, float] = {}
    pools: dict[int, list[np.ndarray[Any, Any]]] = {}
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="onnxcc-calibration-") as directory:
        executable = link_harness(result.source, harness, Path(directory), "calibration", ["-O2"])
        binary = Binary(executable, result.inputs, result.outputs)
        for identifier, data in binary.stream_observations(samples):
            lows[identifier] = min(lows.get(identifier, np.inf), float(data.min()))
            highs[identifier] = max(highs.get(identifier, -np.inf), float(data.max()))
            if options.method == "percentile":
                keep = min(data.size, 512)
                pools.setdefault(identifier, []).append(rng.choice(data, keep, replace=False))
    ranges: dict[str, tuple[float, float]] = {}
    tail = 100.0 - options.percentile
    for identifier, name in result.observed.items():
        low, high = lows[identifier], highs[identifier]
        if options.method == "percentile":
            pool = np.concatenate(pools[identifier])
            low, high = (float(q) for q in np.percentile(pool, [tail, 100.0 - tail]))
        ranges[name] = (low, high)
    return ranges


class Quantizer:
    def __init__(
        self,
        graph: Graph,
        ranges: dict[str, tuple[float, float]],
        options: QuantizeOptions,
    ) -> None:
        self.graph = graph
        self.ranges = ranges
        self.options = options
        self.nodes: list[Node] = []
        self.qname: dict[str, str] = {}
        self.qparams: dict[str, QParams] = {}
        self.floats: set[str] = set(graph.inputs)

    def q_const(self, params: QParams) -> tuple[str, str]:
        scale = self.graph.add_const("scale", np.array(params.scale, F32))
        zero_point = self.graph.add_const("zero_point", np.array(params.zero_point, I8))
        return scale, zero_point

    def range_params(self, name: str) -> QParams:
        if name not in self.ranges:
            raise CompileError(f"calibration saw no data for {name!r}")
        return activation_params(*self.ranges[name])

    def to_int8(self, name: str) -> str:
        """Int8 twin of a float value, inserting a QuantizeLinear when it does not exist yet."""
        if name in self.qname:
            return self.qname[name]
        value = self.graph.values[name]
        if value.is_const:
            data = np.asarray(value.data, F32)
            params = activation_params(float(data.min()), float(data.max()))
            quantized = np.clip(np.rint(data / params.scale) + params.zero_point, -128, 127)
            twin = self.graph.add_const(f"{name}_q", quantized.astype(I8))
        else:
            params = self.range_params(name)
            twin = self.graph.new_value(f"{name}_q", I8, value.shape)
            scale, zero_point = self.q_const(params)
            self.nodes.append(Node("QuantizeLinear", [name, scale, zero_point], [twin], {}, name))
        self.qname[name], self.qparams[twin] = twin, params
        return twin

    def to_float(self, name: str) -> str:
        if name in self.floats or self.graph.values[name].is_const:
            return name
        twin = self.qname[name]
        scale, zero_point = self.q_const(self.qparams[twin])
        self.nodes.append(Node("DequantizeLinear", [twin, scale, zero_point], [name], {}, twin))
        self.floats.add(name)
        return name

    def define(self, name: str, params: QParams) -> str:
        twin = self.graph.new_value(f"{name}_q", I8, self.graph.values[name].shape)
        self.qname[name], self.qparams[twin] = twin, params
        return twin

    def weights(self, data: np.ndarray[Any, Any], axis: int) -> tuple[Any, Any]:
        """Symmetric int8 weights with one scale per slice along `axis`."""
        reduce_axes = tuple(item for item in range(data.ndim) if item != axis)
        peak = np.abs(data).max(axis=reduce_axes)
        if not self.options.per_channel:
            peak = np.full_like(peak, peak.max())
        limit = self.options.weight_limit
        scales = np.where(peak > 0.0, peak / limit, 1.0).astype(F32)
        view = [1] * data.ndim
        view[axis] = -1
        quantized = np.clip(np.rint(data / scales.reshape(view)), -limit, limit).astype(I8)
        return quantized, scales

    def bias(self, name: str, input_scale: float, weight_scales: Any) -> str:
        if not name:
            return ""
        data = self.graph.const(name)
        if data is None:
            raise UnsupportedError(f"bias {name!r} must be constant")
        scaled = np.rint(np.asarray(data, np.float64) / (input_scale * weight_scales))
        limits = np.iinfo(I32)
        return self.graph.add_const("bias_q", np.clip(scaled, limits.min, limits.max).astype(I32))

    def window_args(self, x: str, params: QParams) -> list[str]:
        twin = self.to_int8(x)
        out_scale, out_zero = self.q_const(params)
        in_scale, in_zero = self.q_const(self.qparams[twin])
        return [twin, in_scale, in_zero, out_scale, out_zero]

    def conv(self, node: Node, params: QParams) -> None:
        weights = self.graph.const(node.inputs[1])
        assert weights is not None
        x = self.to_int8(node.inputs[0])
        output = self.define(node.outputs[0], params)
        quantized, scales = self.weights(np.asarray(weights, F32), 0)
        x_scale, x_zero = self.q_const(self.qparams[x])
        y_scale, y_zero = self.q_const(params)
        w_name = self.graph.add_const("weights_q", quantized)
        w_scale = self.graph.add_const("w_scale", scales)
        w_zero = self.graph.add_const("w_zero", np.zeros(scales.size, I8))
        bias_name = node.inputs[2] if len(node.inputs) > 2 else ""
        bias = self.bias(bias_name, self.qparams[x].scale, np.asarray(scales, np.float64))
        inputs = [x, x_scale, x_zero, w_name, w_scale, w_zero, y_scale, y_zero]
        if bias:
            inputs.append(bias)
        attrs = {
            key: node.attrs[key]
            for key in ("dilations", "group", "kernel_shape", "pads", "strides")
        }
        self.nodes.append(Node("QLinearConv", inputs, [output], attrs, node.name))

    def gemm(self, node: Node, params: QParams) -> None:
        weights = self.graph.const(node.inputs[1])
        assert weights is not None
        a = self.to_int8(node.inputs[0])
        output = self.define(node.outputs[0], params)
        quantized, scales = self.weights(np.asarray(weights, F32), 1)
        a_scale, a_zero = self.q_const(self.qparams[a])
        y_scale, y_zero = self.q_const(params)
        w_name = self.graph.add_const("weights_q", quantized)
        w_scale = self.graph.add_const("w_scale", scales)
        w_zero = self.graph.add_const("w_zero", np.zeros(scales.size, I8))
        bias_name = node.inputs[2] if len(node.inputs) > 2 else ""
        bias = self.bias(bias_name, self.qparams[a].scale, np.asarray(scales, np.float64))
        inputs = [a, a_scale, a_zero, w_name, w_scale, w_zero, bias, y_scale, y_zero]
        self.nodes.append(Node("QGemm", inputs, [output], {}, node.name))

    def binary(self, node: Node, params: QParams) -> None:
        a = self.to_int8(node.inputs[0])
        b = self.to_int8(node.inputs[1])
        output = self.define(node.outputs[0], params)
        inputs = [a, *self.q_const(self.qparams[a]), b, *self.q_const(self.qparams[b])]
        inputs += self.q_const(params)
        self.nodes.append(Node(f"QLinear{node.op}", inputs, [output], {}, node.name))

    def average_pool(self, node: Node, params: QParams) -> None:
        inputs = self.window_args(node.inputs[0], params)
        output = self.define(node.outputs[0], params)
        keys = ("ceil_mode", "count_include_pad", "kernel_shape", "pads", "strides")
        attrs = {key: node.attrs[key] for key in keys if key in node.attrs}
        self.nodes.append(Node("QLinearAveragePool", inputs, [output], attrs, node.name))

    def global_pool(self, node: Node, params: QParams) -> None:
        inputs = self.window_args(node.inputs[0], params)
        output = self.define(node.outputs[0], params)
        self.nodes.append(Node("QLinearGlobalAveragePool", inputs, [output], {}, node.name))

    def concat(self, node: Node, params: QParams) -> None:
        names = [self.to_int8(item) for item in node.inputs]
        output = self.define(node.outputs[0], params)
        inputs = list(self.q_const(params))
        for name in names:
            inputs += [name, *self.q_const(self.qparams[name])]
        attrs = {"axis": node.attrs["axis"]}
        self.nodes.append(Node("QLinearConcat", inputs, [output], attrs, node.name))

    def passthrough(self, node: Node) -> None:
        """Ops that move values without changing them keep their input's parameters."""
        x = self.to_int8(node.inputs[0])
        params = self.qparams[x]
        output = self.define(node.outputs[0], params)
        inputs = [x, *node.inputs[1:]]
        attrs = dict(node.attrs)
        if node.op == "Pad":
            fill = self.graph.const(node.inputs[2]) if len(node.inputs) > 2 else None
            real = 0.0 if fill is None else float(fill)
            level = np.clip(round(real / params.scale) + params.zero_point, -128, 127)
            inputs = [x, node.inputs[1], self.graph.add_const("fill", np.array(level, I8))]
            inputs += node.inputs[3:]
        if node.op == "Clip":
            low, high = clip_bounds(self.graph, node)
            bounds = []
            for bound, fallback in ((low, -128), (high, 127)):
                level = fallback
                if bound is not None:
                    scaled = np.rint(bound / params.scale) + params.zero_point
                    level = int(np.clip(scaled, -128, 127))
                bounds.append(self.graph.add_const("bound", np.array(level, I8)))
            inputs = [x, *bounds]
        self.nodes.append(Node(node.op, inputs, [output], attrs, node.name))

    def absorbed_activations(self) -> dict[str, Node]:
        """Clips fed only by a requantizing op, which saturates to the same range anyway.

        The quantized range always contains zero, so saturation can stand in for a clip only when
        its bounds contain zero too. Clip(0.5, 1) has to stay a real operation.
        """
        absorbed = {}
        for node in self.graph.nodes:
            if node.op != "Clip":
                continue
            low, high = clip_bounds(self.graph, node)
            contains_zero = (low is None or low <= 0.0) and (high is None or high >= 0.0)
            producer = self.graph.producer(node.inputs[0])
            if (
                contains_zero
                and producer is not None
                and producer.op in REQUANTIZING
                and self.graph.sole_consumer(node.inputs[0]) is node
            ):
                absorbed[producer.outputs[0]] = node
        return absorbed

    def is_quantized_input(self, name: str) -> bool:
        return name in self.qname or name in self.graph.inputs

    def run(self) -> None:
        graph = self.graph
        absorbed = self.absorbed_activations()
        skipped = {id(node) for node in absorbed.values()}
        for node in graph.nodes:
            if id(node) in skipped:
                continue
            activation = absorbed.get(node.outputs[0])
            params_name = activation.outputs[0] if activation is not None else node.outputs[0]
            handled = True
            if node.op == "Conv" and graph.const(node.inputs[1]) is not None:
                self.conv(node, self.range_params(params_name))
            elif node.op == "Gemm" and graph.const(node.inputs[1]) is not None:
                self.gemm(node, self.range_params(params_name))
            elif node.op in ("Add", "Mul"):
                self.binary(node, self.range_params(params_name))
            elif node.op == "AveragePool":
                self.average_pool(node, self.range_params(params_name))
            elif node.op == "GlobalAveragePool":
                self.global_pool(node, self.range_params(params_name))
            elif node.op == "Concat":
                self.concat(node, self.range_params(params_name))
            elif node.op in PASSTHROUGH and self.is_quantized_input(node.inputs[0]):
                self.passthrough(node)
            else:
                handled = False
            if handled:
                if activation is not None:
                    self.qname[activation.outputs[0]] = self.qname[node.outputs[0]]
                continue
            node.inputs = [self.to_float(item) if item else item for item in node.inputs]
            self.nodes.append(node)
            self.floats.update(node.outputs)
        if self.options.int8_io:
            self.strip_io()
        else:
            for name in graph.outputs:
                self.to_float(name)
        graph.nodes = self.nodes

    def strip_io(self) -> None:
        graph = self.graph
        new_inputs = []
        for name in graph.inputs:
            quantize = next(
                (n for n in self.nodes if n.op == "QuantizeLinear" and n.inputs[0] == name), None
            )
            float_users = [n for n in self.nodes if name in n.inputs and n is not quantize]
            if quantize is None or float_users:
                raise UnsupportedError(f"input {name!r} is also consumed by a float operator")
            self.nodes.remove(quantize)
            new_inputs.append(quantize.outputs[0])
        graph.inputs = new_inputs
        new_outputs = []
        for name in graph.outputs:
            if name not in self.qname or name in self.floats:
                raise UnsupportedError(f"output {name!r} comes from a float operator")
            new_outputs.append(self.qname[name])
        graph.outputs = new_outputs


REQUANTIZING = frozenset(
    {"Add", "AveragePool", "Concat", "Conv", "Gemm", "GlobalAveragePool", "Mul"}
)
PASSTHROUGH = frozenset({"Clip", "MaxPool", "Pad", "Reshape", "Slice", "Transpose"})


def quantize_graph(
    graph: Graph, samples: Sequence[np.ndarray[Any, Any]], options: QuantizeOptions
) -> dict[str, QParams]:
    """Rewrite `graph` in place and return the interface tensors' quantization parameters."""
    if any(graph.values[name].dtype != F32 for name in graph.inputs):
        raise UnsupportedError("only float models can be quantized")
    if any(node.op.startswith("QLinear") or node.op == "QGemm" for node in graph.nodes):
        raise UnsupportedError("the model is already quantized")
    if len(samples) != len(graph.inputs):
        raise CompileError(f"need calibration data for {len(graph.inputs)} inputs")
    ranges = collect_ranges(graph, samples, options)
    quantizer = Quantizer(graph, ranges, options)
    quantizer.run()
    interface = {}
    for name in [*graph.inputs, *graph.outputs]:
        if name in quantizer.qparams:
            interface[name] = quantizer.qparams[name]
    return interface
