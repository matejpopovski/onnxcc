"""Emit one C99 translation unit from a planned graph.

Every statement in the run function carries a machine-readable `/*@ ... */` line naming the regions
it reads and writes. `memcheck` verifies the plan from those lines and the statements they annotate,
without importing anything from the compiler.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

from .errors import CompileError, UnsupportedError
from .fixedpoint import add_multipliers, multiplier
from .ir import F32, I8, I32, Graph, Node, Value
from .liveness import Liveness, arena_of, data_inputs, is_alias
from .passes import clip_bounds
from .planner import Plan
from .shapes import Shape

C_TYPES: dict[np.dtype[Any], str] = {F32: "float", I8: "int8_t", I32: "int32_t"}
ACCUMULATOR_LIMIT = 2**31 - 1
MAX_RANK = 8


@dataclass
class Kernel:
    name: str
    requires: tuple[str, ...]
    generic: bool
    body: str


def load_kernels() -> dict[str, Kernel]:
    text = resources.files(__package__).joinpath("kernels.c").read_text()
    pattern = re.compile(r"/\*@kernel (\w+)(.*?)\*/\n(.*?)/\*@end\*/\n", re.S)
    kernels: dict[str, Kernel] = {}
    for name, options, body in pattern.findall(text):
        requires = re.search(r"requires=(\S+)", options)
        kernels[name] = Kernel(
            name,
            tuple(requires.group(1).split(",")) if requires else (),
            "generic" in options,
            body,
        )
    return kernels


KERNELS = load_kernels()


def float_literal(value: float) -> str:
    """Exact C99 hex literal for a float32, because weights must round-trip bit for bit."""
    if not np.isfinite(value):
        raise CompileError(f"non-finite constant {value} has no C99 literal without libm")
    number = float(value)
    if number == 0.0:
        return "-0x0p+0f" if np.signbit(number) else "0x0p+0f"
    mantissa, exponent = number.hex().split("p")
    return f"{mantissa.rstrip('0').rstrip('.')}p{exponent}f"


def int_literal(value: int) -> str:
    return "(-2147483647 - 1)" if value == -(2**31) else str(value)


def comment_text(text: str) -> str:
    """ONNX names are arbitrary strings. Keep comments free of terminators and trigraphs."""
    return re.sub(r"[^A-Za-z0-9_.:/-]", "_", text.replace("*/", "_"))[:80]


def c_identifier(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", text)
    return f"m_{name}" if not name or name[0].isdigit() else name


def _strides(dims: Sequence[int]) -> list[int]:
    strides = [1] * len(dims)
    for axis in range(len(dims) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * dims[axis + 1]
    return strides


def _collapse(dims: list[int], *strides: list[int]) -> tuple[list[int], list[list[int]]]:
    """Merge neighbouring axes that are contiguous in every operand to shorten the loop nest."""
    out_dims = [dims[0]]
    out_strides = [[s[0]] for s in strides]
    for axis in range(1, len(dims)):
        mergeable = all(
            s[-1] == t[axis] * dims[axis] for s, t in zip(out_strides, strides, strict=True)
        )
        if mergeable:
            out_dims[-1] *= dims[axis]
            for merged, source in zip(out_strides, strides, strict=True):
                merged[-1] = source[axis]
        else:
            out_dims.append(dims[axis])
            for merged, source in zip(out_strides, strides, strict=True):
                merged.append(source[axis])
    return out_dims, out_strides


def _prod(dims: Sequence[int]) -> int:
    return int(np.prod(dims, dtype=np.int64))


@dataclass
class Signature:
    name: str
    c_type: str
    shape: Shape
    elements: int


class Emitter:
    def __init__(
        self,
        graph: Graph,
        liveness: Liveness,
        plans: dict[str, Plan],
        name: str,
        isolated: bool = False,
        observe: bool = False,
    ) -> None:
        self.graph = graph
        self.liveness = liveness
        self.plans = plans
        self.name = c_identifier(name)
        self.isolated = isolated
        self.observe = observe
        self.kernels: set[str] = {"prelude"}
        self.constants: dict[tuple[str, bytes], str] = {}
        self.const_lines: list[str] = []
        self.param_lines: list[str] = []
        self.counter = 0
        self.value_ids: dict[str, str] = {}
        self.observed: dict[int, str] = {}
        self.constant_bytes = 0
        self.scratch_elements = 0

    def label(self, name: str) -> str:
        return self.value_ids.setdefault(name, f"v{len(self.value_ids)}")

    def unique(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def _location(self, name: str) -> tuple[str, int | None]:
        """(symbol, element offset). Offset None means a caller argument or an isolated array."""
        graph = self.graph
        owner = self.liveness.storage[name]
        if owner in graph.inputs:
            return f"in{graph.inputs.index(owner)}", None
        if owner in graph.outputs:
            return f"out{graph.outputs.index(owner)}", None
        if self.isolated:
            return f"T_{self.label(owner)}", None
        arena = arena_of(graph, owner)
        return arena, self.plans[arena].offsets[owner]

    def ref(self, name: str) -> str:
        value = self.graph.values[name]
        if value.is_const:
            return self.array(np.asarray(value.data))
        symbol, offset = self._location(name)
        return symbol if offset is None else f"{symbol} + {offset}"

    def access(self, mode: str, name: str) -> str:
        symbol, offset = self._location(name)
        size = self.graph.values[name].size
        start = 0 if offset is None else offset
        return f"{mode}={symbol}[{start}:{start + size}]:{self.label(name)}"

    def array(self, data: np.ndarray[Any, Any], c_type: str | None = None) -> str:
        data = np.atleast_1d(data)
        dtype = data.dtype
        if c_type is None:
            if dtype not in C_TYPES:
                raise UnsupportedError(f"constants of type {dtype} cannot reach a runtime kernel")
            c_type = C_TYPES[dtype]
        key = (c_type, data.tobytes())
        if key in self.constants:
            return self.constants[key]
        symbol = self.unique("W")
        self.constants[key] = symbol
        self.constant_bytes += data.nbytes
        flat = data.reshape(-1).tolist()
        if dtype == F32:
            items = [float_literal(item) for item in flat]
        else:
            items = [int_literal(int(item)) for item in flat]
        body = textwrap.fill(", ".join(items), 96, initial_indent="    ", subsequent_indent="    ")
        self.const_lines.append(f"static const {c_type} {symbol}[{len(flat)}] = {{\n{body}\n}};")
        return symbol

    def int_array(self, items: Sequence[int]) -> str:
        return self.array(np.array(list(items), np.int32))

    def param(self, c_type: str, initializer: str) -> str:
        symbol = self.unique("P")
        self.param_lines.append(f"static const {c_type} {symbol} = {{{initializer}}};")
        return f"&{symbol}"

    def need(self, kernel: str) -> str:
        self.kernels.add(kernel)
        return f"ox_{kernel}"

    def suffix(self, name: str) -> str:
        return "s8" if self.graph.values[name].dtype == I8 else "f32"

    def walk(self, dims: Sequence[int], strides: Sequence[int], start: int = 0) -> str:
        self.kernels.add("walk_t")
        merged, [merged_strides] = _collapse(list(dims) or [1], list(strides) or [0])
        if len(merged) > MAX_RANK:
            raise UnsupportedError(f"rank {len(merged)} exceeds the kernel limit of {MAX_RANK}")
        d = self.int_array(merged)
        s = self.int_array(merged_strides)
        return self.param("ox_walk_t", f"{len(merged)}, {start}, {d}, {s}")

    def window(self, node: Node, x_name: str, y_name: str) -> str:
        self.kernels.add("window_t")
        attrs = node.attrs
        x, y = self.graph.values[x_name], self.graph.values[y_name]
        assert x.shape is not None and y.shape is not None
        if len(x.shape) != 4:
            raise UnsupportedError(f"{node.name}: quantized 1-D windows are not supported")
        k_h, k_w = attrs["kernel_shape"]
        s_h, s_w = attrs["strides"]
        d_h, d_w = attrs["dilations"]
        p_t, p_l = attrs["pads"][0], attrs["pads"][1]
        fields = [
            x.shape[0], x.shape[1], x.shape[2], x.shape[3], y.shape[1], y.shape[2], y.shape[3],
            k_h, k_w, s_h, s_w, d_h, d_w, p_t, p_l, attrs.get("group", 1),
        ]  # fmt: skip
        return self.param("ox_window_t", ", ".join(str(item) for item in fields))

    def bcast(self, a: Value, b: Value, y: Value) -> str:
        self.kernels.add("bcast_t")
        assert a.shape is not None and b.shape is not None and y.shape is not None
        dims = list(y.shape) or [1]
        rank = len(dims)

        def aligned(shape: Shape) -> list[int]:
            full = [1] * (rank - len(shape)) + list(shape)
            strides = _strides(full)
            return [0 if size == 1 else stride for size, stride in zip(full, strides, strict=True)]

        merged, [sa, sb] = _collapse(dims, aligned(a.shape), aligned(b.shape))
        if len(merged) > MAX_RANK:
            raise UnsupportedError(f"rank {len(merged)} exceeds the kernel limit of {MAX_RANK}")
        fields = [self.int_array(merged), self.int_array(sa), self.int_array(sb)]
        return self.param("ox_bcast_t", f"{len(merged)}, {', '.join(fields)}")

    def scalar(self, name: str) -> Any:
        data = self.graph.const(name)
        if data is None:
            raise UnsupportedError(f"{name!r} must be a compile-time constant")
        return data.reshape(-1)[0]

    def statement(self, node: Node) -> str:
        emit = EMITTERS.get(node.op)
        if emit is None:
            raise CompileError(f"no code generator for {node.op}")
        return emit(self, node)

    def run_body(self) -> list[str]:
        graph = self.graph
        lines: list[str] = []
        if self.observe:
            for name in graph.inputs:
                lines.extend(self._observation(name))
        steps = {id(node): index for index, node in enumerate(self.liveness.steps)}
        for node in graph.nodes:
            if is_alias(graph, node):
                lines.append(
                    f"/*@ alias {self.label(node.outputs[0])} {self.label(node.inputs[0])} "
                    f"{graph.values[node.outputs[0]].size} {graph.values[node.inputs[0]].size} "
                    "Reshape */"
                )
                continue
            index = steps[id(node)]
            output = node.outputs[0]
            accesses = [self.access("w", output)]
            for item in dict.fromkeys(data_inputs(graph, node)):
                accesses.append(self.access("r", item))
            statement = self.statement(node)
            lines.append(f"/* {comment_text(node.name)} */")
            lines.append(f"/*@ {index} {node.op} {' '.join(accesses)} */")
            lines.append(statement)
            if self.observe:
                lines.extend(self._observation(output))
        return lines

    def _observation(self, name: str) -> list[str]:
        value = self.graph.values[name]
        if value.dtype != F32:
            return []
        identifier = len(self.observed)
        self.observed[identifier] = name
        return [f"onnxcc_observe({identifier}, {self.ref(name)}, {value.size});"]

    def arena_lines(self) -> list[str]:
        lines = [f"static int32_t S[{self.scratch_elements}];"] if self.scratch_elements else []
        if self.isolated:
            for arena, buffers in self.liveness.buffers.items():
                c_type = "float" if arena == "F" else "int8_t"
                for buffer in buffers:
                    lines.append(f"static {c_type} T_{self.label(buffer.key)}[{buffer.size}];")
            return lines
        if self.plans["F"].size:
            lines.append(f"static float F[{self.plans['F'].size}];")
        if self.plans["Q"].size:
            lines.append(f"static int8_t Q[{self.plans['Q'].size}];")
        return lines

    def signature(self) -> tuple[list[Signature], list[Signature]]:
        def describe(prefix: str, names: list[str]) -> list[Signature]:
            items = []
            for index, name in enumerate(names):
                value = self.graph.values[name]
                assert value.dtype is not None and value.shape is not None
                items.append(
                    Signature(f"{prefix}{index}", C_TYPES[value.dtype], value.shape, value.size)
                )
            return items

        return describe("in", self.graph.inputs), describe("out", self.graph.outputs)

    def describe_io(self) -> list[str]:
        inputs, outputs = self.signature()
        names = [*self.graph.inputs, *self.graph.outputs]
        lines = []
        for item, name in zip([*inputs, *outputs], names, strict=True):
            shape = "x".join(str(dim) for dim in item.shape) or "scalar"
            lines.append(f"{item.name}: {item.c_type}[{item.elements}], tensor {shape}")
            lines[-1] += f", onnx value {comment_text(name)}"
        return lines

    def prototype(self) -> str:
        inputs, outputs = self.signature()
        args = [f"const {item.c_type} *restrict {item.name}" for item in inputs]
        args += [f"{item.c_type} *restrict {item.name}" for item in outputs]
        return f"void {self.name}_run({', '.join(args) or 'void'})"

    def render_kernels(self) -> str:
        needed: dict[str, None] = {}

        def visit(name: str) -> None:
            base = name
            if name not in KERNELS:
                base = name.replace("_s8", "_f32")
            kernel = KERNELS[base]
            for item in kernel.requires:
                visit(item)
            needed.setdefault(name)

        for name in sorted(self.kernels):
            visit(name)
        order = {name: index for index, name in enumerate(KERNELS)}
        chunks = []

        def position(name: str) -> tuple[int, str]:
            base = name if name in KERNELS else name.replace("_s8", "_f32")
            return order[base], name

        for name in sorted(needed, key=position):
            if name in KERNELS:
                chunks.append(KERNELS[name].body)
            else:
                body = KERNELS[name.replace("_s8", "_f32")].body
                chunks.append(body.replace("_f32", "_s8").replace("float", "int8_t"))
        return "\n".join(chunks)

    def render(self, banner: Sequence[str], body: list[str]) -> str:
        arenas = self.arena_lines()
        sections = ["/* " + banner[0], *[" * " + line if line else " *" for line in banner[1:]]]
        sections.append(" */")
        text = "\n".join(line.rstrip() for line in sections) + "\n\n"
        text += self.render_kernels() + "\n"
        if self.observe:
            text += "void onnxcc_observe(int32_t id, const float *data, int32_t n);\n\n"
        if self.const_lines:
            text += "\n".join(self.const_lines) + "\n\n"
        if self.param_lines:
            text += "\n".join(self.param_lines) + "\n\n"
        if arenas:
            text += "\n".join(arenas) + "\n\n"
        text += f"{self.prototype()};\n\n{self.prototype()}\n{{\n"
        unused = self._unused_arguments(body)
        text += "".join(f"    (void){item};\n" for item in unused)
        text += "".join(f"    {line}\n" for line in body)
        text += "}\n"
        return text

    def _unused_arguments(self, body: list[str]) -> list[str]:
        inputs, _ = self.signature()
        joined = "\n".join(line for line in body if not line.startswith("/*"))
        return [item.name for item in inputs if not re.search(rf"\b{item.name}\b", joined)]


def _clamp_literals(node: Node) -> str:
    bounds = node.attrs.get("clamp")
    if bounds is None:
        return "1.0f, 0.0f"
    low, high = bounds
    largest = float(np.finfo(np.float32).max)
    return (
        f"{float_literal(-largest if low is None else low)}, "
        f"{float_literal(largest if high is None else high)}"
    )


def _optional(emitter: Emitter, node: Node, position: int) -> str:
    if len(node.inputs) > position and node.inputs[position]:
        return emitter.ref(node.inputs[position])
    return "0"


def _conv(e: Emitter, node: Node) -> str:
    x, w = node.inputs[0], node.inputs[1]
    y = node.outputs[0]
    if e.graph.const(w) is None:
        raise UnsupportedError(f"{node.name}: Conv weights must be constant")
    bias = _optional(e, node, 2)
    fn = e.need("conv2d_f32")
    params = e.window(node, x, y)
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {e.ref(w)}, {bias}, {params}, {_clamp_literals(node)});"


def _gemm(e: Emitter, node: Node) -> str:
    a, b = node.inputs[0], node.inputs[1]
    y = node.outputs[0]
    shape_a, shape_y = e.graph.values[a].shape, e.graph.values[y].shape
    assert shape_a is not None and shape_y is not None
    bias = _optional(e, node, 2)
    fn = e.need("dense_f32")
    return (
        f"{fn}({e.ref(y)}, {e.ref(a)}, {e.ref(b)}, {bias}, {shape_a[0]}, {shape_a[1]}, "
        f"{shape_y[1]}, {_clamp_literals(node)});"
    )


UNARY = {
    "Abs": "abs_f32",
    "Exp": "exp_f32",
    "HardSwish": "hard_swish_f32",
    "Neg": "neg_f32",
    "Sigmoid": "sigmoid_f32",
    "Tanh": "tanh_f32",
}


def _unary(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    fn = e.need(UNARY[node.op])
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {e.graph.values[y].size}, 0.0f, 0.0f);"


def _leaky_relu(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    fn = e.need("leaky_relu_f32")
    alpha = float_literal(float(np.float32(node.attrs.get("alpha", 0.01))))
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {e.graph.values[y].size}, {alpha}, 0.0f);"


def _hard_sigmoid(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    fn = e.need("hard_sigmoid_f32")
    alpha = float_literal(float(np.float32(node.attrs.get("alpha", 0.2))))
    beta = float_literal(float(np.float32(node.attrs.get("beta", 0.5))))
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {e.graph.values[y].size}, {alpha}, {beta});"


def _clip(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    low, high = clip_bounds(e.graph, node)
    size = e.graph.values[y].size
    if e.graph.values[x].dtype == I8:
        lo = -128 if low is None else int(low)
        hi = 127 if high is None else int(high)
        return f"{e.need('clip_s8')}({e.ref(y)}, {e.ref(x)}, {size}, {lo}, {hi});"
    largest = float(np.finfo(np.float32).max)
    lo_text = float_literal(-largest if low is None else low)
    hi_text = float_literal(largest if high is None else high)
    return f"{e.need('clip_f32')}({e.ref(y)}, {e.ref(x)}, {size}, {lo_text}, {hi_text});"


BINARY = {
    "Add": "add_f32",
    "Div": "div_f32",
    "Max": "max_f32",
    "Min": "min_f32",
    "Mul": "mul_f32",
    "PRelu": "prelu_f32",
    "Sub": "sub_f32",
}


def _binary(e: Emitter, node: Node) -> str:
    a, b = node.inputs[0], node.inputs[1]
    y = node.outputs[0]
    values = e.graph.values
    for item in (a, b):
        if values[item].dtype != F32:
            raise UnsupportedError(f"{node.name}: {node.op} runs on float32 only at runtime")
    params = e.bcast(values[a], values[b], values[y])
    return f"{e.need(BINARY[node.op])}({e.ref(y)}, {e.ref(a)}, {e.ref(b)}, {params}, 0);"


def _softmax(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    shape = e.graph.values[x].shape
    assert shape is not None
    axis = node.attrs["axis"]
    outer, inner = _prod(shape[:axis]), _prod(shape[axis + 1 :])
    fn = e.need("softmax_f32")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {outer}, {shape[axis]}, {inner});"


def _pool(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    params = e.window(node, x, y)
    if node.op == "MaxPool":
        fn = e.need(f"maxpool_{e.suffix(x)}")
        return f"{fn}({e.ref(y)}, {e.ref(x)}, {params});"
    include = int(node.attrs.get("count_include_pad", 0))
    return f"{e.need('avgpool_f32')}({e.ref(y)}, {e.ref(x)}, {params}, {include});"


def _global_pool(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    shape = e.graph.values[x].shape
    assert shape is not None
    fn = e.need("gap_f32")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {shape[0] * shape[1]}, {shape[2] * shape[3]});"


def _reduce(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    shape = e.graph.values[x].shape
    assert shape is not None
    axes = node.attrs["resolved_axes"]
    dims = list(shape) or [1]
    kept = [dim for axis, dim in enumerate(dims) if axis not in axes]
    kept_strides = iter(_strides(kept))
    strides = [0 if axis in axes else next(kept_strides) for axis in range(len(dims))]
    divisor = _prod([dims[axis] for axis in axes]) if node.op == "ReduceMean" else 1
    params = e.walk(dims, strides)
    size = e.graph.values[y].size
    fn = e.need("reduce_f32")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {params}, {size}, {float_literal(float(divisor))});"


def _copy(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    fn = e.need(f"copy_{e.suffix(y)}")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {e.graph.values[y].size});"


def _transpose(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    shape = e.graph.values[x].shape
    assert shape is not None
    perm = node.attrs["perm"]
    source = _strides(shape)
    params = e.walk([shape[axis] for axis in perm], [source[axis] for axis in perm])
    return f"{e.need(f'gather_{e.suffix(y)}')}({e.ref(y)}, {e.ref(x)}, {params});"


def _slice(e: Emitter, node: Node) -> str:
    from .folding import slice_ranges

    x, y = node.inputs[0], node.outputs[0]
    shape = e.graph.values[x].shape
    assert shape is not None
    constants = [e.graph.const(item) if item else None for item in node.inputs[1:]]
    constants += [None] * (4 - len(constants))
    starts, ends, axes, steps = constants
    assert starts is not None and ends is not None
    ranges = slice_ranges(shape, starts, ends, axes, steps)
    source = _strides(shape)
    start = sum(item[0] * stride for item, stride in zip(ranges, source, strict=True))
    dims = [item[1] for item in ranges]
    strides = [item[2] * stride for item, stride in zip(ranges, source, strict=True)]
    params = e.walk(dims, strides, start)
    return f"{e.need(f'gather_{e.suffix(y)}')}({e.ref(y)}, {e.ref(x)}, {params});"


def _pad(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    shape_x, shape_y = e.graph.values[x].shape, e.graph.values[y].shape
    assert shape_x is not None and shape_y is not None
    if not shape_x:
        raise UnsupportedError(f"{node.name}: cannot pad a scalar")
    pads = node.attrs["resolved_pads"]
    strides = _strides(shape_y)
    start = sum(pads[axis] * stride for axis, stride in enumerate(strides))
    params = e.walk(shape_x, strides, start)
    fill = e.graph.const(node.inputs[2]) if len(node.inputs) > 2 and node.inputs[2] else None
    suffix = e.suffix(y)
    if suffix == "s8":
        literal = str(0 if fill is None else int(fill))
    else:
        literal = float_literal(0.0 if fill is None else float(fill))
    size = e.graph.values[y].size
    fn = e.need(f"pad_{suffix}")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {literal}, {size}, {params});"


def _concat_layout(e: Emitter, inputs: list[str], axis: int) -> tuple[str, str, int, str]:
    shapes = [e.graph.values[item].shape for item in inputs]
    assert all(shape is not None for shape in shapes)
    blocks = _prod(shapes[0][:axis])  # type: ignore[index]
    chunks = e.int_array([_prod(shape[axis:]) for shape in shapes])  # type: ignore[index]
    pointers = ", ".join(e.ref(item) for item in inputs)
    return chunks, pointers, blocks, C_TYPES[e.graph.values[inputs[0]].dtype]  # type: ignore[index]


def _concat(e: Emitter, node: Node) -> str:
    y = node.outputs[0]
    chunks, pointers, blocks, c_type = _concat_layout(e, node.inputs, node.attrs["axis"])
    fn = e.need(f"concat_{e.suffix(y)}")
    return (
        f"{{ const {c_type} *const inputs[] = {{{pointers}}}; "
        f"{fn}({e.ref(y)}, inputs, {chunks}, {len(node.inputs)}, {blocks}); }}"
    )


def _qparams(e: Emitter, scale: str, zero_point: str) -> tuple[np.ndarray[Any, Any], int]:
    scale_data = e.graph.const(scale)
    zero_data = e.graph.const(zero_point)
    if scale_data is None or zero_data is None:
        raise UnsupportedError("quantization parameters must be constants")
    return np.asarray(scale_data, np.float64).reshape(-1), int(zero_data.reshape(-1)[0])


def _check_weights(node: Node, weights: Any, zero_point: Any) -> None:
    if weights is None:
        raise UnsupportedError(f"{node.name}: quantized weights must be constant")
    if weights.dtype != I8:
        raise UnsupportedError(f"{node.name}: quantized weights must be int8")
    if np.any(np.asarray(zero_point) != 0):
        raise UnsupportedError(f"{node.name}: weight zero point must be 0")


def _channel_multipliers(
    e: Emitter, x_scale: float, w_scale: np.ndarray[Any, Any], y_scale: float, channels: int
) -> tuple[str, str]:
    ratios = np.broadcast_to(w_scale, (channels,)) * x_scale / y_scale
    pairs = [multiplier(float(ratio)) for ratio in ratios]
    return e.int_array([p[0] for p in pairs]), e.int_array([p[1] for p in pairs])


def _bound_accumulator(node: Node, weights: Any, bias: Any) -> None:
    """Prove at compile time that int32 accumulators cannot overflow on any input."""
    per_channel = np.abs(weights.astype(np.int64)).reshape(weights.shape[0], -1).sum(axis=1)
    worst = per_channel * 255
    if bias is not None:
        worst = worst + np.abs(bias.astype(np.int64))
    if int(worst.max()) > ACCUMULATOR_LIMIT:
        raise CompileError(f"{node.name}: int32 accumulator could overflow")


def _qlinear_conv(e: Emitter, node: Node) -> str:
    inputs = node.inputs
    x, w, y = inputs[0], inputs[3], node.outputs[0]
    x_scale, x_zp = _qparams(e, inputs[1], inputs[2])
    w_scale, _ = _qparams(e, inputs[4], inputs[5])
    y_scale, y_zp = _qparams(e, inputs[6], inputs[7])
    weights = e.graph.const(w)
    _check_weights(node, weights, e.graph.const(inputs[5]))
    assert weights is not None
    if weights.ndim != 4:
        raise UnsupportedError(f"{node.name}: QLinearConv supports 2-D kernels")
    bias_data = e.graph.const(inputs[8]) if len(inputs) > 8 and inputs[8] else None
    _bound_accumulator(node, weights, bias_data)
    channels = weights.shape[0]
    mult, shift = _channel_multipliers(e, float(x_scale[0]), w_scale, float(y_scale[0]), channels)
    bias = _optional(e, node, 8)
    params = e.window(node, x, y)
    fn = e.need("qconv2d_s8")
    shape = e.graph.values[y].shape
    assert shape is not None
    e.scratch_elements = max(e.scratch_elements, shape[2] * shape[3])
    return (
        f"{fn}({e.ref(y)}, {e.ref(x)}, {e.ref(w)}, {bias}, {mult}, {shift}, S, {params}, "
        f"{x_zp}, {y_zp});"
    )


def _qdense(e: Emitter, node: Node, y_inputs: tuple[str, str], bias_name: str) -> str:
    inputs = node.inputs
    a, w, y = inputs[0], inputs[3], node.outputs[0]
    a_scale, a_zp = _qparams(e, inputs[1], inputs[2])
    w_scale, _ = _qparams(e, inputs[4], inputs[5])
    y_scale, y_zp = _qparams(e, *y_inputs)
    weights = e.graph.const(w)
    _check_weights(node, weights, e.graph.const(inputs[5]))
    assert weights is not None
    if node.attrs.get("alpha", 1.0) != 1.0:
        raise UnsupportedError(f"{node.name}: QGemm alpha must be 1")
    # Stored per output column so the accumulator loop walks consecutive memory.
    columns = weights if node.attrs.get("transB", 0) else weights.T
    n, k = columns.shape
    bias_data = e.graph.const(bias_name) if bias_name else None
    if bias_data is not None:
        bias_data = np.broadcast_to(bias_data.reshape(-1), (n,))
    _bound_accumulator(node, columns, bias_data)
    mult, shift = _channel_multipliers(e, float(a_scale[0]), w_scale, float(y_scale[0]), n)
    shape_a = e.graph.values[a].shape
    assert shape_a is not None
    bias = "0" if bias_data is None else e.array(bias_data.astype(np.int32))
    fn = e.need("qdense_s8")
    return (
        f"{fn}({e.ref(y)}, {e.ref(a)}, {e.array(columns)}, {bias}, {mult}, {shift}, "
        f"{shape_a[0]}, {k}, {n}, {a_zp}, {y_zp});"
    )


def _qgemm(e: Emitter, node: Node) -> str:
    bias = node.inputs[6] if len(node.inputs) > 6 else ""
    return _qdense(e, node, (node.inputs[7], node.inputs[8]), bias)


def _qlinear_matmul(e: Emitter, node: Node) -> str:
    return _qdense(e, node, (node.inputs[6], node.inputs[7]), "")


def _qlinear_binary(e: Emitter, node: Node) -> str:
    inputs = node.inputs
    a, b, y = inputs[0], inputs[3], node.outputs[0]
    a_scale, a_zp = _qparams(e, inputs[1], inputs[2])
    b_scale, b_zp = _qparams(e, inputs[4], inputs[5])
    y_scale, y_zp = _qparams(e, inputs[6], inputs[7])
    sa, sb, sy = float(a_scale[0]), float(b_scale[0]), float(y_scale[0])
    if node.op == "QLinearAdd":
        a_mult, b_mult, bits = add_multipliers(sa / sy, sb / sy)
        mult, shift = 0, 0
        kernel = "qadd_s8"
    else:
        a_mult, b_mult, bits = 0, 0, 0
        mult, shift = multiplier(sa * sb / sy)
        kernel = "qmul_s8"
    values = e.graph.values
    params = e.bcast(values[a], values[b], values[y])
    fields = [a_zp, b_zp, y_zp, a_mult, b_mult, bits, mult, shift]
    q = e.param("ox_qbin_t", ", ".join(str(item) for item in fields))
    return f"{e.need(kernel)}({e.ref(y)}, {e.ref(a)}, {e.ref(b)}, {params}, {q});"


def _qlinear_average_pool(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    x_scale, x_zp = _qparams(e, node.inputs[1], node.inputs[2])
    y_scale, y_zp = _qparams(e, node.inputs[3], node.inputs[4])
    k_h, k_w = node.attrs["kernel_shape"]
    if k_h * k_w * 255 * 256 > ACCUMULATOR_LIMIT:
        raise CompileError(f"{node.name}: window too large for an int32 accumulator")
    mult, shift = multiplier(float(x_scale[0]) / float(y_scale[0]) / 256.0)
    include = int(node.attrs.get("count_include_pad", 0))
    params = e.window(node, x, y)
    fn = e.need("qavgpool_s8")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {params}, {include}, {x_zp}, {y_zp}, {mult}, {shift});"


def _qlinear_global_pool(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    x_scale, x_zp = _qparams(e, node.inputs[1], node.inputs[2])
    y_scale, y_zp = _qparams(e, node.inputs[3], node.inputs[4])
    shape = e.graph.values[x].shape
    assert shape is not None
    if len(shape) != 4:
        raise UnsupportedError(f"{node.name}: QLinearGlobalAveragePool supports rank-4 input")
    size = shape[2] * shape[3]
    if size * 255 > ACCUMULATOR_LIMIT:
        raise CompileError(f"{node.name}: input too large for an int32 accumulator")
    mult, shift = multiplier(float(x_scale[0]) / float(y_scale[0]) / size)
    fn = e.need("qgap_s8")
    return (
        f"{fn}({e.ref(y)}, {e.ref(x)}, {shape[0] * shape[1]}, {size}, {x_zp}, {y_zp}, "
        f"{mult}, {shift});"
    )


def _qlinear_concat(e: Emitter, node: Node) -> str:
    y = node.outputs[0]
    y_scale, y_zp = _qparams(e, node.inputs[0], node.inputs[1])
    triples = node.inputs[2:]
    names = triples[0::3]
    params = [_qparams(e, s, z) for s, z in zip(triples[1::3], triples[2::3], strict=True)]
    chunks, pointers, blocks, _ = _concat_layout(e, names, node.attrs["axis"])
    head = f"{{ const int8_t *const inputs[] = {{{pointers}}}; "
    if all(s[0] == y_scale[0] and z == y_zp for s, z in params):
        fn = e.need("concat_s8")
        return f"{head}{fn}({e.ref(y)}, inputs, {chunks}, {len(names)}, {blocks}); }}"
    ratios = [multiplier(float(s[0]) / float(y_scale[0]) / 256.0) for s, _ in params]
    zero_points = e.int_array([z for _, z in params])
    mults = e.int_array([item[0] for item in ratios])
    shifts = e.int_array([item[1] for item in ratios])
    fn = e.need("qconcat_s8")
    return (
        f"{head}{fn}({e.ref(y)}, inputs, {chunks}, {len(names)}, {blocks}, "
        f"{zero_points}, {mults}, {shifts}, {y_zp}); }}"
    )


def _quantize(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    scale, zero_point = _qparams(e, node.inputs[1], node.inputs[2])
    literal = float_literal(float(np.float32(scale[0])))
    size = e.graph.values[y].size
    fn = e.need("quantize_s8")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {size}, {literal}, {zero_point});"


def _dequantize(e: Emitter, node: Node) -> str:
    x, y = node.inputs[0], node.outputs[0]
    scale_name = node.inputs[1]
    zero_name = node.inputs[2] if len(node.inputs) > 2 else ""
    scale = float(e.scalar(scale_name))
    zero_point = int(e.scalar(zero_name)) if zero_name else 0
    size = e.graph.values[y].size
    fn = e.need("dequantize_s8")
    return f"{fn}({e.ref(y)}, {e.ref(x)}, {size}, {float_literal(scale)}, {zero_point});"


EMITTERS = {
    "AveragePool": _pool,
    "Clip": _clip,
    "Concat": _concat,
    "Conv": _conv,
    "DequantizeLinear": _dequantize,
    "Gemm": _gemm,
    "GlobalAveragePool": _global_pool,
    "HardSigmoid": _hard_sigmoid,
    "LeakyRelu": _leaky_relu,
    "MaxPool": _pool,
    "Pad": _pad,
    "QGemm": _qgemm,
    "QLinearAdd": _qlinear_binary,
    "QLinearAveragePool": _qlinear_average_pool,
    "QLinearConcat": _qlinear_concat,
    "QLinearConv": _qlinear_conv,
    "QLinearGlobalAveragePool": _qlinear_global_pool,
    "QLinearMatMul": _qlinear_matmul,
    "QLinearMul": _qlinear_binary,
    "QuantizeLinear": _quantize,
    "ReduceMean": _reduce,
    "ReduceSum": _reduce,
    "Reshape": _copy,
    "Slice": _slice,
    "Softmax": _softmax,
    "Transpose": _transpose,
    **dict.fromkeys(BINARY, _binary),
    **dict.fromkeys(UNARY, _unary),
}
