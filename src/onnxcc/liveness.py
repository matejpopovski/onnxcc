"""Decide which tensors share storage and for how long each storage is needed.

Sharing comes in two kinds. A Reshape output is the same bytes as its input, so it costs nothing
and emits no code. An element-wise op may also overwrite an input that dies at that step. Both
kinds are folded into one storage record here, and the planner sees only storages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import F32, Graph, Node
from .planner import Buffer

ELEMENTWISE_UNARY = frozenset(
    {
        "Abs",
        "Clip",
        "Exp",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Neg",
        "Sigmoid",
        "Softmax",
        "Tanh",
    }
)
ELEMENTWISE_BINARY = frozenset({"Add", "Div", "Max", "Min", "Mul", "PRelu", "Sub"})
QLINEAR_BINARY = frozenset({"QLinearAdd", "QLinearMul"})


def data_inputs(graph: Graph, node: Node) -> list[str]:
    return [item for item in node.inputs if item and not graph.values[item].is_const]


def _inplace_candidates(graph: Graph, node: Node) -> list[str]:
    """Inputs whose storage this node may overwrite. The kernels have no `restrict` on them."""
    if node.op in ELEMENTWISE_UNARY:
        return node.inputs[:1]
    if node.op in ELEMENTWISE_BINARY:
        return node.inputs[:2]
    if node.op in QLINEAR_BINARY:
        return [node.inputs[0], node.inputs[3]]
    return []


def is_alias(graph: Graph, node: Node) -> bool:
    return (
        node.op == "Reshape"
        and node.outputs[0] not in graph.outputs
        and not graph.values[node.inputs[0]].is_const
    )


@dataclass
class Liveness:
    steps: list[Node]
    # Value name -> name of the value that owns the storage.
    storage: dict[str, str]
    # Arena letter -> buffers to place. Graph inputs and outputs live in caller memory.
    buffers: dict[str, list[Buffer]] = field(default_factory=dict)


def arena_of(graph: Graph, name: str) -> str:
    return "F" if graph.values[name].dtype == F32 else "Q"


def analyze(graph: Graph, inplace: bool = True) -> Liveness:
    root: dict[str, str] = {}
    steps: list[Node] = []
    for node in graph.nodes:
        if is_alias(graph, node):
            source = node.inputs[0]
            root[node.outputs[0]] = root.get(source, source)
        else:
            steps.append(node)

    def storage_of(name: str) -> str:
        current = root.get(name, name)
        while current in root:
            current = root[current]
        return current

    caller_owned = set(graph.inputs) | set(graph.outputs)
    defined: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for index, node in enumerate(steps):
        for item in data_inputs(graph, node):
            last_use[storage_of(item)] = index
        defined[node.outputs[0]] = index
        last_use.setdefault(node.outputs[0], index)

    if inplace:
        for index, node in enumerate(steps):
            output = node.outputs[0]
            if output in caller_owned:
                continue
            for candidate in _inplace_candidates(graph, node):
                value = graph.values[candidate]
                owner = storage_of(candidate)
                if (
                    not value.is_const
                    and owner not in caller_owned
                    and last_use[owner] == index
                    and value.size == graph.values[output].size
                    and value.dtype == graph.values[output].dtype
                ):
                    root[output] = owner
                    last_use[owner] = max(index, last_use.pop(output))
                    break

    storage = {name: storage_of(name) for name in graph.values if not graph.values[name].is_const}
    buffers: dict[str, list[Buffer]] = {"F": [], "Q": []}
    for name, first in defined.items():
        if name in caller_owned or storage[name] != name:
            continue
        buffer = Buffer(name, graph.values[name].size, first, last_use[name])
        buffers[arena_of(graph, name)].append(buffer)
    return Liveness(steps, storage, buffers)
