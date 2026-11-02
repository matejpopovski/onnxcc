"""In-memory graph. Node semantics stay those of ONNX, so a graph can be written back out.

The one exception is the `clamp` attribute on Conv and Gemm, which records a fused activation
and exists only after `passes.fuse_activations`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

F32 = np.dtype(np.float32)
I8 = np.dtype(np.int8)
I32 = np.dtype(np.int32)
I64 = np.dtype(np.int64)
RUNTIME_DTYPES = (F32, I8)


@dataclass
class Value:
    name: str
    dtype: np.dtype[Any] | None = None
    shape: tuple[int, ...] | None = None
    data: np.ndarray[Any, Any] | None = None

    @property
    def is_const(self) -> bool:
        return self.data is not None

    @property
    def size(self) -> int:
        assert self.shape is not None, f"{self.name} has no shape"
        return int(np.prod(self.shape, dtype=np.int64))


@dataclass
class Node:
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str = ""


@dataclass
class Graph:
    name: str
    values: dict[str, Value]
    nodes: list[Node]
    inputs: list[str]
    outputs: list[str]
    opset: int = 17
    _counter: int = 0

    def fresh(self, hint: str) -> str:
        while True:
            self._counter += 1
            name = f"{hint}__{self._counter}"
            if name not in self.values:
                return name

    def new_value(
        self, hint: str, dtype: np.dtype[Any] | None = None, shape: Sequence[int] | None = None
    ) -> str:
        name = self.fresh(hint)
        self.values[name] = Value(name, dtype, None if shape is None else tuple(shape))
        return name

    def add_const(self, hint: str, data: np.ndarray[Any, Any]) -> str:
        name = self.fresh(hint)
        self.values[name] = Value(name, data.dtype, tuple(data.shape), data)
        return name

    def const(self, name: str) -> np.ndarray[Any, Any] | None:
        """Constant contents of `name`, or None for a runtime value or an absent optional input."""
        value = self.values.get(name) if name else None
        return None if value is None else value.data

    def consumers(self, name: str) -> list[Node]:
        return [node for node in self.nodes if name in node.inputs]

    def producer(self, name: str) -> Node | None:
        return next((node for node in self.nodes if name in node.outputs), None)

    def replace_uses(self, old: str, new: str) -> None:
        for node in self.nodes:
            node.inputs = [new if item == old else item for item in node.inputs]

    def sole_consumer(self, name: str) -> Node | None:
        """The only consumer of `name`, provided it is not also observable as a graph output."""
        users = self.consumers(name)
        if name in self.outputs or len(users) != 1:
            return None
        if sum(item == name for item in users[0].inputs) != 1:
            return None
        return users[0]

    def eliminate_dead_code(self) -> None:
        live = set(self.outputs)
        kept: list[Node] = []
        for node in reversed(self.nodes):
            if any(output in live for output in node.outputs):
                kept.append(node)
                live.update(item for item in node.inputs if item)
        self.nodes = kept[::-1]


def live_constants(graph: Graph, nodes: Iterable[Node]) -> list[str]:
    """Names of constants that the given nodes actually read, in first-use order."""
    seen: dict[str, None] = {}
    for node in nodes:
        for item in node.inputs:
            if item and graph.values[item].is_const:
                seen.setdefault(item)
    return list(seen)
