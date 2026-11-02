"""Write a graph back to ONNX, so onnxruntime can execute exactly what was quantized."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .frontend import MICROSOFT_OPS
from .ir import Graph, live_constants

INTERNAL_ATTRS = ("materialize", "resolved_axes", "resolved_pads", "resolved_split")


def to_onnx(graph: Graph) -> onnx.ModelProto:
    nodes = []
    for index, node in enumerate(graph.nodes):
        attrs = {key: value for key, value in node.attrs.items() if key not in INTERNAL_ATTRS}
        if node.op in ("QLinearAveragePool",):
            attrs.pop("dilations", None)
        domain = "com.microsoft" if node.op in MICROSOFT_OPS else ""
        # Lowering can give several nodes one source name, which ONNX forbids.
        name = f"{node.name}_{index}"
        nodes.append(
            helper.make_node(node.op, node.inputs, node.outputs, name, domain=domain, **attrs)
        )

    def info(name: str) -> onnx.ValueInfoProto:
        value = graph.values[name]
        assert value.dtype is not None and value.shape is not None
        element = helper.np_dtype_to_tensor_dtype(np.dtype(value.dtype))
        return helper.make_tensor_value_info(name, element, list(value.shape))

    constants = [
        numpy_helper.from_array(np.asarray(graph.values[name].data), name)
        for name in live_constants(graph, graph.nodes)
    ]
    body = helper.make_graph(
        nodes,
        graph.name,
        [info(name) for name in graph.inputs],
        [info(name) for name in graph.outputs],
        constants,
    )
    model = helper.make_model(
        body,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=8,
    )
    return model
