"""End-to-end int8 behaviour on small synthetic graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from onnxcc.compiler import compile_model, quantized_graph
from onnxcc.errors import CompileError, UnsupportedError
from onnxcc.export import to_onnx
from onnxcc.memcheck import verify
from onnxcc.quantize import QuantizeOptions, activation_params

from .cases_float import noise
from .support import build, make_model, ort_run

node = helper.make_node
F = np.float32
DIFFERENTIAL = QuantizeOptions(method="minmax", weight_limit=63)


def small_net() -> object:
    w = noise("qnet", 4, 2, 3, 3)
    d = noise("qnet2", 8, 3)
    b = noise("qnet3", 3)
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        node("Relu", ["c"], ["r"]),
        node("Add", ["r", "r2"], ["s"]),
        node("AveragePool", ["s"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
        node("MaxPool", ["p"], ["q"], kernel_shape=[2, 2], strides=[2, 2]),
        node("Concat", ["q", "q"], ["k"], axis=1),
        node("GlobalAveragePool", ["k"], ["g"]),
        node("Flatten", ["g"], ["f"]),
        node("Gemm", ["f", "d", "b"], ["y"]),
    ]
    nodes.insert(2, node("Sigmoid", ["c"], ["r2"]))
    x = np.zeros((1, 2, 8, 8), F)
    return make_model(nodes, {"x": x}, {"w": w, "d": d, "b": b}, ["y"])


def calibration(count: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (count, 1, 2, 8, 8)).astype(F)


def test_zero_is_exactly_representable_for_any_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(1000):
        low, high = np.sort(rng.normal(0, 10, 2))
        params = activation_params(float(low), float(high))
        assert -128 <= params.zero_point <= 127
        assert params.scale > 0
    assert activation_params(0.0, 0.0).scale == 1.0
    assert activation_params(0.0, 6.0).zero_point == -128


def test_quantized_c_tracks_float_and_onnxruntime(tmp_path: Path) -> None:
    model = small_net()
    samples = [calibration()]
    result = compile_model(model, name="net", calibration=samples, quantize=DIFFERENTIAL)
    verify(result.source)
    assert "QLinearConv" in {item.op for item in result.graph.nodes}
    assert "Sigmoid" in {item.op for item in result.graph.nodes}, "no int8 kernel, stays float"
    probe = calibration(16, seed=1)
    outputs = build(result, tmp_path, ["-std=c99", "-O1"]).run([probe])[0]
    reference = to_onnx(quantized_graph(model, samples, DIFFERENTIAL))
    for index in range(len(probe)):
        expected = ort_run(reference, {"x": probe[index]})[0]
        float_expected = ort_run(model, {"x": probe[index]})[0]
        spread = float(np.ptp(float_expected)) + 1e-3
        assert np.abs(outputs[index] - expected).max() <= 0.05 * spread
        assert np.abs(outputs[index] - float_expected).max() <= 0.2 * spread


def test_int8_interface_removes_float_kernels(tmp_path: Path) -> None:
    w = noise("io", 2, 1, 3, 3)
    d = noise("io2", 2, 3)
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], strides=[2, 2]),
        node("Relu", ["c"], ["r"]),
        node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
        node("Flatten", ["p"], ["f"]),
        node("Gemm", ["f", "d"], ["y"]),
    ]
    model = make_model(nodes, {"x": np.zeros((1, 1, 8, 8), F)}, {"w": w, "d": d}, ["y"])
    samples = [np.random.default_rng(2).uniform(0, 1, (32, 1, 1, 8, 8)).astype(F)]
    plain = compile_model(model, name="a", calibration=samples, quantize=QuantizeOptions())
    pure = compile_model(
        model, name="b", calibration=samples, quantize=QuantizeOptions(int8_io=True)
    )
    assert "float" not in pure.source.split("static int8_t Q[")[1]
    assert "(void)" not in pure.prototype
    assert pure.inputs[0].c_type == "int8_t" and pure.outputs[0].c_type == "int8_t"
    assert len(pure.interface) == 2
    q_in, q_out = pure.interface.values()
    probe = samples[0][:8]
    float_out = build(plain, tmp_path, ["-O1"]).run([probe])[0]
    rounded = np.clip(np.rint(probe / q_in.scale) + q_in.zero_point, -128, 127).astype(np.int8)
    int_out = build(pure, tmp_path, ["-O1"]).run([rounded])[0]
    np.testing.assert_allclose(
        (int_out.astype(F) - q_out.zero_point) * q_out.scale, float_out, atol=1e-5
    )


def test_float_consumer_input_blocks_int8_interface() -> None:
    nodes = [node("Sigmoid", ["x"], ["y"])]
    model = make_model(nodes, {"x": np.zeros(4, F)}, {}, ["y"])
    samples = [np.zeros((4, 4), F)]
    with pytest.raises(UnsupportedError):
        compile_model(model, calibration=samples, quantize=QuantizeOptions(int8_io=True))


def test_values_reshaped_after_a_float_op_are_calibrated(tmp_path: Path) -> None:
    """Reshape emits no step, so its output needs its own observation."""
    x = np.zeros((1, 2, 4, 4), F)
    nodes = [
        node("Sigmoid", ["x"], ["s"]),
        node("Flatten", ["s"], ["f"]),
        node("Gemm", ["f", "d"], ["y"]),
    ]
    model = make_model(nodes, {"x": x}, {"d": noise("reshape", 32, 3)}, ["y"])
    samples = [calibration(8)[:, :, :, :4, :4]]
    result = compile_model(model, name="m", calibration=samples, quantize=DIFFERENTIAL)
    probe = samples[0][:1]
    output = build(result, tmp_path, ["-O1"]).run([probe])[0][0]
    expected = ort_run(model, {"x": probe[0]})[0]
    assert np.abs(output - expected).max() < 0.1 * (np.ptp(expected) + 1e-3)


def test_a_clip_excluding_zero_is_not_absorbed_by_saturation(tmp_path: Path) -> None:
    """Quantized ranges contain zero, so saturating cannot implement Clip(0.5, 1)."""
    x = np.zeros((1, 1, 4, 4), F)
    bounds = {"lo": np.array(0.5, F), "hi": np.array(1.0, F)}
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[1, 1]),
        node("Clip", ["c", "lo", "hi"], ["y"]),
    ]
    model = make_model(nodes, {"x": x}, {"w": np.ones((1, 1, 1, 1), F), **bounds}, ["y"])
    samples = [np.random.default_rng(0).uniform(-1, 1, (32, 1, 1, 4, 4)).astype(F)]
    result = compile_model(model, name="m", calibration=samples, quantize=DIFFERENTIAL)
    assert "Clip" in {item.op for item in result.graph.nodes}
    output = build(result, tmp_path, ["-O1"]).run([samples[0][:1]])[0][0]
    assert output.min() >= 0.49 and output.max() <= 1.01


def test_exported_graph_has_unique_node_names() -> None:
    """Split lowers to several Slices that inherit one name, which onnxruntime rejects."""
    x = np.zeros((1, 2, 4, 4), F)
    nodes = [
        node("Split", ["x"], ["a", "b"], axis=1, num_outputs=2, name="split"),
        node("Add", ["a", "b"], ["y"]),
    ]
    model = make_model(nodes, {"x": x}, {}, ["y"], opset=18)
    samples = [calibration(8)[:, :, :, :4, :4]]
    exported = to_onnx(quantized_graph(model, samples, DIFFERENTIAL))
    names = [item.name for item in exported.graph.node]
    assert len(names) == len(set(names))


def test_percentile_range_is_narrower_than_minmax_under_outliers() -> None:
    w = np.ones((1, 1, 1, 1), F)
    model = make_model(
        [node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1])],
        {"x": np.zeros((1, 1, 32, 32), F)},
        {"w": w},
        ["y"],
    )
    data = np.random.default_rng(0).normal(0, 1, (16, 1, 1, 32, 32)).astype(F)
    data[0, 0, 0, 0, 0] = 1000.0

    def scale(method: str) -> float:
        graph = quantized_graph(model, [data], QuantizeOptions(method=method))
        conv = next(item for item in graph.nodes if item.op == "QLinearConv")
        return float(graph.values[conv.inputs[6]].data)

    assert scale("percentile") * 50 < scale("minmax")


def test_per_channel_gives_one_scale_per_filter() -> None:
    model = small_net()
    graph = quantized_graph(model, [calibration()], QuantizeOptions())
    conv = next(item for item in graph.nodes if item.op == "QLinearConv")
    assert graph.values[conv.inputs[4]].data.shape == (4,)
    tensor = quantized_graph(model, [calibration()], QuantizeOptions(per_channel=False))
    conv = next(item for item in tensor.nodes if item.op == "QLinearConv")
    assert len(set(graph.values[conv.inputs[4]].data.tolist())) >= 1
    assert len(set(tensor.values[conv.inputs[4]].data.tolist())) == 1


def test_rejects_already_quantized_models_and_bad_calibration() -> None:
    x = np.zeros((2, 2), F)
    model = make_model(
        [
            node("QuantizeLinear", ["x", "s", "z"], ["q"]),
            node(
                "QLinearMul",
                ["q", "s", "z", "q", "s", "z", "s", "z"],
                ["m"],
                domain="com.microsoft",
            ),
            node("DequantizeLinear", ["m", "s", "z"], ["y"]),
        ],
        {"x": x},
        {"s": np.array(0.1, F), "z": np.array(0, np.int8)},
        ["y"],
        microsoft=True,
    )
    with pytest.raises(UnsupportedError, match="already quantized"):
        compile_model(model, calibration=[x[None]], quantize=QuantizeOptions())
    with pytest.raises(CompileError, match="calibration"):
        compile_model(small_net(), calibration=[], quantize=QuantizeOptions())


def test_accumulator_overflow_is_a_compile_error() -> None:
    """A 300x300 kernel of full-scale weights can exceed int32. The compiler must refuse it."""
    size = 300
    w = np.full((1, 1, size, size), 127, np.int8)
    inputs = ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"]
    init = {
        "w": w,
        "xs": np.array(1.0, F),
        "xz": np.array(0, np.int8),
        "ws": np.array(1.0, F),
        "wz": np.array(0, np.int8),
        "ys": np.array(1.0, F),
        "yz": np.array(0, np.int8),
    }
    model = make_model(
        [node("QLinearConv", inputs, ["y"], kernel_shape=[size, size])],
        {"x": np.zeros((1, 1, size, size), np.int8)},
        init,
        ["y"],
    )
    with pytest.raises(CompileError, match="overflow"):
        compile_model(model)
