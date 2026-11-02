"""The trained reporting models, on data small enough to commit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from onnxcc.compiler import compile_model, quantized_graph
from onnxcc.export import to_onnx
from onnxcc.memcheck import verify
from onnxcc.quantize import QuantizeOptions
from onnxcc.toolchain import SANITIZE_FLAGS, STRICT_FLAGS

from .support import assert_close, build, ort_run, sanitizers_work

REPO = Path(__file__).resolve().parent.parent
MEAN, STD = 0.2860, 0.3530


def fashion() -> tuple[onnx.ModelProto, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(REPO / "tests" / "data" / "fashion_sample.npz") as data:
        calibration = (data["calibration"].astype(np.float32) / 255.0 - MEAN) / STD
        images = (data["test_images"].astype(np.float32) / 255.0 - MEAN) / STD
        labels = data["test_labels"].astype(np.int64)
    return (
        onnx.load(str(REPO / "models" / "fashion_cnn.onnx")),
        calibration[:, None],
        images[:, None],
        labels,
    )


def mobilenet_inputs() -> tuple[onnx.ModelProto, np.ndarray, np.ndarray]:
    """CIFAR-10 has no redistribution licence, so synthetic images stand in for tests."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, (48, 1, 3, 32, 32)).astype(np.float32)
    return onnx.load(str(REPO / "models" / "cifar_mobilenet.onnx")), data[:32], data[32:]


def top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits[:, 0].argmax(1) == labels).mean())


def test_fashion_float_matches_onnxruntime_on_every_sample(tmp_path: Path) -> None:
    model, _, images, _ = fashion()
    result = compile_model(model, name="fashion")
    verify(result.source)
    outputs = build(result, tmp_path, [*STRICT_FLAGS, "-O2"]).run([images[:128]])[0]
    for index in range(128):
        assert_close([outputs[index]], ort_run(model, {"image": images[index]}), 1e-3, 1e-3)


def test_fashion_int8_accuracy_stays_close_to_float(tmp_path: Path) -> None:
    model, calibration, images, labels = fashion()
    float_result = compile_model(model, name="a")
    int8_result = compile_model(
        model, name="b", calibration=[calibration], quantize=QuantizeOptions()
    )
    verify(int8_result.source)
    float_logits = build(float_result, tmp_path, [*STRICT_FLAGS, "-O2"]).run([images])[0]
    int8_logits = build(int8_result, tmp_path, [*STRICT_FLAGS, "-O2"]).run([images])[0]
    assert top1(float_logits, labels) > 0.88
    assert top1(int8_logits, labels) >= top1(float_logits, labels) - 0.015


def test_fashion_int8_agrees_with_onnxruntime_on_the_same_graph(tmp_path: Path) -> None:
    model, calibration, images, _ = fashion()
    options = QuantizeOptions(weight_limit=63)
    result = compile_model(model, name="fashion", calibration=[calibration], quantize=options)
    outputs = build(result, tmp_path, ["-std=c99", "-O2"]).run([images[:128]])[0]
    reference = to_onnx(quantized_graph(model, [calibration], options))
    expected = np.stack([ort_run(reference, {"image": images[i]})[0] for i in range(128)])
    assert (outputs.argmax(-1) == expected.argmax(-1)).mean() >= 0.98
    assert np.abs(outputs - expected).max() < 0.5


@pytest.mark.parametrize("variant", ["float", "int8"])
def test_trained_models_run_clean_under_asan_and_ubsan(variant: str, tmp_path: Path) -> None:
    if not sanitizers_work():
        pytest.skip("no sanitizer runtime for this compiler")
    model, calibration, images, _ = fashion()
    options = (
        {"calibration": [calibration], "quantize": QuantizeOptions()} if variant == "int8" else {}
    )
    result = compile_model(model, name="fashion", isolated=True, **options)
    build(result, tmp_path, [*STRICT_FLAGS, *SANITIZE_FLAGS, "-O1"]).run([images[:8]])


def test_mobilenet_float_and_int8(tmp_path: Path) -> None:
    path = REPO / "models" / "cifar_mobilenet.onnx"
    if not path.exists():
        pytest.skip("model not trained yet")
    model, calibration, probe = mobilenet_inputs()
    float_result = compile_model(model, name="net")
    verify(float_result.source)
    assert float_result.memory.arena_bytes < float_result.memory.naive_bytes / 2
    outputs = build(float_result, tmp_path, [*STRICT_FLAGS, "-O2"]).run([probe])[0]
    for index in range(len(probe)):
        assert_close([outputs[index]], ort_run(model, {"image": probe[index]}), 2e-3, 2e-3)
    options = QuantizeOptions(weight_limit=63)
    int8_result = compile_model(model, name="net8", calibration=[calibration], quantize=options)
    verify(int8_result.source)
    assert "QLinearAdd" in {item.op for item in int8_result.graph.nodes}
    compiled = build(int8_result, tmp_path, [*STRICT_FLAGS, "-O2"]).run([probe])[0]
    reference = to_onnx(quantized_graph(model, [calibration], options))
    expected = np.stack([ort_run(reference, {"image": item})[0] for item in probe])
    assert np.abs(compiled - expected).max() < 1.0
