"""Accuracy on full test sets: onnxruntime float, generated C float, generated C int8.

    uv run python -m experiments.evaluate

Writes results/accuracy.json. Calibration samples come from the training set, never the test set.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from onnxcc.compiler import compile_model, quantized_graph
from onnxcc.export import to_onnx
from onnxcc.memcheck import verify
from onnxcc.quantize import QuantizeOptions
from onnxcc.toolchain import Binary, harness_source, link_harness

from .models import RESULTS, Reported, available

VARIANTS = {
    "percentile99.99_per_channel": QuantizeOptions("percentile", 99.99, True),
    "minmax_per_channel": QuantizeOptions("minmax", 99.99, True),
    "percentile99.99_per_tensor": QuantizeOptions("percentile", 99.99, False),
    "minmax_per_tensor": QuantizeOptions("minmax", 99.99, False),
}


def ort_outputs(model: onnx.ModelProto, name: str, images: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    session = ort.InferenceSession(model.SerializeToString(), options)
    return np.concatenate(
        [session.run(None, {name: images[index]})[0] for index in range(len(images))]
    )


def c_outputs(result: Any, images: np.ndarray, directory: Path) -> np.ndarray:
    verify(result.source)
    harness = harness_source(result.c_name, result.prototype, result.inputs, result.outputs, False)
    path = link_harness(result.source, harness, directory, result.c_name, ["-std=c99", "-O2"])
    return Binary(path, result.inputs, result.outputs).run([images])[0][:, 0]


def top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(1) == labels).mean())


def evaluate(entry: Reported) -> dict[str, Any]:
    calibration, images, labels = entry.splits()
    model = onnx.load(str(entry.path))
    input_name = model.graph.input[0].name
    report: dict[str, Any] = {"test_samples": len(labels), "calibration_samples": len(calibration)}
    reference = ort_outputs(model, input_name, images)
    report["float_onnxruntime_top1"] = top1(reference, labels)
    with tempfile.TemporaryDirectory(prefix="onnxcc-eval-") as scratch:
        directory = Path(scratch)
        compiled = c_outputs(compile_model(model, name="float_model"), images, directory)
        report["float_c_top1"] = top1(compiled, labels)
        report["float_c_vs_ort_max_abs_logit_diff"] = float(np.abs(compiled - reference).max())
        report["float_c_vs_ort_prediction_agreement"] = float(
            (compiled.argmax(1) == reference.argmax(1)).mean()
        )
        report["int8"] = {}
        for label, options in VARIANTS.items():
            result = compile_model(
                model, name="int8_model", calibration=[calibration], quantize=options
            )
            quantized = c_outputs(result, images, directory)
            graph = quantized_graph(model, [calibration], options)
            ort_int8 = ort_outputs(to_onnx(graph), input_name, images)
            report["int8"][label] = {
                "c_top1": top1(quantized, labels),
                "accuracy_drop_points": 100.0 * (report["float_c_top1"] - top1(quantized, labels)),
                "onnxruntime_same_graph_top1": top1(ort_int8, labels),
                "c_vs_ort_prediction_agreement": float(
                    (quantized.argmax(1) == ort_int8.argmax(1)).mean()
                ),
            }
            print(entry.name, label, report["int8"][label], flush=True)
    return report


def main() -> None:
    results = {entry.name: evaluate(entry) for entry in available()}
    (RESULTS / "accuracy.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
