"""Single-thread latency of generated C against onnxruntime, on this host only.

    uv run python -m experiments.latency

Both run the same model on the same input for repeated rounds, and the median round is reported.
The C timing is taken in a native loop. The onnxruntime timing goes through its Python binding,
which adds a small constant per call, so a trivial model is timed as well to expose that overhead.
Writes results/latency.json.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

from onnxcc.compiler import compile_model, quantized_graph
from onnxcc.export import to_onnx
from onnxcc.quantize import QuantizeOptions
from onnxcc.toolchain import host_cc

from .models import RESULTS, available

ROUNDS = 15
BENCH = """
#include <stdint.h>
#include <stdio.h>
#include <time.h>
{prototype};
static {in_type} in0[{in_size}];
static {out_type} out0[{out_size}];
static uint64_t now(void) {{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}}
int main(int argc, char **argv) {{
    FILE *file = fopen(argv[1], "rb");
    (void)argc;
    if (!file || fread(in0, sizeof in0[0], {in_size}, file) != {in_size}) return 1;
    for (int round = 0; round < {rounds}; ++round) {{
        const uint64_t start = now();
        for (int i = 0; i < {iterations}; ++i) {name}_run(in0, out0);
        printf("%llu\\n", (unsigned long long)(now() - start) / {iterations});
    }}
    return out0[0] == ({out_type})0.1234f;
}}
"""


def c_latency(result: Any, sample: np.ndarray, directory: Path, iterations: int) -> list[float]:
    input_spec, output_spec = result.inputs[0], result.outputs[0]
    bench = directory / f"{result.c_name}_bench.c"
    model = directory / f"{result.c_name}.c"
    binary = directory / f"{result.c_name}.bin"
    data = directory / f"{result.c_name}.raw"
    model.write_text(result.source)
    bench.write_text(
        BENCH.format(
            prototype=result.prototype,
            name=result.c_name,
            in_type=input_spec.c_type,
            in_size=input_spec.elements,
            out_type=output_spec.c_type,
            out_size=output_spec.elements,
            rounds=ROUNDS,
            iterations=iterations,
        )
    )
    data.write_bytes(sample.tobytes())
    subprocess.run(
        [host_cc(), "-O3", "-std=c99", str(model), str(bench), "-o", str(binary)], check=True
    )
    done = subprocess.run([binary, data], capture_output=True, text=True)
    return [float(line) for line in done.stdout.split()]


def host_size(result: Any, directory: Path) -> dict[str, int]:
    """Object file sections from the platform `size` tool, -O2, for this host's architecture."""
    source = directory / f"{result.c_name}.c"
    obj = directory / f"{result.c_name}_size.o"
    source.write_text(result.source)
    subprocess.run([host_cc(), "-O2", "-std=c99", "-c", str(source), "-o", str(obj)], check=True)
    listing = subprocess.run(["size", str(obj)], capture_output=True, text=True, check=True)
    header, row = listing.stdout.strip().splitlines()[:2]
    return dict(zip(header.split()[:3], [int(item) for item in row.split()[:3]], strict=True))


def runtime_library_bytes() -> int:
    """On-disk size of the onnxruntime native library, for scale."""
    capi = Path(ort.__file__).parent / "capi"
    return max(item.stat().st_size for item in capi.iterdir() if item.suffix in (".so", ".dylib"))


def ort_session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options)


def ort_latency(
    model: onnx.ModelProto, feed: dict[str, np.ndarray], iterations: int
) -> list[float]:
    session = ort_session(model)
    rounds = []
    for _ in range(ROUNDS):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            session.run(None, feed)
        rounds.append((time.perf_counter_ns() - start) / iterations)
    return rounds


def binding_overhead() -> float:
    info = helper.make_tensor_value_info("x", 1, [1])
    out = helper.make_tensor_value_info("y", 1, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "i", [info], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    return statistics.median(ort_latency(model, {"x": np.zeros(1, np.float32)}, 2000))


def summary(rounds: list[float]) -> dict[str, float]:
    ordered = sorted(rounds)
    return {
        "median_us": statistics.median(ordered) / 1000.0,
        "fastest_round_us": ordered[0] / 1000.0,
        "slowest_round_us": ordered[-1] / 1000.0,
    }


def main() -> None:
    report: dict[str, Any] = {
        "host": "Apple M3 Max, 36 GB, macOS, machine shared with other jobs during the run",
        "machine": platform.machine(),
        "c_flags": "-O3 -std=c99",
        "onnxruntime": ort.__version__,
        "threads": 1,
        "rounds": ROUNDS,
        "onnxruntime_binding_overhead_us": binding_overhead() / 1000.0,
        "onnxruntime_native_library_bytes": runtime_library_bytes(),
    }
    with tempfile.TemporaryDirectory(prefix="onnxcc-latency-") as scratch:
        directory = Path(scratch)
        for entry in available():
            calibration, images, _ = entry.splits()
            model = onnx.load(str(entry.path))
            sample = images[0]
            feed = {model.graph.input[0].name: sample}
            float_result = compile_model(model, name="float_model")
            int8_result = compile_model(
                model, name="int8_model", calibration=[calibration], quantize=QuantizeOptions()
            )
            int8_model = to_onnx(quantized_graph(model, [calibration], QuantizeOptions()))
            iterations = 300
            report[entry.name] = {
                "float_c": summary(c_latency(float_result, sample, directory, iterations)),
                "float_onnxruntime": summary(ort_latency(model, feed, iterations)),
                "int8_c": summary(c_latency(int8_result, sample, directory, iterations)),
                "int8_onnxruntime_same_graph": summary(ort_latency(int8_model, feed, iterations)),
                "host_object_float": host_size(float_result, directory),
                "host_object_int8": host_size(int8_result, directory),
            }
            print(entry.name, report[entry.name], flush=True)
    (RESULTS / "latency.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
