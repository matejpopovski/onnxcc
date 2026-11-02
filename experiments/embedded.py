"""Cross-compile for Cortex-M4F and, when QEMU is present, run the result under emulation.

    uv run python -m experiments.embedded

Sizes come from arm-none-eabi-size on the object file and on a linked image. The emulator proves
the cross-compiled code computes the same answers as the host build. It is not a cycle model: with
-icount shift=0 QEMU advances virtual time one nanosecond per instruction, so timer ticks measure
retired instructions, not cycles and not latency on a physical microcontroller.

Writes results/embedded.json.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from onnxcc.compiler import CompileResult, compile_model
from onnxcc.quantize import QuantizeOptions
from onnxcc.toolchain import (
    STRICT_FLAGS,
    Binary,
    freestanding_flags,
    harness_source,
    link_harness,
)

from .models import REPO, RESULTS, available

CPU = ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
GCC = "arm-none-eabi-gcc"
SIZE = "arm-none-eabi-size"
FREESTANDING = [*freestanding_flags(GCC), "-fno-common"]
NANOSECONDS_PER_TIMER_TICK = 40  # 25 MHz APB clock on MPS2
VECTORS = 8


def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, **options)


def section_sizes(path: Path) -> dict[str, int]:
    """Totals per section family from `size -A`, so weights and code are reported apart."""
    listing = run([SIZE, "-A", "-d", str(path)]).stdout
    sizes: dict[str, int] = {}
    for line in listing.splitlines():
        match = re.match(r"(\.\w+)\S*\s+(\d+)\s+\d+", line)
        if match and match.group(1) in (".text", ".rodata", ".data", ".bss"):
            sizes[match.group(1)] = sizes.get(match.group(1), 0) + int(match.group(2))
    return sizes


def stack_usage(path: Path) -> int:
    """Largest single frame from -fstack-usage, which needs no emulator."""
    frames = [int(line.split("\t")[1]) for line in path.read_text().splitlines() if line]
    return max(frames, default=0)


def vectors_header(result: CompileResult, inputs: np.ndarray) -> str:
    spec_in, spec_out = result.inputs[0], result.outputs[0]
    rows = []
    for sample in inputs:
        if spec_in.c_type == "float":
            values = [float(v).hex() + "f" for v in sample.reshape(-1)]
        else:
            values = [str(int(v)) for v in sample.reshape(-1)]
        rows.append("{" + ", ".join(values) + "}")
    joined = ",\n".join(rows)
    return (
        f"#define VECTOR_COUNT {len(inputs)}\n"
        f"#define MODEL_INPUT_ELEMENTS {spec_in.elements}\n"
        f"#define MODEL_OUTPUT_ELEMENTS {spec_out.elements}\n"
        f"typedef {spec_in.c_type} model_input_t;\n"
        f"typedef {spec_out.c_type} model_output_t;\n"
        f"#define MODEL_RUN {result.c_name}_run\n"
        f"{result.prototype};\n"
        f"static const model_input_t vector_inputs[VECTOR_COUNT][MODEL_INPUT_ELEMENTS] = {{\n"
        f"{joined}\n}};\n"
    )


def build(result: CompileResult, inputs: np.ndarray, directory: Path, level: str) -> dict[str, Any]:
    source = directory / f"{result.c_name}.c"
    source.write_text(result.source)
    obj = directory / f"{result.c_name}.o"
    flags = [*STRICT_FLAGS, *CPU, *FREESTANDING, level, "-fstack-usage"]
    run([GCC, *flags, "-c", str(source), "-o", str(obj)])
    undefined = run(["arm-none-eabi-nm", "-u", str(obj)]).stdout.split()
    (directory / "vectors.h").write_text(vectors_header(result, inputs))
    image = directory / f"{result.c_name}.elf"
    run(
        [
            GCC,
            *CPU,
            *FREESTANDING,
            "-std=gnu99",
            level,
            "-nostdlib",
            "-nostartfiles",
            f"-T{REPO / 'embedded' / 'mps2_an386.ld'}",
            f"-I{directory}",
            str(obj),
            str(REPO / "embedded" / "startup.c"),
            str(REPO / "embedded" / "app_main.c"),
            "-lgcc",
            "-o",
            str(image),
        ]
    )
    return {
        "object_sections": section_sizes(obj),
        "linked_image_sections": section_sizes(image),
        "largest_stack_frame_bytes": stack_usage(obj.with_suffix(".su")),
        "undefined_symbols_in_object": [s for s in undefined if s != "U"],
        "image": image,
    }


def emulate(image: Path) -> tuple[list[int], list[list[int]], int]:
    if shutil.which("qemu-system-arm") is None:
        return [], [], 0
    done = run(
        [
            "qemu-system-arm",
            "-machine", "mps2-an386",
            "-cpu", "cortex-m4",
            "-nographic",
            "-monitor", "none",
            "-serial", "none",
            "-semihosting-config", "enable=on,target=native",
            "-icount", "shift=0,align=off",
            "-kernel", str(image),
        ],
        timeout=600,
    )  # fmt: skip
    ticks, outputs = [], []
    stack = 0
    # QEMU routes semihosting text to stderr.
    for line in (done.stdout + done.stderr).splitlines():
        if line.startswith("stack "):
            stack = int(line.split()[1], 16)
            continue
        words = [int(word, 16) for word in line.split()]
        if words:
            ticks.append(words[0])
            outputs.append(words[1:])
    return ticks, outputs, stack


def host_outputs(result: CompileResult, inputs: np.ndarray, directory: Path) -> np.ndarray:
    harness = harness_source(result.c_name, result.prototype, result.inputs, result.outputs, False)
    path = link_harness(result.source, harness, directory, f"host_{result.c_name}", ["-O2"])
    return Binary(path, result.inputs, result.outputs).run([inputs])[0]


def measure(
    result: CompileResult, inputs: np.ndarray, directory: Path, level: str
) -> dict[str, Any]:
    built = build(result, inputs, directory, level)
    image = built.pop("image")
    ticks, words, stack = emulate(image)
    if ticks:
        built["qemu_stack_high_water_bytes"] = stack
        expected = host_outputs(result, inputs, directory)
        if result.outputs[0].c_type == "float":
            emulated = np.array(words, np.uint32).view(np.float32).reshape(expected.shape)
            difference = float(np.abs(emulated - expected).max())
            built["qemu_vs_host_max_abs_diff"] = difference
            built["qemu_vs_host_same_prediction"] = bool(
                np.array_equal(emulated[:, 0].argmax(1), expected[:, 0].argmax(1))
            )
        else:
            emulated = np.array(words, np.uint32).astype(np.int8).reshape(expected.shape)
            built["qemu_vs_host_bit_exact"] = bool(np.array_equal(emulated, expected))
        built["qemu_instructions_per_inference_estimate"] = int(
            np.median(ticks) * NANOSECONDS_PER_TIMER_TICK
        )
    return built


def main() -> None:
    gcc_version = run([GCC, "--version"]).stdout.splitlines()[0]
    report: dict[str, Any] = {
        "toolchain": gcc_version,
        "flags": " ".join([*CPU, *FREESTANDING]),
        "qemu": run(["qemu-system-arm", "--version"]).stdout.splitlines()[0]
        if shutil.which("qemu-system-arm")
        else None,
        "qemu_note": "instruction counts from -icount shift=0 under TCG emulation, not cycles",
    }
    with tempfile.TemporaryDirectory(prefix="onnxcc-embedded-") as scratch:
        directory = Path(scratch)
        for entry in available():
            calibration, images, _ = entry.splits()
            model = onnx.load(str(entry.path))
            inputs = images[:VECTORS]
            float_result = compile_model(model, name="float_model")
            int8_result = compile_model(
                model, name="int8_model", calibration=[calibration], quantize=QuantizeOptions()
            )
            options = QuantizeOptions(int8_io=True)
            pure = compile_model(
                model, name="int8_io_model", calibration=[calibration], quantize=options
            )
            scale = next(iter(pure.interface.values()))
            quantized_inputs = np.clip(
                np.rint(inputs / scale.scale) + scale.zero_point, -128, 127
            ).astype(np.int8)
            report[entry.name] = {
                level: {
                    "float32": measure(float_result, inputs, directory, level),
                    "int8_float_interface": measure(int8_result, inputs, directory, level),
                    "int8_integer_only": measure(pure, quantized_inputs, directory, level),
                }
                for level in ("-Os", "-O2")
            }
            print(entry.name, json.dumps(report[entry.name], indent=2), flush=True)
    (RESULTS / "embedded.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
