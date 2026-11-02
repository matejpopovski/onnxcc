from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import onnx
from onnx import helper

from onnxcc.cli import main
from onnxcc.compiler import compile_model
from onnxcc.report import arena_svg
from onnxcc.toolchain import STRICT_FLAGS, compile_object

from .support import make_model

node = helper.make_node


def write_model(path: Path) -> None:
    x = np.zeros((1, 1, 4, 4), np.float32)
    w = np.ones((2, 1, 3, 3), np.float32)
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3]),
        node("Relu", ["c"], ["r"]),
        node("Flatten", ["r"], ["y"]),
    ]
    onnx.save(make_model(nodes, {"x": x}, {"w": w}, ["y"]), path)


def test_compile_verify_header_and_json(tmp_path: Path, capsys) -> None:
    model = tmp_path / "net.onnx"
    write_model(model)
    report = tmp_path / "net.json"
    assert main(["compile", str(model), "--name", "net", "--header", "--json", str(report)]) == 0
    output = capsys.readouterr().out
    assert "net_run" in output and "static RAM" in output
    compile_object(tmp_path / "net.c", tmp_path / "net.o", [*STRICT_FLAGS, "-O1"])
    assert "NET_STATIC_RAM_BYTES" in (tmp_path / "net.h").read_text()
    assert json.loads(report.read_text())["steps"] == 2
    assert main(["verify", str(tmp_path / "net.c")]) == 0


def test_quantize_from_the_command_line(tmp_path: Path) -> None:
    model = tmp_path / "net.onnx"
    write_model(model)
    calibration = tmp_path / "calibration.npz"
    samples = np.random.default_rng(0).uniform(-1, 1, (8, 1, 1, 4, 4)).astype(np.float32)
    np.savez(calibration, x=samples)
    output = tmp_path / "out.c"
    assert (
        main(
            ["compile", str(model), "-o", str(output), "--quantize", str(calibration), "--int8-io"]
        )
        == 0
    )
    assert "const int8_t *restrict in0" in output.read_text()
    output = tmp_path / "with_header.c"
    arguments = ["compile", str(model), "-o", str(output), "--quantize", str(calibration)]
    assert main([*arguments, "--int8-io", "--header"]) == 0
    assert "IN0_ZERO_POINT" in output.with_suffix(".h").read_text()


def test_arena_diagram_draws_one_box_per_buffer(tmp_path: Path) -> None:
    model = tmp_path / "net.onnx"
    write_model(model)
    result = compile_model(model, name="net")
    root = ElementTree.fromstring(arena_svg(result, "F", "net"))
    boxes = root.findall("{http://www.w3.org/2000/svg}rect")
    assert len(boxes) == len(result.buffers["F"]) + 2  # background and plot area


def test_verify_reports_a_broken_file(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.c"
    broken.write_text("int main(void) { return 0; }\n")
    assert main(["verify", str(broken)]) == 1
    assert "no annotated steps" in capsys.readouterr().err


def test_compile_error_exit_code(tmp_path: Path, capsys) -> None:
    model = tmp_path / "bad.onnx"
    x = np.zeros(2, np.float32)
    onnx.save(make_model([node("Sin", ["x"], ["y"])], {"x": x}, {}, ["y"]), model)
    assert main(["compile", str(model)]) == 1
    assert "unsupported operators: Sin" in capsys.readouterr().err
