"""Command line front end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .compiler import CompileResult, compile_model
from .errors import CompileError
from .memcheck import PlanViolation, verify
from .planner import ORDERINGS
from .quantize import QuantizeOptions
from .report import datasheet


def _parse_shapes(items: list[str]) -> dict[str, list[int]]:
    shapes = {}
    for item in items:
        name, _, dims = item.rpartition("=")
        if not name:
            raise CompileError(f"--input-shape expects name=1,3,32,32, got {item!r}")
        shapes[name] = [int(dim) for dim in dims.split(",")]
    return shapes


def _load_calibration(path: Path) -> list[np.ndarray[Any, Any]]:
    with np.load(path) as archive:
        return [archive[key] for key in archive.files]


def _compile(args: argparse.Namespace) -> int:
    quantize = None
    calibration = None
    if args.quantize:
        calibration = _load_calibration(args.quantize)
        quantize = QuantizeOptions(
            method=args.calibration, per_channel=not args.per_tensor, int8_io=args.int8_io
        )
    result = compile_model(
        args.model,
        name=args.name,
        input_shapes=_parse_shapes(args.input_shape),
        strategy=args.plan,
        calibration=calibration,
        quantize=quantize,
    )
    verify(result.source)
    output = args.output or args.model.with_suffix(".c")
    output.write_text(result.source)
    if args.header:
        output.with_suffix(".h").write_text(result.header)
    if args.json:
        args.json.write_text(json.dumps(datasheet(result), indent=2) + "\n")
    print(render_datasheet(result))
    return 0


def render_datasheet(result: CompileResult) -> str:
    sheet = datasheet(result)
    rows = [
        ("entry point", f"{result.c_name}_run"),
        ("steps", sheet["steps"]),
        ("arena", f"{sheet['arena_bytes']} bytes"),
        ("live peak (lower bound)", f"{sheet['peak_live_bytes']} bytes"),
        ("without planner", f"{sheet['naive_bytes']} bytes"),
        ("constants", f"{sheet['constant_bytes']} bytes"),
        ("kernels", ", ".join(sheet["kernels"])),
    ]
    width = max(len(str(key)) for key, _ in rows)
    return "\n".join(f"{key:<{width}}  {value}" for key, value in rows)


def _verify(args: argparse.Namespace) -> int:
    program = verify(args.source.read_text())
    print(f"ok: {len(program.steps)} steps, arenas {program.arenas}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnxcc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compile_cmd = commands.add_parser("compile", help="compile an ONNX model to C99")
    compile_cmd.add_argument("model", type=Path)
    compile_cmd.add_argument("-o", "--output", type=Path)
    compile_cmd.add_argument("--name", help="prefix of the generated entry point")
    compile_cmd.add_argument("--input-shape", action="append", default=[], metavar="NAME=D,D,...")
    compile_cmd.add_argument("--plan", default="best", choices=["best", "naive", *ORDERINGS])
    compile_cmd.add_argument(
        "--quantize", type=Path, metavar="NPZ", help="calibration inputs, one array per model input"
    )
    compile_cmd.add_argument(
        "--calibration", choices=["percentile", "minmax"], default="percentile"
    )
    compile_cmd.add_argument(
        "--per-tensor", action="store_true", help="one scale per weight tensor"
    )
    compile_cmd.add_argument("--int8-io", action="store_true", help="drop float at the interface")
    compile_cmd.add_argument("--header", action="store_true", help="also write a .h next to the .c")
    compile_cmd.add_argument("--json", type=Path, help="write the datasheet as JSON")
    compile_cmd.set_defaults(handler=_compile)

    verify_cmd = commands.add_parser("verify", help="check the memory plan of a generated file")
    verify_cmd.add_argument("source", type=Path)
    verify_cmd.set_defaults(handler=_verify)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CompileError, PlanViolation) as error:
        print(f"onnxcc: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
