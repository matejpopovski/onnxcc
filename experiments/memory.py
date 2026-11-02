"""Arena size with planning against one array per tensor, and arena diagrams for the README.

    uv run python -m experiments.memory

Writes results/memory.json and docs/arena_*.svg.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import onnx

from onnxcc.compiler import compile_model
from onnxcc.planner import ORDERINGS, make_plan
from onnxcc.quantize import QuantizeOptions
from onnxcc.report import arena_svg

from .models import REPO, RESULTS, available


def strategy_table(result: Any) -> dict[str, int]:
    """Bytes per strategy, with the same aliasing and in-place decisions."""
    table = {}
    element = {"F": 4, "Q": 1}
    for strategy in [*ORDERINGS, "naive"]:
        table[strategy] = sum(
            make_plan(buffers, strategy).size * element[arena]
            for arena, buffers in result.buffers.items()
        )
    return table


def main() -> None:
    report: dict[str, Any] = {}
    docs = REPO / "docs"
    docs.mkdir(exist_ok=True)
    for entry in available():
        calibration, _, _ = entry.splits()
        model = onnx.load(str(entry.path))
        options = QuantizeOptions()
        variants = {
            "float32": compile_model(model, name="m"),
            "int8": compile_model(model, name="m", calibration=[calibration], quantize=options),
        }
        no_inplace = {
            "float32": compile_model(model, name="m", inplace=False),
            "int8": compile_model(
                model, name="m", inplace=False, calibration=[calibration], quantize=options
            ),
        }
        report[entry.name] = {}
        for label, result in variants.items():
            report[entry.name][label] = {
                **asdict(result.memory),
                "arena_without_in_place": no_inplace[label].memory.arena_bytes,
                "per_strategy_bytes": strategy_table(result),
            }
        planned = variants["int8"]
        tensors = planned.memory.arena_bytes - planned.memory.scratch_bytes
        title = f"{entry.name} int8: {tensors} byte tensor arena, step on x, offset on y"
        (docs / f"arena_{entry.name}.svg").write_text(arena_svg(planned, "Q", title))
    (RESULTS / "memory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
