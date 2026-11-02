"""Machine-readable datasheet for a compiled model, and an SVG of its arena."""

from __future__ import annotations

import re
from typing import Any

from .compiler import CompileResult


def datasheet(result: CompileResult) -> dict[str, Any]:
    memory = result.memory
    run_body = result.source.rsplit("_run(", 1)[-1]
    kernels = sorted(set(re.findall(r"^\s+(?:\{ .*?; )?ox_(\w+)\(", run_body, re.M)))
    return {
        "entry_point": f"{result.c_name}_run",
        "inputs": [{"type": s.c_type, "shape": list(s.shape)} for s in result.inputs],
        "outputs": [{"type": s.c_type, "shape": list(s.shape)} for s in result.outputs],
        "steps": len(re.findall(r"^\s*/\*@ \d+ ", result.source, re.M)),
        "arena_bytes": memory.arena_bytes,
        "scratch_bytes": memory.scratch_bytes,
        "peak_live_bytes": memory.peak_live_bytes,
        "naive_bytes": memory.naive_bytes,
        "constant_bytes": memory.constant_bytes,
        "strategy": memory.strategy,
        "kernels": kernels,
        "interface_qparams": {
            name: {"scale": params.scale, "zero_point": params.zero_point}
            for name, params in result.interface.items()
        },
    }


def arena_svg(result: CompileResult, arena: str, title: str) -> str:
    """Time on the x axis, arena offset on the y axis, one rectangle per buffer."""
    buffers = result.buffers[arena]
    plan = result.plans[arena]
    width, height, left, top, bottom = 760, 300, 64, 34, 34
    plot_w, plot_h = width - left - 16, height - top - bottom
    steps = max((b.last for b in buffers), default=0) + 1
    size = max(plan.size, 1)
    unit = "bytes" if arena == "Q" else "floats"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'font-family="ui-monospace, Menlo, monospace" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="#fbfaf6"/>',
        f'<text x="{left}" y="20" font-size="13" fill="#1c1c1c">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f0eee6" '
        'stroke="#8a8778"/>',
    ]
    for index, buffer in enumerate(sorted(buffers, key=lambda b: b.first)):
        x = left + buffer.first / steps * plot_w
        w = (buffer.last - buffer.first + 1) / steps * plot_w
        y = top + plan.offsets[buffer.key] / size * plot_h
        h = max(buffer.size / size * plot_h, 1.0)
        color = ("#2f6f8f", "#c2703d", "#5b8a4a", "#8a5a9e")[index % 4]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{color}" fill-opacity="0.8" stroke="#1c1c1c" stroke-width="0.5"/>'
        )
    parts.append(f'<text x="{left}" y="{height - 12}" fill="#444">step 0</text>')
    parts.append(
        f'<text x="{left + plot_w}" y="{height - 12}" text-anchor="end" fill="#444">'
        f"step {steps - 1}</text>"
    )
    parts.append(f'<text x="8" y="{top + 10}" fill="#444">0</text>')
    parts.append(f'<text x="8" y="{top + plot_h}" fill="#444">{plan.size}</text>')
    parts.append(f'<text x="8" y="{top + plot_h + 14}" fill="#444">{unit}</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)
