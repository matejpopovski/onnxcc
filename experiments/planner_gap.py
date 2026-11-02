"""How far is the planner from optimal? Exact search on small random instances.

    uv run python -m experiments.planner_gap

Any arena layout can be normalized to first-fit under some ordering, so trying every ordering of up
to seven buffers finds the true optimum. Writes results/planner_gap.json.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

from onnxcc.memcheck import check_intervals
from onnxcc.planner import Buffer, _first_fit, extent, make_plan, peak_live

from .models import RESULTS

INSTANCES = 1000
BUFFERS = 7


def exact(buffers: list[Buffer]) -> int:
    return min(extent(buffers, _first_fit(order)) for order in itertools.permutations(buffers))


def main() -> None:
    rng = random.Random(0)
    gaps = []
    for _ in range(INSTANCES):
        buffers = []
        for index in range(BUFFERS):
            first = rng.randrange(0, 10)
            buffers.append(
                Buffer(f"b{index}", rng.randrange(1, 100), first, first + rng.randrange(0, 5))
            )
        plan = make_plan(buffers)
        check_intervals(
            {b.key: (b.size, b.first, b.last) for b in buffers}, plan.offsets, plan.size
        )
        best = exact(buffers)
        assert peak_live(buffers) <= best <= plan.size
        gaps.append(plan.size / best)
    report: dict[str, Any] = {
        "instances": INSTANCES,
        "buffers_per_instance": BUFFERS,
        "optimal_fraction": sum(gap == 1.0 for gap in gaps) / INSTANCES,
        "mean_ratio_to_optimal": sum(gaps) / INSTANCES,
        "worst_ratio_to_optimal": max(gaps),
    }
    (RESULTS / "planner_gap.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
