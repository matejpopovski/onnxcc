"""Long run of the property tests with fresh random seeds.

    uv run python -m experiments.fuzz_campaign

CI runs the same tests with a fixed seed and a small budget. Writes results/fuzz_campaign.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from .models import REPO, RESULTS

SCALE = 12
TESTS = ["tests/test_planner.py", "tests/test_fuzz_graphs.py", "tests/test_fuzz_windows.py"]


def main() -> None:
    environment = {
        **os.environ,
        "HYPOTHESIS_PROFILE": "campaign",
        "ONNXCC_FUZZ_SCALE": str(SCALE),
    }
    started = time.perf_counter()
    # One process per file: Hypothesis statistics do not survive pytest-xdist.
    runs = [
        subprocess.Popen(
            [sys.executable, "-m", "pytest", path, "--hypothesis-show-statistics"],
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for path in TESTS
    ]
    outputs = [run.communicate()[0] for run in runs]
    counts = re.findall(r"(\d+) passing, (\d+) failing", "\n".join(outputs))
    report = {
        "exit_codes": [run.returncode for run in runs],
        "seconds": round(time.perf_counter() - started),
        "budget_scale": SCALE,
        "property_tests": len(counts),
        "passing_examples": sum(int(passing) for passing, _ in counts),
        "failing_examples": sum(int(failing) for _, failing in counts),
    }
    (RESULTS / "fuzz_campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
