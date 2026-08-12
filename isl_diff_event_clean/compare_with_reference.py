"""Compare clean reconstruction arrays with the frozen legacy baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurosr.output import compare_result_directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, nargs="?", default=Path("results/fig/tyf_test"))
    parser.add_argument(
        "reference",
        type=Path,
        nargs="?",
        default=Path("../isl_diff_event/results/fig/tyf_test"),
    )
    arguments = parser.parse_args()
    report = compare_result_directories(arguments.candidate, arguments.reference)
    print(json.dumps(report, indent=2))
    if not report or not all(
        item["numerically_equivalent"] for item in report.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
