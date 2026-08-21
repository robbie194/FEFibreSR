#!/usr/bin/env python3
"""Compare compact and B-spline blind motion models on shared observations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fibre_iwe.comparison import run_motion_comparison
from fibre_iwe.config import load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "complex_motion.yaml"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="root containing observations/ and receiving comparison/",
    )
    parser.add_argument(
        "--reuse-observations",
        action="store_true",
        help="compare models without regenerating APS/events",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.data_root is not None:
        config = replace(config, output_root=arguments.data_root.resolve())
    run_motion_comparison(
        config,
        generate=not arguments.reuse_observations,
        device_name=arguments.device,
    )


if __name__ == "__main__":
    main()
