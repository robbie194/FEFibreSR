#!/usr/bin/env python3
"""Run the real-data-compatible core-IWE simulation and reconstruction."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fibre_iwe import load_config, run_pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "baseline.yaml"
    )
    parser.add_argument(
        "--reuse-observations",
        action="store_true",
        help="skip simulation and reconstruct existing observation files",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="root containing observations/ and receiving results/",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.data_root is not None:
        config = replace(config, output_root=arguments.data_root.resolve())
    run_pipeline(
        config,
        generate=not arguments.reuse_observations,
        device_name=arguments.device,
    )


if __name__ == "__main__":
    main()
