"""Run the clean, reproducible NeuroSR reconstruction experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from neurosr import ExperimentConfig, run_experiment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ExperimentConfig()
    parser.add_argument("--input", type=Path, default=defaults.input_path)
    parser.add_argument("--output", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--iterations",
        type=int,
        default=defaults.reconstruction_iterations,
        help="iterations per optimization stage (2100 reproduces the baseline)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = replace(
        ExperimentConfig(),
        input_path=arguments.input,
        output_dir=arguments.output,
        motion_iterations=arguments.iterations,
        reconstruction_iterations=arguments.iterations,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
