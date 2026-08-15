#!/usr/bin/env python3
"""Reconstruct a honeycomb-free distal frame from fibre APS and events."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATOR_SRC = PROJECT_ROOT.parent / "fibre_frame_event_sim" / "src"
if str(SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SRC))

from neurosr.fibre_config import FibreReconstructionConfig
from neurosr.fibre_pipeline import run_fibre_reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fibre-aware Scheme-A reconstruction using core APS and events"
    )
    parser.add_argument("--config", type=Path, help="simulation YAML path")
    parser.add_argument("--output", type=Path, help="result directory")
    parser.add_argument(
        "--iterations", type=int, help="iterations per reconstruction mode"
    )
    parser.add_argument(
        "--device", default=None, help="PyTorch device, e.g. cuda or cpu"
    )
    parser.add_argument(
        "--latent-size", type=int, help="square latent reconstruction size"
    )
    parser.add_argument("--event-weight", type=float, help="core-event loss weight")
    parser.add_argument(
        "--tv-weight", type=float, help="total-variation regularisation weight"
    )
    parser.add_argument(
        "--event-time-stride",
        type=int,
        help="use every Nth simulated timestamp",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("aps_only", "events_only", "joint"),
        default=("aps_only", "events_only", "joint"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FibreReconstructionConfig()
    changes = {}
    if args.config is not None:
        changes["simulation_config"] = args.config
    if args.output is not None:
        changes["output_dir"] = args.output
    if args.iterations is not None:
        changes["iterations"] = args.iterations
    if args.device is not None:
        changes["device"] = args.device
    if args.latent_size is not None:
        changes["latent_shape"] = (args.latent_size, args.latent_size)
    if args.event_weight is not None:
        changes["event_weight"] = args.event_weight
    if args.tv_weight is not None:
        changes["tv_weight"] = args.tv_weight
    if args.event_time_stride is not None:
        changes["event_time_stride"] = args.event_time_stride
    config = replace(config, **changes)
    summary = run_fibre_reconstruction(config, args.modes)
    print(f"Saved reconstruction to {config.resolve(PROJECT_ROOT).output_dir}")
    joint = summary["reconstructions"].get("joint")
    if joint:
        print("Joint metrics:", joint["metrics_on_observable_region"])


if __name__ == "__main__":
    main()
