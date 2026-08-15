#!/usr/bin/env python3
"""Generate and reconstruct the sigma=0/0.8 two-dimensional scan datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATOR_SRC = PROJECT_ROOT.parent / "fibre_frame_event_sim" / "src"
if str(SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SRC))

from neurosr.fibre_sweep import run_xy_sigma_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2-D scan reconstruction for GRIN sigma 0 and 0.8 um"
    )
    parser.add_argument("--iterations", type=int, default=1_200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.005,
        help="2-D reconstruction TV weight (default: 0.005)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("aps_only", "events_only", "joint"),
        default=("aps_only", "events_only", "joint"),
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="reuse existing simulator outputs",
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="reuse existing reconstruction outputs and only rebuild summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_xy_sigma_sweep(
        project_root=PROJECT_ROOT,
        iterations=args.iterations,
        device=args.device,
        tv_weight=args.tv_weight,
        modes=args.modes,
        run_simulation=not args.skip_simulation,
        run_reconstruction=not args.skip_reconstruction,
    )
    output = PROJECT_ROOT / "results" / "fibre_neurosr" / "phase2_xy_sigma_sweep"
    print(f"Saved cross-case comparison to {output}")
    print(
        "All directional checks passed:",
        summary["all_directional_quality_checks_passed"],
    )


if __name__ == "__main__":
    main()
