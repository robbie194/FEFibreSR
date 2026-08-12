from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fibre_sim.config import load_config


def load_cli_config(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase1_usaf.yaml",
        help="simulation YAML (default: configs/phase1_usaf.yaml)",
    )
    args = parser.parse_args()
    return load_config(args.config)

