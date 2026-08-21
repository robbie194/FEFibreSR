"""End-to-end generation, blind motion estimation, reconstruction and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_core_observations
from .motion import estimate_motion
from .output import (
    save_generation_preview,
    save_motion_diagnostics,
    save_reconstruction_results,
)
from .reconstruction import reconstruct
from .simulation import generate_observations


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _load_optional_array(path: Path, key: str | None = None) -> np.ndarray | None:
    """Load evaluation truth when present without making it an inverse input."""
    if not path.is_file():
        return None
    if key is None:
        return np.load(path)
    with np.load(path) as values:
        return values[key]


def run_pipeline(
    config: ExperimentConfig,
    *,
    generate: bool = True,
    device_name: str | None = None,
) -> dict:
    """Run the full experiment while enforcing the inverse data boundary."""
    seed = int(config.values["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(device_name or str(config.values["project"]["device"]))
    config.output_root.mkdir(parents=True, exist_ok=True)
    if generate:
        print("[1/4] generating raw APS/events and isolated truth", flush=True)
        generation_summary = generate_observations(config, device)
    else:
        generation_summary = {"status": "reused existing observations"}

    print("[2/4] loading only core_mask.npz and recording.h5", flush=True)
    observations = load_core_observations(config.observations_dir)
    print(
        f"      cores={len(observations.centres_xy)} "
        f"usable_events={len(observations.event_xy)}",
        flush=True,
    )
    print("[3/4] estimating trajectory from APS/event consistency", flush=True)
    motion = estimate_motion(observations, config.values["motion_estimation"], device)
    print(
        f"      estimated endpoint={motion.control_positions_xy[-1].round(3).tolist()}",
        flush=True,
    )
    print("[4/4] reconstructing effective image with APS + core-IWE", flush=True)
    result = reconstruct(observations, motion, config.values["reconstruction"], device)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    save_generation_preview(config.results_dir, config.observations_dir)
    truth_motion = _load_optional_array(
        config.private_truth_dir / "motion_truth.npz", "shifts_xy_px"
    )
    motion_metrics = save_motion_diagnostics(
        config.results_dir, observations, motion, truth_motion
    )
    truth = _load_optional_array(
        config.private_truth_dir / "object_effective_reference.npy"
    )
    summary = save_reconstruction_results(
        config.results_dir, result, truth, generation_summary, motion_metrics
    )
    if summary["metrics"] is not None:
        print("      joint metrics:", summary["metrics"]["joint"], flush=True)
    else:
        print("      data fidelity:", summary["data_fidelity"], flush=True)
    print(f"      saved: {config.results_dir}", flush=True)
    return summary
