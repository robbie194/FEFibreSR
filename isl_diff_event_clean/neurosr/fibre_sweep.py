"""Run and compare the two-dimensional GRIN blur reconstruction cases."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .fibre_config import FibreReconstructionConfig
from .fibre_pipeline import run_fibre_reconstruction
from .fibre_output import image_metrics


XY_CASES = (
    ("sigma0", "phase2_xy_usaf_sigma0.yaml", "phase2_xy_sigma0"),
    ("sigma08", "phase2_xy_usaf_sigma08.yaml", "phase2_xy_sigma08"),
)


def _save_cross_case_comparison(
    output_dir: Path,
    case_outputs: dict[str, Path],
) -> None:
    first_output = next(iter(case_outputs.values()))
    first_summary = json.loads((first_output / "run_summary.json").read_text())
    y0, y1, x0, x1 = first_summary["observable_crop_yx"]
    crop = np.s_[y0:y1, x0:x1]
    truth = np.load(first_output / "truth_for_evaluation_only.npy")[crop]
    panels = [("Truth", truth)]
    for label, case_output in case_outputs.items():
        for mode, mode_title in (("aps_only", "APS only"), ("joint", "APS + events")):
            path = case_output / mode / "reconstruction.npy"
            if path.is_file():
                panels.append((f"{label}: {mode_title}", np.load(path)[crop]))

    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=200)
    for axis, (title, image) in zip(np.atleast_1d(axes), panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "sigma_comparison.png")
    plt.close(figure)


def _compare_with_horizontal_baseline(
    project_root: Path,
    output_dir: Path,
    case_outputs: dict[str, Path],
) -> dict:
    """Compare all joint results on the exact common observable crop."""
    horizontal_output = project_root / "results" / "fibre_neurosr" / "phase1_usaf"
    all_outputs = {"horizontal_sigma0": horizontal_output, **case_outputs}
    summaries = {
        label: json.loads((path / "run_summary.json").read_text())
        for label, path in all_outputs.items()
    }
    crops = [summary["observable_crop_yx"] for summary in summaries.values()]
    y0 = max(crop[0] for crop in crops)
    y1 = min(crop[1] for crop in crops)
    x0 = max(crop[2] for crop in crops)
    x1 = min(crop[3] for crop in crops)
    common_crop = np.s_[y0:y1, x0:x1]

    truth = np.load(horizontal_output / "truth_for_evaluation_only.npy")[common_crop]
    metrics = {}
    panels = [("Truth", truth)]
    titles = {
        "horizontal_sigma0": "horizontal only, sigma=0",
        "sigma0": "2-D scan, sigma=0",
        "sigma08": "2-D scan, sigma=0.8",
    }
    for label, path in all_outputs.items():
        reconstruction = np.load(path / "joint" / "reconstruction.npy")[common_crop]
        metrics[label] = image_metrics(reconstruction, truth)
        panels.append((titles[label], reconstruction))

    baseline = metrics["horizontal_sigma0"]
    delta_from_horizontal = {
        label: {
            metric: values[metric] - baseline[metric]
            for metric in (
                "psnr_db",
                "ssim",
                "correlation",
                "gradient_x_correlation",
                "gradient_y_correlation",
            )
        }
        for label, values in metrics.items()
        if label != "horizontal_sigma0"
    }

    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=200)
    for axis, (title, image) in zip(np.atleast_1d(axes), panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_comparison.png")
    plt.close(figure)
    return {
        "common_crop_yx": [y0, y1, x0, x1],
        "metrics": metrics,
        "delta_from_horizontal": delta_from_horizontal,
    }


def run_xy_sigma_sweep(
    *,
    project_root: Path,
    iterations: int,
    device: str,
    tv_weight: float,
    modes: Iterable[str],
    run_simulation: bool,
    run_reconstruction: bool,
) -> dict:
    """Generate both datasets, reconstruct them, and save a shared summary."""
    from fibre_sim.config import load_config, output_root
    from fibre_sim.pipeline import run_all

    simulation_root = project_root.parent / "fibre_frame_event_sim"
    sweep_output = project_root / "results" / "fibre_neurosr" / "phase2_xy_sigma_sweep"
    sweep_output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    case_outputs: dict[str, Path] = {}

    for label, config_name, output_name in XY_CASES:
        simulation_config = simulation_root / "configs" / config_name
        simulation_cfg = load_config(simulation_config)
        if run_simulation:
            print(f"[{label}] generating complete simulation", flush=True)
            run_all(simulation_cfg)
        simulation_output = output_root(simulation_cfg)
        validation = json.loads(
            (simulation_output / "07_validation" / "validation_report.json").read_text()
        )
        event_stats = json.loads(
            (simulation_output / "06_events" / "stats.json").read_text()
        )
        reconstruction_output = (
            project_root / "results" / "fibre_neurosr" / output_name
        )
        if run_reconstruction:
            reconstruction_config = replace(
                FibreReconstructionConfig(),
                simulation_config=simulation_config,
                output_dir=reconstruction_output,
                iterations=iterations,
                device=device,
                tv_weight=tv_weight,
            )
            print(f"[{label}] reconstructing APS/events", flush=True)
            reconstruction_summary = run_fibre_reconstruction(
                reconstruction_config, modes
            )
        else:
            reconstruction_summary = json.loads(
                (reconstruction_output / "run_summary.json").read_text()
            )
        summaries[label] = {
            "simulation_validation": validation,
            "event_stats": event_stats,
            "reconstruction": reconstruction_summary,
        }
        case_outputs[label] = reconstruction_output

    _save_cross_case_comparison(sweep_output, case_outputs)
    horizontal_comparison = _compare_with_horizontal_baseline(
        project_root, sweep_output, case_outputs
    )
    quality_passed = all(
        summary["simulation_validation"]["all_passed"]
        and all(summary["reconstruction"].get("quality_checks", {}).values())
        for summary in summaries.values()
    )
    sweep_summary = {
        "experiment": "two_dimensional_L_scan_GRIN_sigma_sweep",
        "trajectory": [[0.0, 0.0], [4.5, 0.0], [4.5, 4.5]],
        "cases": summaries,
        "comparison_to_horizontal_baseline": horizontal_comparison,
        "all_directional_quality_checks_passed": quality_passed,
    }
    (sweep_output / "run_summary.json").write_text(
        json.dumps(sweep_summary, indent=2), encoding="utf-8"
    )
    if not quality_passed:
        raise RuntimeError(
            "at least one simulation or directional reconstruction check failed; "
            f"see {sweep_output / 'run_summary.json'}"
        )
    return sweep_summary
