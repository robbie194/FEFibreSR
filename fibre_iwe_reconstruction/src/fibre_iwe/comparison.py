"""Fair low-dimensional versus B-spline motion comparison on shared data."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_core_observations
from .io import write_json
from .motion import MotionEstimate, estimate_motion, jointly_refine_bspline_motion
from .output import (
    save_generation_preview,
    save_motion_diagnostics,
    save_reconstruction_results,
)
from .reconstruction import ReconstructionResult, reconstruct
from .simulation import generate_observations


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _optional_array(path: Path, key: str | None = None) -> np.ndarray | None:
    if not path.is_file():
        return None
    if key is None:
        return np.load(path)
    with np.load(path) as values:
        return values[key]


def _sample_truth(truth: np.ndarray, sample_count: int) -> np.ndarray:
    source_time = np.linspace(0, 1, len(truth))
    target_time = np.linspace(0, 1, sample_count)
    sampled = np.column_stack(
        [np.interp(target_time, source_time, truth[:, axis]) for axis in range(2)]
    )
    return sampled - sampled[0]


def _save_trajectory_comparison(
    output_dir: Path,
    motions: dict[str, MotionEstimate],
    truth: np.ndarray | None,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    colors = {"low_dimensional": "#3572A5", "bspline": "#C54B3C"}
    if truth is not None:
        truth = truth - truth[0]
        truth_time = np.linspace(0, 1, len(truth))
        axes[0, 0].plot(truth[:, 0], truth[:, 1], "k-", label="truth")
        axes[0, 1].plot(truth_time, truth[:, 0], "k-", label="truth")
        axes[1, 0].plot(truth_time, truth[:, 1], "k-", label="truth")
    for name, motion in motions.items():
        positions = motion.control_positions_xy
        time = np.linspace(0, 1, len(positions))
        label = "Low-dimensional" if name == "low_dimensional" else "B-spline"
        axes[0, 0].plot(
            positions[:, 0], positions[:, 1], "o-", ms=3, color=colors[name], label=label
        )
        axes[0, 1].plot(time, positions[:, 0], color=colors[name], label=label)
        axes[1, 0].plot(time, positions[:, 1], color=colors[name], label=label)
        if truth is not None:
            target = _sample_truth(truth, len(positions))
            error = np.linalg.norm(positions - target, axis=1)
            axes[1, 1].plot(time, error, color=colors[name], label=label)
    axes[0, 0].invert_yaxis()
    axes[0, 0].axis("equal")
    axes[0, 0].set_title("2-D trajectory")
    axes[0, 0].set(xlabel="x displacement (px)", ylabel="y displacement (px)")
    axes[0, 1].set_title("Horizontal displacement")
    axes[0, 1].set(xlabel="normalized time", ylabel="x (px)")
    axes[1, 0].set_title("Vertical displacement")
    axes[1, 0].set(xlabel="normalized time", ylabel="y (px)")
    axes[1, 1].set_title("Trajectory error")
    axes[1, 1].set(xlabel="normalized time", ylabel="Euclidean error (px)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "01_trajectory_model_comparison.png")
    plt.close(figure)


def _save_reconstruction_comparison(
    output_dir: Path,
    results: dict[str, ReconstructionResult],
    truth: np.ndarray | None,
) -> None:
    low = results["low_dimensional"].joint
    spline = results["bspline"].joint
    panels: list[tuple[str, np.ndarray, str, tuple[float, float]]] = []
    if truth is not None:
        panels.append(("Effective GT", truth, "gray", (0, 1)))
    panels.extend(
        (
            ("APS interpolation", results["low_dimensional"].initial_aps, "gray", (0, 1)),
            ("Low-dimensional joint", low, "gray", (0, 1)),
            ("B-spline joint", spline, "gray", (0, 1)),
        )
    )
    if truth is not None:
        error_limit = float(np.percentile(np.abs(low - truth), 99))
        panels.extend(
            (
                ("Low-dimensional error", np.abs(low - truth), "magma", (0, error_limit)),
                ("B-spline error", np.abs(spline - truth), "magma", (0, error_limit)),
            )
        )
    figure, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.8), dpi=190)
    for axis, (title, image, cmap, limits) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "02_reconstruction_model_comparison.png")
    plt.close(figure)


def _save_metric_comparison(output_dir: Path, summaries: dict[str, dict]) -> None:
    names = ("low_dimensional", "bspline")
    labels = ("Low-dimensional", "B-spline")
    colors = ("#3572A5", "#C54B3C")
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.7), dpi=180)
    fields = (
        ("motion", "trajectory_control_rmse_px", "Trajectory RMSE (px)"),
        ("metrics", "joint", "Joint PSNR (dB)"),
        ("metrics", "joint_ssim", "Joint SSIM"),
        ("data_fidelity", "temporal_iwe_cosine_similarity", "Temporal IWE cosine"),
    )
    for axis, (section, field, title) in zip(axes, fields, strict=True):
        values = []
        for name in names:
            summary = summaries[name]
            if section == "metrics" and field == "joint":
                value = summary["metrics"]["joint"]["psnr_db"]
            elif section == "metrics" and field == "joint_ssim":
                value = summary["metrics"]["joint"]["ssim"]
            else:
                value = summary[section][field]
            values.append(value)
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "03_metric_model_comparison.png")
    plt.close(figure)


def _save_bspline_history(output_dir: Path, motion: MotionEstimate) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=180)
    history = motion.refinement_history
    axes[0].plot(history[:, 0], history[:, 2], label="event")
    axes[0].plot(history[:, 0], history[:, 1], label="total")
    axes[0].set_title("B-spline observation refinement")
    joint = motion.joint_refinement_history
    if len(joint):
        axes[1].plot(joint[:, 0], joint[:, 2], label="APS")
        axes[1].plot(joint[:, 0], joint[:, 3], label="event")
        active_motion = np.flatnonzero(joint[:, 6] > 1e-10)
        if len(active_motion):
            axes[1].axvline(
                joint[active_motion[0], 0],
                color="k",
                ls="--",
                lw=1,
                label="motion released",
            )
    else:
        axes[1].text(
            0.5,
            0.5,
            "B-spline candidate rejected\nlow-dimensional path retained",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Joint image-motion refinement")
    for axis in axes:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "04_bspline_optimization.png")
    plt.close(figure)


def run_motion_comparison(
    config: ExperimentConfig,
    *,
    generate: bool = True,
    device_name: str | None = None,
) -> dict:
    """Run both motion models against exactly the same APS/events."""
    seed = int(config.values["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(device_name or str(config.values["project"]["device"]))
    config.output_root.mkdir(parents=True, exist_ok=True)
    generation = (
        generate_observations(config, device)
        if generate
        else {"status": "reused existing observations"}
    )
    observations = load_core_observations(config.observations_dir)
    comparison_dir = config.output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_generation_preview(comparison_dir, config.observations_dir)

    motions = {}
    results = {}
    for model_name in ("low_dimensional", "bspline"):
        print(f"[{model_name}] estimating motion", flush=True)
        motion_config = dict(config.values["motion_estimation"])
        motion_config["model"] = model_name
        motion = estimate_motion(observations, motion_config, device)
        if model_name == "bspline" and bool(
            motion_config.get("spline_joint_refinement", True)
        ):
            motion = jointly_refine_bspline_motion(
                observations,
                motion,
                motion_config,
                config.values["reconstruction"],
                device,
            )
        print(f"[{model_name}] reconstructing image", flush=True)
        result = reconstruct(
            observations, motion, config.values["reconstruction"], device
        )
        motions[model_name] = motion
        results[model_name] = result

    # Evaluation truth is loaded only after both inverse runs have finished.
    truth_motion = _optional_array(
        config.private_truth_dir / "motion_truth.npz", "shifts_xy_px"
    )
    truth_image = _optional_array(
        config.private_truth_dir / "object_effective_reference.npy"
    )
    summaries = {}
    for model_name in ("low_dimensional", "bspline"):
        model_dir = comparison_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        motion_metrics = save_motion_diagnostics(
            model_dir, observations, motions[model_name], truth_motion
        )
        summaries[model_name] = save_reconstruction_results(
            model_dir,
            results[model_name],
            truth_image,
            generation,
            motion_metrics,
        )

    _save_trajectory_comparison(comparison_dir, motions, truth_motion)
    _save_reconstruction_comparison(comparison_dir, results, truth_image)
    if truth_image is not None and truth_motion is not None:
        _save_metric_comparison(comparison_dir, summaries)
    _save_bspline_history(comparison_dir, motions["bspline"])
    low = summaries["low_dimensional"]
    spline = summaries["bspline"]
    improvement = {
        "temporal_iwe_cosine_gain": (
            spline["data_fidelity"]["temporal_iwe_cosine_similarity"]
            - low["data_fidelity"]["temporal_iwe_cosine_similarity"]
        )
    }
    if truth_motion is not None:
        improvement["trajectory_rmse_reduction_px"] = (
            low["motion"]["trajectory_control_rmse_px"]
            - spline["motion"]["trajectory_control_rmse_px"]
        )
    if truth_image is not None:
        improvement["joint_psnr_gain_db"] = (
            spline["metrics"]["joint"]["psnr_db"]
            - low["metrics"]["joint"]["psnr_db"]
        )
        improvement["joint_ssim_gain"] = (
            spline["metrics"]["joint"]["ssim"]
            - low["metrics"]["joint"]["ssim"]
        )
    report = {
        "models": summaries,
        "bspline_candidate_selected": motions["bspline"].model_name == "bspline",
        "bspline_improvement": improvement,
    }
    write_json(comparison_dir / "comparison_summary.json", report)
    return report
