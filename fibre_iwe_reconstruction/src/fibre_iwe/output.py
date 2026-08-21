"""Diagnostics and evaluation kept separate from the inverse inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .data import CoreObservations
from .io import load_core_mask, load_recording, write_json
from .motion import MotionEstimate
from .reconstruction import ReconstructionResult


def _metrics(image: np.ndarray, truth: np.ndarray, border: int = 12) -> dict[str, float]:
    candidate = image[border:-border, border:-border]
    reference = truth[border:-border, border:-border]
    return {
        "psnr_db": float(peak_signal_noise_ratio(reference, candidate, data_range=1.0)),
        "ssim": float(structural_similarity(reference, candidate, data_range=1.0)),
        "correlation": float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]),
        "rmse": float(np.sqrt(np.mean((reference - candidate) ** 2))),
    }


def _data_fidelity_metrics(result: ReconstructionResult) -> dict[str, float]:
    aps_observed = result.aps_observed_predicted[:, 0]
    aps_predicted = result.aps_observed_predicted[:, 1]
    observed_iwe = result.observed_iwe.ravel().astype(np.float64)
    predicted_iwe = result.predicted_iwe.ravel().astype(np.float64)
    observed_iwe -= observed_iwe.mean()
    predicted_iwe -= predicted_iwe.mean()
    denominator = np.linalg.norm(observed_iwe) * np.linalg.norm(predicted_iwe)
    observed_bins = result.observed_iwe_bins.reshape(len(result.observed_iwe_bins), -1)
    predicted_bins = result.predicted_iwe_bins.reshape(
        len(result.predicted_iwe_bins), -1
    )
    observed_bins = observed_bins - observed_bins.mean(1, keepdims=True)
    predicted_bins = predicted_bins - predicted_bins.mean(1, keepdims=True)
    observed_norms = np.linalg.norm(observed_bins, axis=1)
    predicted_norms = np.linalg.norm(predicted_bins, axis=1)
    active = observed_norms > 1e-12
    temporal_cosines = np.sum(observed_bins * predicted_bins, axis=1) / np.maximum(
        observed_norms * predicted_norms, 1e-12
    )
    return {
        "aps_reprojection_rmse": float(
            np.sqrt(np.mean(np.square(aps_observed - aps_predicted)))
        ),
        "iwe_cosine_similarity": float(
            np.dot(observed_iwe, predicted_iwe) / max(denominator, 1e-12)
        ),
        "temporal_iwe_cosine_similarity": float(
            np.average(temporal_cosines[active], weights=observed_norms[active])
        ),
    }


def save_generation_preview(output_dir: Path, observations_dir: Path) -> None:
    mask = load_core_mask(observations_dir / "core_mask.npz")
    recording = load_recording(observations_dir / "recording.h5")
    event_map = np.zeros(recording.sensor_shape, dtype=np.float32)
    if len(recording.events):
        x = recording.events[:, 1].astype(np.int32)
        y = recording.events[:, 2].astype(np.int32)
        valid = (
            (x >= 0)
            & (x < recording.sensor_shape[1])
            & (y >= 0)
            & (y < recording.sensor_shape[0])
        )
        np.add.at(event_map, (y[valid], x[valid]), recording.events[valid, 3])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    axes[0].imshow(mask.labels, cmap="turbo")
    axes[0].set_title("Known core mask")
    axes[1].imshow(recording.aps_frame, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Raw APS observation")
    limit = np.percentile(np.abs(event_map), 99.5)
    axes[2].imshow(event_map, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[2].set_title("Raw sensor events")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "01_generated_observations.png")
    plt.close(figure)


def save_motion_diagnostics(
    output_dir: Path,
    observations: CoreObservations,
    motion: MotionEstimate,
    truth_path: np.ndarray | None,
) -> dict[str, Any]:
    from .render import render_iwe
    import torch

    device = torch.device("cpu")
    xy = torch.as_tensor(observations.event_xy)
    time = torch.as_tensor(observations.event_time_normalized)
    polarity = torch.as_tensor(observations.event_polarity)
    initial = torch.as_tensor(motion.initial_control_positions_xy)
    final = torch.as_tensor(motion.control_positions_xy)
    initial_iwe = render_iwe(
        xy, time, polarity, initial, observations.sensor_shape, 1.0, signed=False
    ).numpy()
    final_iwe = render_iwe(
        xy, time, polarity, final, observations.sensor_shape, 1.0, signed=False
    ).numpy()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    axes[0].imshow(initial_iwe, cmap="magma")
    axes[0].set_title("IWE after linear endpoint")
    axes[1].imshow(final_iwe, cmap="magma")
    axes[1].set_title(f"IWE after {motion.model_name} path")
    axes[2].plot(
        motion.initial_control_positions_xy[:, 0],
        motion.initial_control_positions_xy[:, 1],
        "o--",
        label="linear",
    )
    axes[2].plot(
        motion.control_positions_xy[:, 0],
        motion.control_positions_xy[:, 1],
        "o-",
        label=motion.model_name,
    )
    metrics: dict[str, Any] = {
        "model": motion.model_name,
        "initial_iwe_contrast": float(np.var(initial_iwe)),
        "final_iwe_contrast": float(np.var(final_iwe)),
        "estimated_endpoint_xy_px": motion.control_positions_xy[-1].tolist(),
    }
    if len(motion.coarse_scores):
        cmax_best = motion.coarse_scores[np.argmax(motion.coarse_scores[:, 3])]
        metrics["cmax_endpoint_xy_px"] = cmax_best[:2].tolist()
    if len(motion.endpoint_validation_scores):
        validation_best = motion.endpoint_validation_scores[
            np.argmin(motion.endpoint_validation_scores[:, 3])
        ]
        metrics["aps_event_endpoint_xy_px"] = validation_best[:2].tolist()
    if motion.model_name.startswith("low_dimensional") and len(
        motion.path_validation_scores
    ):
        path_best = motion.path_validation_scores[
            np.argmin(motion.path_validation_scores[:, 2])
        ]
        metrics["estimated_curvature_px"] = float(path_best[0])
        metrics["estimated_easing_px"] = float(path_best[1])
    if len(motion.spline_control_positions_xy):
        metrics["spline_control_count"] = len(motion.spline_control_positions_xy)
        metrics["spline_final_objective"] = float(motion.refinement_history[-1, 1])
        metrics["spline_candidate_event_improvement_fraction"] = float(
            motion.candidate_event_improvement_fraction
        )
    if truth_path is not None:
        truth_time = np.linspace(0, 1, len(truth_path))
        control_time = np.linspace(0, 1, len(motion.control_positions_xy))
        truth_controls = np.column_stack(
            [np.interp(control_time, truth_time, truth_path[:, axis]) for axis in range(2)]
        )
        truth_controls -= truth_controls[0]
        axes[2].plot(truth_controls[:, 0], truth_controls[:, 1], "k-", label="truth")
        metrics["trajectory_control_rmse_px"] = float(
            np.sqrt(np.mean((motion.control_positions_xy - truth_controls) ** 2))
        )
        metrics["endpoint_error_px"] = float(
            np.linalg.norm(motion.control_positions_xy[-1] - truth_controls[-1])
        )
    axes[2].invert_yaxis()
    axes[2].axis("equal")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Trajectory (evaluation truth optional)")
    axes[0].axis("off")
    axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "02_motion_estimation.png")
    plt.close(figure)
    np.savez(
        output_dir / "motion_estimate.npz",
        initial_control_positions_xy=motion.initial_control_positions_xy,
        control_positions_xy=motion.control_positions_xy,
        coarse_scores=motion.coarse_scores,
        endpoint_validation_scores=motion.endpoint_validation_scores,
        path_validation_scores=motion.path_validation_scores,
        model_name=np.asarray(motion.model_name),
        spline_control_positions_xy=motion.spline_control_positions_xy,
        refinement_history=motion.refinement_history,
        joint_refinement_history=motion.joint_refinement_history,
        candidate_event_improvement_fraction=np.asarray(
            motion.candidate_event_improvement_fraction
        ),
    )
    return metrics


def save_reconstruction_results(
    output_dir: Path,
    result: ReconstructionResult,
    truth: np.ndarray | None,
    generation_summary: dict,
    motion_metrics: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "initial_aps": result.initial_aps,
        "event_only": result.event_only,
        "aps_only": result.aps_only,
        "joint": result.joint,
        "observed_iwe": result.observed_iwe,
        "predicted_iwe": result.predicted_iwe,
        "observability": result.observability,
        "observed_iwe_bins": result.observed_iwe_bins,
        "predicted_iwe_bins": result.predicted_iwe_bins,
        "event_flow_xy_bins": result.event_flow_xy_bins,
        "loss_history_event_only": result.loss_history_event_only,
        "loss_history_aps": result.loss_history_aps,
        "loss_history_joint": result.loss_history_joint,
        "aps_observed_predicted": result.aps_observed_predicted,
    }
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)
    panels = [
        ("APS interpolation", result.initial_aps),
        ("Event-only image\n(shared blind motion)", result.event_only),
        ("APS-only deblur", result.aps_only),
        ("APS + core-IWE", result.joint),
    ]
    if truth is not None:
        panels.insert(0, ("Effective GT (evaluation only)", truth))
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=200)
    for axis, (title, image) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "03_reconstruction_comparison.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    limit_obs = np.percentile(np.abs(result.observed_iwe), 99.5)
    limit_pred = np.percentile(np.abs(result.predicted_iwe), 99.5)
    axes[0].imshow(result.observed_iwe, cmap="coolwarm", vmin=-limit_obs, vmax=limit_obs)
    axes[0].set_title("Observed core-IWE")
    axes[1].imshow(result.predicted_iwe, cmap="coolwarm", vmin=-limit_pred, vmax=limit_pred)
    axes[1].set_title("Predicted gradient IWE")
    axes[2].imshow(result.observability, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Continuous observability map")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "04_iwe_and_observability.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    axes[0].plot(result.loss_history_joint[:, 0], result.loss_history_joint[:, 1], label="total")
    axes[0].plot(result.loss_history_joint[:, 0], result.loss_history_joint[:, 2], label="APS")
    axes[0].plot(result.loss_history_joint[:, 0], result.loss_history_joint[:, 3], label="IWE")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Joint losses (per scale)")
    axes[1].scatter(
        result.aps_observed_predicted[:, 0], result.aps_observed_predicted[:, 1], s=3
    )
    axes[1].plot((0, 1), (0, 1), "k--", linewidth=1)
    axes[1].set(xlabel="observed core APS", ylabel="predicted core APS", title="APS reprojection")
    figure.tight_layout()
    figure.savefig(output_dir / "05_loss_and_reprojection.png")
    plt.close(figure)

    displayed_bins = np.linspace(
        0, len(result.observed_iwe_bins) - 1, min(4, len(result.observed_iwe_bins))
    ).round().astype(int)
    figure, axes = plt.subplots(
        2, len(displayed_bins), figsize=(3.2 * len(displayed_bins), 6.2), dpi=180
    )
    axes = np.asarray(axes).reshape(2, -1)
    for column, bin_index in enumerate(displayed_bins):
        observed = result.observed_iwe_bins[bin_index]
        predicted = result.predicted_iwe_bins[bin_index]
        observed_limit = max(float(np.percentile(np.abs(observed), 99.5)), 1e-8)
        predicted_limit = max(float(np.percentile(np.abs(predicted), 99.5)), 1e-8)
        axes[0, column].imshow(
            observed, cmap="coolwarm", vmin=-observed_limit, vmax=observed_limit
        )
        axes[1, column].imshow(
            predicted, cmap="coolwarm", vmin=-predicted_limit, vmax=predicted_limit
        )
        axes[0, column].set_title(f"Observed bin {bin_index + 1}")
        axes[1, column].set_title(f"Predicted bin {bin_index + 1}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "06_temporal_iwe_bins.png")
    plt.close(figure)

    reference_metrics = None
    if truth is not None:
        reference_metrics = {
            "aps_interpolation": _metrics(result.initial_aps, truth),
            "event_only": _metrics(result.event_only, truth),
            "aps_only": _metrics(result.aps_only, truth),
            "joint": _metrics(result.joint, truth),
        }
    summary = {
        "inverse_input_audit": {
            "used": [
                "observations/core_mask.npz (labels only)",
                "observations/recording.h5",
            ],
            "not_used_by_reconstruction": [
                "private_truth/object_effective_reference.npy",
                "private_truth/motion_truth.npz",
                "simulation event threshold",
                "simulation PSF",
            ],
        },
        "generation": generation_summary,
        "motion": motion_metrics,
        "data_fidelity": _data_fidelity_metrics(result),
        "ground_truth_available": truth is not None,
        "event_only_note": (
            "Events constrain structure but not absolute log-intensity offset or scale; "
            "a configured mean/std gauge is used without APS."
        ),
        "metrics": reference_metrics,
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
