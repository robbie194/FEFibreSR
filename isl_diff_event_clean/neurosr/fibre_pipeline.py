"""End-to-end fibre-aware APS/event reconstruction pipeline."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata

from .fibre_config import FibreReconstructionConfig
from .fibre_data import FibreObservations, load_fibre_observations
from .fibre_event_loss import (
    isotropic_total_variation,
    predict_cumulative_event_change,
    robust_event_loss,
    trapezoidal_average,
)
from .fibre_forward import FibreCoreForward, expand_latent_image
from .fibre_output import image_metrics, save_comparison, save_mode_result


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _time_indices(sample_count: int, stride: int) -> np.ndarray:
    indices = np.arange(0, sample_count, stride, dtype=np.int64)
    if indices[-1] != sample_count - 1:
        indices = np.append(indices, sample_count - 1)
    return indices


def _initial_image(
    observations: FibreObservations,
    source_shape: tuple[int, int],
    source_pixel_size_um: float,
    grin_magnification: float,
) -> np.ndarray:
    """Interpolate core APS values at their exposure-midpoint object positions."""
    height, width = source_shape
    effective_xy = observations.core_centres_xy_um / float(grin_magnification)
    effective_xy = effective_xy - observations.shifts_xy_um.mean(axis=0)
    x_um = (np.arange(width) - (width - 1) / 2) * float(source_pixel_size_um)
    y_um = (np.arange(height) - (height - 1) / 2) * float(source_pixel_size_um)
    grid_x, grid_y = np.meshgrid(x_um, y_um)
    linear = griddata(
        effective_xy,
        observations.core_aps,
        (grid_x, grid_y),
        method="linear",
    )
    nearest = griddata(
        effective_xy,
        observations.core_aps,
        (grid_x, grid_y),
        method="nearest",
    )
    interpolated = np.where(np.isfinite(linear), linear, nearest)
    return np.clip(interpolated, 0.0, 1.0).astype(np.float32)


def _observable_crop(
    observations: FibreObservations,
    source_shape: tuple[int, int],
    source_pixel_size_um: float,
    grin_magnification: float,
    core_radius_um: float,
) -> tuple[slice, slice]:
    centres = observations.core_centres_xy_um / float(grin_magnification)
    sampled_x = centres[:, 0, None] - observations.shifts_xy_um[None, :, 0]
    sampled_y = centres[:, 1, None] - observations.shifts_xy_um[None, :, 1]
    x_limits = (sampled_x.min() - core_radius_um, sampled_x.max() + core_radius_um)
    y_limits = (sampled_y.min() - core_radius_um, sampled_y.max() + core_radius_um)
    height, width = source_shape

    def to_index(value: float, length: int) -> int:
        return int(round((length - 1) / 2 + value / source_pixel_size_um))

    x0 = max(0, to_index(x_limits[0], width))
    x1 = min(width, to_index(x_limits[1], width) + 1)
    y0 = max(0, to_index(y_limits[0], height))
    y1 = min(height, to_index(y_limits[1], height) + 1)
    return slice(y0, y1), slice(x0, x1)


def _make_forward(
    simulation_cfg: dict,
    observations: FibreObservations,
    device: torch.device,
) -> FibreCoreForward:
    from fibre_sim.config import derived_parameters

    derived = derived_parameters(simulation_cfg)
    fibre_cfg = simulation_cfg["fibre"]
    grin_cfg = simulation_cfg["grin"]
    return FibreCoreForward(
        source_shape=derived["source_shape_px"],
        source_pixel_size_um=float(simulation_cfg["source"]["pixel_size_um"]),
        fibre_shape=derived["fibre_shape_px"],
        fibre_pixel_size_um=float(grin_cfg["fibre_grid_pixel_size_um"]),
        grin_magnification=float(grin_cfg["magnification"]),
        grin_sigma_um=float(grin_cfg["sigma_um"]),
        grin_transmission=float(grin_cfg["transmission"]),
        fibre_transmission=float(fibre_cfg["transmission"]),
        core_centres_xy_um=observations.core_centres_xy_um,
        core_diameter_um=float(fibre_cfg["core_diameter_um"]),
        aperture_supersample=int(fibre_cfg["aperture_supersample"]),
    ).to(device)


@torch.no_grad()
def validate_forward_model(
    model: FibreCoreForward,
    observations: FibreObservations,
    simulation_cfg: dict,
    output_root: Path,
    device: torch.device,
) -> dict[str, float]:
    """Use saved truth only to verify the implementation, never as inverse input."""
    import h5py

    truth_object = np.load(output_root / "00_source" / "object_intensity.npy")
    with h5py.File(output_root / "03_fibre" / "fibre_sequence.h5", "r") as handle:
        truth_signals = handle["core_signals"][:]
    prediction_batches = []
    image = torch.as_tensor(truth_object, dtype=torch.float32, device=device)
    shifts = torch.as_tensor(observations.shifts_xy_um, device=device)
    for start in range(0, len(shifts), 32):
        prediction_batches.append(model(image, shifts[start : start + 32]).cpu())
    prediction = torch.cat(prediction_batches).numpy()
    signal_error = prediction - truth_signals

    gain = torch.as_tensor(observations.calibration.gain, device=device)
    predicted_change = predict_cumulative_event_change(
        torch.as_tensor(prediction, device=device),
        gain,
        float(simulation_cfg["events"]["input_white_dn"]),
    ).cpu().numpy()
    event_error = predicted_change - observations.cumulative_event_change
    random_image = torch.rand_like(image) * 0.95 + 0.05
    random_signals = model(random_image, shifts[_time_indices(len(shifts), 5)])
    random_change = predict_cumulative_event_change(
        random_signals, gain, float(simulation_cfg["events"]["input_white_dn"])
    )
    random_observed = torch.as_tensor(
        observations.cumulative_event_change[_time_indices(len(shifts), 5)], device=device
    )
    return {
        "core_signal_mae": float(np.mean(np.abs(signal_error))),
        "core_signal_rmse": float(np.sqrt(np.mean(signal_error**2))),
        "core_signal_max_abs_error": float(np.max(np.abs(signal_error))),
        "core_signal_correlation": float(
            np.corrcoef(prediction.ravel(), truth_signals.ravel())[0, 1]
        ),
        "truth_event_residual_mae": float(np.mean(np.abs(event_error))),
        "truth_event_residual_rmse": float(np.sqrt(np.mean(event_error**2))),
        "truth_event_correlation": float(
            np.corrcoef(
                predicted_change.ravel(),
                observations.cumulative_event_change.ravel(),
            )[0, 1]
        ),
        "random_event_residual_mae": float(
            torch.mean(torch.abs(random_change - random_observed)).cpu()
        ),
    }


def _optimise(
    *,
    mode: str,
    model: FibreCoreForward,
    observations: FibreObservations,
    simulation_cfg: dict,
    config: FibreReconstructionConfig,
    initial_image: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    use_aps = mode in {"aps_only", "joint"}
    use_events = mode in {"events_only", "joint"}
    if not (use_aps or use_events):
        raise ValueError(f"unknown reconstruction mode: {mode}")

    if mode == "events_only":
        # Events constrain temporal contrast but not absolute brightness. A
        # neutral constant avoids quietly injecting APS structure into this
        # diagnostic baseline.
        initial = torch.full(model.source_shape, 0.5, device=device)
    else:
        initial = torch.as_tensor(initial_image, device=device)
    latent = F.interpolate(
        initial[None, None], size=config.latent_shape, mode="area"
    )[0, 0].detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([latent], lr=config.learning_rate)
    indices = _time_indices(len(observations.timestamps_s), config.event_time_stride)
    shifts = torch.as_tensor(observations.shifts_xy_um[indices], device=device)
    times = torch.as_tensor(
        observations.timestamps_s[indices], device=device, dtype=torch.float32
    )
    observed_aps = torch.as_tensor(observations.core_aps, device=device)
    observed_events = torch.as_tensor(
        observations.cumulative_event_change[indices], device=device
    )
    gain = torch.as_tensor(observations.calibration.gain, device=device)
    source_shape = model.source_shape
    history: list[tuple[float, float, float, float]] = []
    white_dn = float(simulation_cfg["events"]["input_white_dn"])

    for iteration in range(config.iterations):
        optimiser.zero_grad(set_to_none=True)
        image = expand_latent_image(latent, source_shape)
        core_signals = model(image, shifts)
        predicted_aps = trapezoidal_average(core_signals, times)
        aps_loss = F.mse_loss(predicted_aps, observed_aps)
        predicted_events = predict_cumulative_event_change(core_signals, gain, white_dn)
        event_loss = robust_event_loss(
            predicted_events.mean(dim=-1),
            observed_events.mean(dim=-1),
            config.huber_beta,
        )
        tv_loss = isotropic_total_variation(latent)
        total = (
            (aps_loss if use_aps else 0.0)
            + (config.event_weight * event_loss if use_events else 0.0)
            + config.tv_weight * tv_loss
        )
        total.backward()
        optimiser.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
        if iteration % config.checkpoint_interval == 0 or iteration == config.iterations - 1:
            values = (float(total), float(aps_loss), float(event_loss), float(tv_loss))
            history.append(values)
            print(
                f"[{mode}] {iteration + 1:4d}/{config.iterations}: "
                f"total={values[0]:.6g} aps={values[1]:.6g} "
                f"event={values[2]:.6g} tv={values[3]:.6g}",
                flush=True,
            )

    with torch.no_grad():
        reconstruction = expand_latent_image(latent, source_shape)
        signals = model(reconstruction, shifts)
        aps_prediction = trapezoidal_average(signals, times)
        event_prediction = predict_cumulative_event_change(signals, gain, white_dn)
    reprojection = np.column_stack((observations.core_aps, aps_prediction.cpu().numpy()))
    residual = (
        event_prediction.cpu().numpy()
        - observations.cumulative_event_change[indices]
    )
    return (
        reconstruction.cpu().numpy().astype(np.float32),
        np.asarray(history, dtype=np.float64),
        reprojection.astype(np.float32),
        residual.astype(np.float32),
    )


def run_fibre_reconstruction(
    config: FibreReconstructionConfig,
    modes: Iterable[str] = ("aps_only", "events_only", "joint"),
) -> dict:
    """Validate the model, reconstruct requested modes, and save all evidence."""
    project_root = Path(__file__).resolve().parents[1]
    config = config.resolve(project_root)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    device = _device(config.device)

    from fibre_sim.config import derived_parameters, load_config, output_root

    simulation_cfg = load_config(config.simulation_config)
    data_root = output_root(simulation_cfg)
    observations = load_fibre_observations(
        simulation_cfg, data_root, config.calibration_pixels_per_core
    )
    model = _make_forward(simulation_cfg, observations, device)
    parity = validate_forward_model(
        model, observations, simulation_cfg, data_root, device
    )
    print("[validation]", json.dumps(parity, indent=2), flush=True)
    if parity["core_signal_rmse"] > 2e-3:
        raise RuntimeError("differentiable forward model does not match the simulator")
    if parity["truth_event_residual_mae"] >= parity["random_event_residual_mae"]:
        raise RuntimeError("event observations do not discriminate truth from a random object")

    derived = derived_parameters(simulation_cfg)
    initial = _initial_image(
        observations,
        derived["source_shape_px"],
        float(simulation_cfg["source"]["pixel_size_um"]),
        float(simulation_cfg["grin"]["magnification"]),
    )
    crop = _observable_crop(
        observations,
        derived["source_shape_px"],
        float(simulation_cfg["source"]["pixel_size_um"]),
        float(simulation_cfg["grin"]["magnification"]),
        float(simulation_cfg["fibre"]["core_diameter_um"]) / 2,
    )
    truth = np.load(data_root / "00_source" / "object_intensity.npy").astype(np.float32)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "initial_aps_interpolation.npy", initial)
    np.save(output / "truth_for_evaluation_only.npy", truth)
    results: dict[str, np.ndarray] = {}
    summaries: dict[str, dict] = {}
    started = time.time()
    for mode in modes:
        reconstruction, history, reprojection, residual = _optimise(
            mode=mode,
            model=model,
            observations=observations,
            simulation_cfg=simulation_cfg,
            config=config,
            initial_image=initial,
            device=device,
        )
        metrics = image_metrics(reconstruction[crop], truth[crop])
        summary = {
            "mode": mode,
            "optimisation_inputs": (
                "events only"
                if mode == "events_only"
                else "APS only"
                if mode == "aps_only"
                else "APS and events"
            ),
            "metrics_on_observable_region": metrics,
            "aps_reprojection_rmse": float(
                np.sqrt(np.mean((reprojection[:, 1] - reprojection[:, 0]) ** 2))
            ),
            "event_residual_mae": float(np.mean(np.abs(residual))),
            "event_residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "iterations": config.iterations,
        }
        save_mode_result(
            output / mode,
            reconstruction=reconstruction,
            observable_reconstruction=reconstruction[crop],
            loss_history=history,
            aps_reprojection=reprojection,
            event_residual=residual,
            summary=summary,
        )
        results[mode] = reconstruction
        summaries[mode] = summary

    save_comparison(output, truth, initial, results, crop)
    improvements = {}
    quality_checks = {}
    if "aps_only" in summaries and "joint" in summaries:
        aps_metrics = summaries["aps_only"]["metrics_on_observable_region"]
        joint_metrics = summaries["joint"]["metrics_on_observable_region"]
        tracked_metrics = (
            "psnr_db",
            "ssim",
            "correlation",
            "gradient_x_correlation",
            "gradient_y_correlation",
        )
        improvements = {
            name: joint_metrics[name] - aps_metrics[name] for name in tracked_metrics
        }
        quality_checks = {
            "joint_psnr_exceeds_aps_only": improvements["psnr_db"] > 0,
            "joint_ssim_exceeds_aps_only": improvements["ssim"] > 0,
            "joint_x_gradient_correlation_exceeds_aps_only": (
                improvements["gradient_x_correlation"] > 0
            ),
            "joint_y_gradient_correlation_exceeds_aps_only": (
                improvements["gradient_y_correlation"] > 0
            ),
        }
    shift_steps = np.diff(observations.shifts_xy_um, axis=0)
    overall = {
        "method": "scheme_A_core_aggregated_events",
        "simulation_config": str(config.simulation_config),
        "data_root": str(data_root),
        "device": str(device),
        "latent_shape": list(config.latent_shape),
        "event_time_stride": config.event_time_stride,
        "event_weight": config.event_weight,
        "tv_weight": config.tv_weight,
        "grin_sigma_um": float(simulation_cfg["grin"]["sigma_um"]),
        "core_count": len(observations.core_centres_xy_um),
        "selected_pixels_per_core": config.calibration_pixels_per_core,
        "observable_crop_yx": [
            crop[0].start,
            crop[0].stop,
            crop[1].start,
            crop[1].stop,
        ],
        "trajectory_range_xy_um": np.ptp(
            observations.shifts_xy_um, axis=0
        ).tolist(),
        "trajectory_path_length_um": float(
            np.linalg.norm(shift_steps, axis=1).sum()
        ),
        "forward_validation": parity,
        "initial_metrics_on_observable_region": image_metrics(
            initial[crop], truth[crop]
        ),
        "reconstructions": summaries,
        "joint_improvement_over_aps_only": improvements,
        "quality_checks": quality_checks,
        "elapsed_seconds": time.time() - started,
        "truth_usage": "forward parity and evaluation only; never an optimisation input",
    }
    (output / "run_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    return overall
