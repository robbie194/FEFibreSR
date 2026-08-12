"""End-to-end NeuroSR experiment, expressed as five explicit stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import gaussian_blur

from .config import ExperimentConfig
from .data import ExposureSample, load_aedat4, select_exposure_sample
from .events import (
    event_image,
    gaussian_event_image,
    numpy_event_frame,
    warp_events_to_reference,
)
from .motion import (
    PiecewiseLinearTrajectory,
    dense_motion_blur_kernel,
    initialize_segment_increments,
    piecewise_motion_blur_kernel,
)
from .optimization import (
    AdamP,
    block_average,
    blur_image,
    directional_total_variation,
    linear_log_intensity,
    mean_square,
    predicted_iwe,
    smooth_l1_to_zero,
)
from .output import save_results


@dataclass
class EventTensors:
    x: torch.Tensor
    y: torch.Tensor
    timestamps_us: torch.Tensor
    polarity: torch.Tensor


@dataclass
class MotionEstimate:
    segment_increments_xy: torch.Tensor
    dense_trajectory_xy: torch.Tensor
    loss_history: np.ndarray


@dataclass
class ReconstructionState:
    image: torch.Tensor
    background: torch.Tensor
    kernel: torch.Tensor
    target_iwe: torch.Tensor
    predicted_iwe: torch.Tensor
    loss_history: np.ndarray


def _to_event_tensors(sample: ExposureSample, device: torch.device) -> EventTensors:
    """Move one event window to the compute device and reset its first timestamp."""
    timestamps = torch.from_numpy(sample.timestamps_us).float()
    timestamps = (timestamps - timestamps[0]).to(device)
    return EventTensors(
        x=torch.from_numpy(sample.x).float().to(device),
        y=torch.from_numpy(sample.y).float().to(device),
        timestamps_us=timestamps,
        polarity=torch.from_numpy(sample.polarity).float().to(device),
    )


def _build_registration_frames(
    sample: ExposureSample,
    sensor_shape: tuple[int, int],
    trajectory: PiecewiseLinearTrajectory,
    device: torch.device,
) -> torch.Tensor:
    """Create one event image around every trajectory-segment boundary."""
    width = float(trajectory.segment_widths[0].cpu())
    boundaries = trajectory.boundaries.cpu().numpy()
    frames = [
        numpy_event_frame(
            sample.x,
            sample.y,
            sample.timestamps_us,
            sample.polarity,
            sensor_shape,
            start_us=float(boundary - width // 2),
            duration_us=width,
        )
        for boundary in boundaries
    ]
    return torch.from_numpy(np.asarray(frames)).to(device)


def _warp_and_render(
    events: EventTensors,
    dense_trajectory_xy: torch.Tensor,
    reference_time_us: torch.Tensor | float,
    sensor_shape: tuple[int, int],
    *,
    signed: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    warped_x, warped_y = warp_events_to_reference(
        events.x,
        events.y,
        events.timestamps_us,
        dense_trajectory_xy,
        reference_time_us,
    )
    image = event_image(
        warped_x,
        warped_y,
        events.polarity,
        sensor_shape,
        signed=signed,
    )
    return image, warped_x, warped_y


def estimate_motion(
    sample: ExposureSample,
    events: EventTensors,
    config: ExperimentConfig,
    device: torch.device,
) -> MotionEstimate:
    """Initialize and optimize a continuous trajectory by event contrast."""
    model = PiecewiseLinearTrajectory(
        sample.event_window_us, config.trajectory_segments, device
    )
    registration_frames = _build_registration_frames(
        sample, config.sensor_shape, model, device
    )
    initial_increments = initialize_segment_increments(registration_frames).to(device)
    increments = nn.Parameter(initial_increments.clone())
    optimizer = Adam([increments], lr=1e-2)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.9)
    checkpoints: list[float] = []

    for iteration in range(config.motion_iterations):
        optimizer.zero_grad()
        dense_trajectory = model(increments).squeeze().T
        unsigned_iwe, _, _ = _warp_and_render(
            events,
            dense_trajectory,
            reference_time_us=events.timestamps_us[0],
            sensor_shape=config.sensor_shape,
            signed=False,
        )
        focused_iwe = gaussian_blur(unsigned_iwe[None, None], 3, sigma=1).squeeze()
        loss = -focused_iwe.abs().var() + model.regularization(0.2, 1e-4)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % 100 == 0:
            value = float(loss.detach().cpu())
            checkpoints.append(value)
            print(f"[motion] iteration={iteration:4d} loss={value:.12f}")

    # The reference script consumes the trajectory evaluated immediately before
    # the final optimizer step. Preserving that ordering is required for exact
    # numerical comparison with its saved baseline.
    return MotionEstimate(
        segment_increments_xy=increments.detach(),
        dense_trajectory_xy=dense_trajectory.detach(),
        loss_history=np.asarray(checkpoints),
    )


def _render_gaussian_iwe(
    events: EventTensors,
    dense_trajectory_xy: torch.Tensor,
    reference_time_us: int,
    sensor_shape: tuple[int, int],
    scale: int,
    sigma: float,
) -> torch.Tensor:
    _, warped_x, warped_y = _warp_and_render(
        events,
        dense_trajectory_xy,
        reference_time_us,
        sensor_shape,
        signed=True,
    )
    return gaussian_event_image(
        warped_x,
        warped_y,
        events.polarity,
        sensor_shape,
        scale,
        sigma,
        signed=True,
    )


def refine_at_sensor_scale(
    sample: ExposureSample,
    events: EventTensors,
    motion: MotionEstimate,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly refine the latent image, background, and motion at 1x scale."""
    scale = 1
    reference_time = sample.frame_exposure_us // 2
    model = PiecewiseLinearTrajectory(
        sample.event_window_us, config.trajectory_segments, device
    )
    trajectory = nn.Parameter(motion.segment_increments_xy.clone())
    image = nn.Parameter(
        torch.from_numpy(sample.blurred_frame).float().to(device) / 255
    )
    background = nn.Parameter(torch.zeros(config.sensor_shape, device=device))
    threshold = torch.tensor(1.000001, device=device)
    optimizer = Adam(
        [
            {"params": image, "lr": 4e-3},
            {"params": background, "lr": 4e-3},
            {"params": trajectory, "lr": 1e-4},
        ]
    )
    scheduler = StepLR(optimizer, step_size=100, gamma=0.7)
    base_displacement = motion.segment_increments_xy.sum(dim=0).detach()
    direction = base_displacement.cpu().numpy()
    direction = direction / np.max(np.abs(direction))

    for iteration in range(config.reconstruction_iterations):
        optimizer.zero_grad()
        log_image = linear_log_intensity(image * 255, threshold) / np.log(255)
        dense = model(trajectory).squeeze().T
        if iteration == 0:
            event_loss = torch.zeros((), device=device)
            contrast_loss = torch.zeros((), device=device)
            displacement = base_displacement
        else:
            displacement = trajectory.sum(dim=0)
            unsigned_iwe, warped_x, warped_y = _warp_and_render(
                events, dense, reference_time, config.sensor_shape, signed=True
            )
            contrast_loss = -unsigned_iwe.abs().var()
            target_iwe = gaussian_event_image(
                warped_x,
                warped_y,
                events.polarity,
                config.sensor_shape,
                scale,
                config.event_splat_sigma,
                signed=True,
            )
            predicted = predicted_iwe(log_image, displacement)
            event_loss = mean_square(
                predicted / torch.linalg.vector_norm(predicted)
                - target_iwe / torch.linalg.vector_norm(target_iwe)
            )

        kernel = piecewise_motion_blur_kernel(
            trajectory,
            scale,
            sample.frame_exposure_us,
            reference_time,
            sample.event_window_us,
        )
        blurred = blur_image(image, kernel) + background
        observed = torch.from_numpy(sample.blurred_frame).float().to(device) / 255
        frame_loss = mean_square(
            torch.sqrt(blurred.clamp_min(0) + 1e-8) - torch.sqrt(observed)
        )
        loss = (
            2e3 * event_loss
            + frame_loss
            + 4e-2 * directional_total_variation(image, direction)
            + 2e-1 * smooth_l1_to_zero(background, beta=1e-6)
            + contrast_loss
            + model.regularization(2e-2, 1e-4)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            image.clamp_(0, 1)
            finite_state = {
                "image": bool(torch.isfinite(image).all()),
                "background": bool(torch.isfinite(background).all()),
                "trajectory": bool(torch.isfinite(trajectory).all()),
            }
            if not all(finite_state.values()):
                raise FloatingPointError(
                    f"joint 1x optimization became non-finite at iteration "
                    f"{iteration}: {finite_state}"
                )
        if iteration == 1 or (iteration > 0 and iteration % 100 == 0):
            print(f"[joint-1x] iteration={iteration:4d} loss={float(loss):.12f}")
    return image.detach(), background.detach()


def reconstruct_at_two_x(
    sample: ExposureSample,
    events: EventTensors,
    motion: MotionEstimate,
    initial_image: torch.Tensor,
    initial_background: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> ReconstructionState:
    """Hold motion fixed and reconstruct the final 2x latent image."""
    scale = config.super_resolution_scale
    reference_time = sample.frame_exposure_us // 2
    target_iwe = _render_gaussian_iwe(
        events,
        motion.dense_trajectory_xy,
        reference_time,
        config.sensor_shape,
        scale,
        config.event_splat_sigma,
    ).detach()
    valid_border = torch.zeros_like(target_iwe)
    valid_border[1:-1, 1:-1] = 1
    target_iwe = target_iwe * valid_border

    image = nn.Parameter(
        F.interpolate(
            initial_image.detach()[None, None],
            size=(config.sensor_height * scale, config.sensor_width * scale),
        ).squeeze()
    )
    background = nn.Parameter(initial_background.clone())
    displacement = motion.segment_increments_xy.sum(dim=0).detach()
    direction = displacement.cpu().numpy()
    direction = direction / np.max(np.abs(direction))
    kernel = dense_motion_blur_kernel(
        motion.dense_trajectory_xy[:, : sample.frame_exposure_us],
        scale,
        reference_time,
    ).detach()
    optimizer = AdamP(
        [
            {"params": image, "lr": 4e-3},
            {"params": background, "lr": 1e-4},
        ]
    )
    scheduler = StepLR(optimizer, step_size=100, gamma=0.7)
    observed = torch.from_numpy(sample.blurred_frame).float().to(device) / 255
    checkpoints: list[float] = []

    for iteration in range(config.reconstruction_iterations):
        optimizer.zero_grad()
        log_image = linear_log_intensity(image * 255, 1.000001) / np.log(255)
        image_iwe = predicted_iwe(log_image, displacement) * valid_border
        if iteration == 0:
            event_loss = torch.zeros((), device=device)
        else:
            event_loss = mean_square(
                image_iwe / torch.linalg.vector_norm(image_iwe)
                - target_iwe / torch.linalg.vector_norm(target_iwe)
            )
        blurred = block_average(blur_image(image, kernel), scale) + background
        frame_loss = mean_square(
            torch.sqrt(blurred.clamp_min(0) + 1e-8) - torch.sqrt(observed)
        )
        loss = (
            2e3 * event_loss
            + frame_loss
            + 4e-2 * directional_total_variation(image, direction)
            + 2e-1 * smooth_l1_to_zero(background, beta=1e-6)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            image.clamp_(0, 1)
        if iteration % 100 == 0:
            value = float(loss.detach().cpu())
            checkpoints.append(value)
            print(f"[reconstruct-2x] iteration={iteration:4d} loss={value:.12f}")

    final_log = linear_log_intensity(image * 255, 1.000001) / np.log(255)
    final_iwe = predicted_iwe(final_log, displacement) * valid_border
    return ReconstructionState(
        image=image.detach(),
        background=background.detach(),
        kernel=kernel,
        target_iwe=target_iwe,
        predicted_iwe=final_iwe.detach(),
        loss_history=np.asarray(checkpoints),
    )


def run_experiment(config: ExperimentConfig) -> dict[str, np.ndarray]:
    """Execute all stages and write a self-contained reconstruction result."""
    torch.manual_seed(config.random_seed)
    torch.set_num_threads(config.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} input={config.input_path}")

    recording = load_aedat4(config.input_path)
    sample = select_exposure_sample(
        recording, config.requested_start_us, config.use_two_exposures
    )
    events = _to_event_tensors(sample, device)
    print(
        f"[data] frame={sample.frame_index} events={len(sample.timestamps_us)} "
        f"exposure={sample.frame_exposure_us}us window={sample.event_window_us}us"
    )

    motion = estimate_motion(sample, events, config, device)
    coarse_image, coarse_background = refine_at_sensor_scale(
        sample, events, motion, config, device
    )
    reconstruction = reconstruct_at_two_x(
        sample,
        events,
        motion,
        coarse_image,
        coarse_background,
        config,
        device,
    )
    arrays = {
        "frame_sharp": sample.sharp_frame,
        "frame_blurred": sample.blurred_frame,
        "reconstruction": reconstruction.image.cpu().numpy(),
        "event_iwe_target": reconstruction.target_iwe.cpu().numpy(),
        "event_iwe_predicted": reconstruction.predicted_iwe.cpu().numpy(),
        "background": reconstruction.background.cpu().numpy(),
        "motion_blur_kernel": reconstruction.kernel.cpu().numpy(),
        "trajectory_segments_xy": motion.segment_increments_xy.cpu().numpy(),
        "trajectory_dense_xy": motion.dense_trajectory_xy.cpu().numpy(),
        "loss_history": reconstruction.loss_history,
        "motion_loss_history": motion.loss_history,
    }
    summary = {
        "input_path": str(config.input_path),
        "device": str(device),
        "frame_index": sample.frame_index,
        "event_count": len(sample.timestamps_us),
        "event_window_us": sample.event_window_us,
        "frame_exposure_us": sample.frame_exposure_us,
        "reconstruction_scale": config.super_resolution_scale,
        "reconstruction_shape": list(arrays["reconstruction"].shape),
        "trajectory_piece_count": config.trajectory_segments,
        "trajectory_total_displacement_xy_px": arrays[
            "trajectory_segments_xy"
        ].sum(axis=0).tolist(),
        "final_loss": float(reconstruction.loss_history[-1]),
        "contrast_threshold": 1.000001,
    }
    save_results(config.output_dir, arrays, summary)
    print(f"[done] results saved to {config.output_dir}")
    return arrays
