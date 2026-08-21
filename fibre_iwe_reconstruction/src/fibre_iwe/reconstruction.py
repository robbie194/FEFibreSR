"""APS + core-IWE reconstruction without simulation optical parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import CoreObservations, interpolate_core_aps
from .motion import MotionEstimate
from .render import (
    bilinear_splat,
    gaussian_blur,
    render_iwe,
    sample_image_at_points,
    sample_trajectory,
    shift_image,
    warp_events,
)


@dataclass(frozen=True)
class ReconstructionResult:
    initial_aps: np.ndarray
    aps_only: np.ndarray
    joint: np.ndarray
    observed_iwe: np.ndarray
    predicted_iwe: np.ndarray
    observability: np.ndarray
    loss_history_aps: np.ndarray
    loss_history_joint: np.ndarray
    aps_observed_predicted: np.ndarray


def _scaled_geometry(
    observations: CoreObservations,
    motion: MotionEstimate,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_x = shape[1] / observations.sensor_shape[1]
    scale_y = shape[0] / observations.sensor_shape[0]
    scale = torch.tensor((scale_x, scale_y), device=device)
    centres = torch.as_tensor(observations.centres_xy, device=device) * scale
    event_xy = torch.as_tensor(observations.event_xy, device=device) * scale
    event_time = torch.as_tensor(observations.event_time_normalized, device=device)
    polarity = torch.as_tensor(observations.event_polarity, device=device)
    trajectory = torch.as_tensor(motion.control_positions_xy, device=device) * scale
    return centres, event_xy, event_time, polarity, trajectory


def _event_targets(
    observations: CoreObservations,
    motion: MotionEstimate,
    shape: tuple[int, int],
    sigma: float,
    time_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centres, event_xy, event_time, polarity, trajectory = _scaled_geometry(
        observations, motion, shape, device
    )
    observed = render_iwe(
        event_xy, event_time, polarity, trajectory, shape, sigma, signed=True
    )
    uniform_time = torch.linspace(0, 1, time_samples, device=device)
    repeated_centres = centres.repeat(time_samples, 1)
    repeated_time = uniform_time.repeat_interleave(len(centres))
    warped = warp_events(repeated_centres, repeated_time, trajectory)
    weights = torch.full(
        (len(warped),), 1.0 / time_samples, device=device, dtype=warped.dtype
    )
    observability = gaussian_blur(
        bilinear_splat(warped[:, 0], warped[:, 1], weights, shape), sigma
    )
    observability /= observability.max().clamp_min(1e-8)
    return observed, observability


def _motion_average(
    image: torch.Tensor, trajectory: torch.Tensor, sample_count: int
) -> torch.Tensor:
    times = torch.linspace(0, 1, sample_count, device=image.device)
    positions = sample_trajectory(trajectory, times)
    reference = sample_trajectory(
        trajectory, torch.tensor([0.5], device=image.device)
    )[0]
    frames = [shift_image(image, position - reference) for position in positions]
    return torch.stack(frames).mean(dim=0)


def _predicted_iwe(
    image: torch.Tensor,
    trajectory: torch.Tensor,
    observability: torch.Tensor,
) -> torch.Tensor:
    log_image = torch.log(image.clamp_min(1e-3))
    dx = torch.roll(log_image, -1, dims=1) - log_image
    dy = torch.roll(log_image, -1, dims=0) - log_image
    dx[:, -1] = 0
    dy[-1, :] = 0
    displacement = trajectory[-1] - trajectory[0]
    return observability * (-dx * displacement[0] - dy * displacement[1])


def _second_order_regularizer(image: torch.Tensor) -> torch.Tensor:
    dxx = image[:, 2:] - 2 * image[:, 1:-1] + image[:, :-2]
    dyy = image[2:, :] - 2 * image[1:-1, :] + image[:-2, :]
    return torch.sqrt(dxx.square() + 1e-6).mean() + torch.sqrt(dyy.square() + 1e-6).mean()


def _normalised_event_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    predicted = predicted - predicted.mean()
    observed = observed - observed.mean()
    predicted = predicted / torch.linalg.vector_norm(predicted).clamp_min(1e-8)
    observed = observed / torch.linalg.vector_norm(observed).clamp_min(1e-8)
    cosine = (predicted * observed).sum()
    return 1 - cosine


def _optimise_scale(
    initial: torch.Tensor,
    observations: CoreObservations,
    motion: MotionEstimate,
    shape: tuple[int, int],
    config: dict,
    use_events: bool,
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    image = F.interpolate(
        initial[None, None], size=shape, mode="bilinear", align_corners=True
    )[0, 0].detach().clone().requires_grad_(True)
    centres, _, _, _, trajectory = _scaled_geometry(observations, motion, shape, device)
    core_aps = torch.as_tensor(observations.core_aps, device=device)
    observed_iwe, observability = _event_targets(
        observations,
        motion,
        shape,
        float(config["iwe_sigma_px"]) * shape[0] / observations.sensor_shape[0],
        int(config["observability_time_samples"]),
        device,
    )
    optimizer = torch.optim.Adam([image], lr=float(config["learning_rate"]))
    history: list[tuple[int, float, float, float, float]] = []
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        blurred = _motion_average(
            image, trajectory, int(config["aps_time_samples"])
        )
        predicted_aps = sample_image_at_points(blurred, centres)
        aps_loss = F.smooth_l1_loss(
            predicted_aps, core_aps, beta=float(config["aps_huber_beta"])
        )
        predicted_iwe = _predicted_iwe(image, trajectory, observability)
        event_loss = _normalised_event_loss(predicted_iwe, observed_iwe)
        regularizer = _second_order_regularizer(image)
        total = aps_loss + float(config["regularization_weight"]) * regularizer
        if use_events:
            total = total + float(config["event_weight"]) * event_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_([image], 1.0)
        optimizer.step()
        with torch.no_grad():
            image.clamp_(0.01, 1.0)
        if iteration % 25 == 0 or iteration == iterations - 1:
            history.append(
                (
                    iteration,
                    float(total),
                    float(aps_loss),
                    float(event_loss),
                    float(regularizer),
                )
            )
    return image.detach(), np.asarray(history, dtype=np.float64)


def reconstruct(
    observations: CoreObservations,
    motion: MotionEstimate,
    config: dict,
    device: torch.device,
) -> ReconstructionResult:
    """Run APS-only and joint coarse-to-fine effective-image reconstruction."""
    initial_numpy = interpolate_core_aps(observations)
    initial = torch.as_tensor(initial_numpy, device=device)
    scales = [tuple(map(int, value)) for value in config["multiscale_shapes"]]
    iterations = list(map(int, config["iterations_per_scale"]))
    if len(scales) != len(iterations):
        raise ValueError("multiscale_shapes and iterations_per_scale must match")

    aps_image = initial
    aps_histories = []
    for shape, count in zip(scales, iterations, strict=True):
        aps_image, history = _optimise_scale(
            aps_image, observations, motion, shape, config, False, count, device
        )
        aps_histories.append(history)
    joint_image = aps_image
    joint_histories = []
    for shape, count in zip(scales, iterations, strict=True):
        joint_image, history = _optimise_scale(
            joint_image, observations, motion, shape, config, True, count, device
        )
        joint_histories.append(history)

    final_shape = scales[-1]
    centres, _, _, _, trajectory = _scaled_geometry(
        observations, motion, final_shape, device
    )
    observed_iwe, observability = _event_targets(
        observations,
        motion,
        final_shape,
        float(config["iwe_sigma_px"]),
        int(config["observability_time_samples"]),
        device,
    )
    predicted_iwe = _predicted_iwe(joint_image, trajectory, observability)
    blurred = _motion_average(joint_image, trajectory, int(config["aps_time_samples"]))
    predicted_aps = sample_image_at_points(blurred, centres)
    def concatenate_histories(histories: list[np.ndarray]) -> np.ndarray:
        adjusted = []
        offset = 0
        for history in histories:
            current = history.copy()
            current[:, 0] += offset
            offset = int(current[-1, 0]) + 1
            adjusted.append(current)
        return np.concatenate(adjusted)

    return ReconstructionResult(
        initial_numpy,
        aps_image.cpu().numpy().astype(np.float32),
        joint_image.cpu().numpy().astype(np.float32),
        observed_iwe.cpu().numpy().astype(np.float32),
        predicted_iwe.cpu().numpy().astype(np.float32),
        observability.cpu().numpy().astype(np.float32),
        concatenate_histories(aps_histories),
        concatenate_histories(joint_histories),
        np.column_stack((observations.core_aps, predicted_aps.cpu().numpy())).astype(
            np.float32
        ),
    )
