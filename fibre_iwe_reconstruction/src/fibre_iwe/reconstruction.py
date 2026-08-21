"""APS + core-IWE reconstruction without simulation optical parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .data import CoreObservations, interpolate_core_aps
from .event_forward import (
    EventForwardModel,
    build_event_forward_model,
    normalised_event_loss,
    predict_iwe_bins,
)
from .image_forward import predict_core_aps, second_order_regularizer
from .motion import MotionEstimate


@dataclass(frozen=True)
class ReconstructionResult:
    initial_aps: np.ndarray
    event_only: np.ndarray
    aps_only: np.ndarray
    joint: np.ndarray
    observed_iwe: np.ndarray
    predicted_iwe: np.ndarray
    observability: np.ndarray
    observed_iwe_bins: np.ndarray
    predicted_iwe_bins: np.ndarray
    event_flow_xy_bins: np.ndarray
    loss_history_event_only: np.ndarray
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


def _event_forward_model(
    observations: CoreObservations,
    motion: MotionEstimate,
    shape: tuple[int, int],
    sigma: float,
    time_samples: int,
    bin_count: int,
    device: torch.device,
) -> EventForwardModel:
    centres, event_xy, event_time, polarity, trajectory = _scaled_geometry(
        observations, motion, shape, device
    )
    return build_event_forward_model(
        centres,
        event_xy,
        event_time,
        polarity,
        trajectory,
        shape,
        sigma,
        time_samples,
        bin_count,
    )


def _optimise_scale(
    initial: torch.Tensor,
    observations: CoreObservations,
    motion: MotionEstimate,
    shape: tuple[int, int],
    config: dict,
    mode: Literal["event", "aps", "joint"],
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    image = F.interpolate(
        initial[None, None], size=shape, mode="bilinear", align_corners=True
    )[0, 0].detach().clone().requires_grad_(True)
    centres, _, _, _, trajectory = _scaled_geometry(observations, motion, shape, device)
    core_aps = torch.as_tensor(observations.core_aps, device=device)
    event_model = None
    if mode != "aps":
        event_model = _event_forward_model(
            observations,
            motion,
            shape,
            float(config["iwe_sigma_px"]) * shape[0] / observations.sensor_shape[0],
            int(config["observability_time_samples"]),
            int(config["event_time_bins"]),
            device,
        )
    optimizer = torch.optim.Adam([image], lr=float(config["learning_rate"]))
    history: list[tuple[int, float, float, float, float]] = []
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        aps_loss = torch.zeros((), device=device)
        if mode != "event":
            predicted_aps = predict_core_aps(
                image, centres, trajectory, int(config["aps_time_samples"])
            )
            aps_loss = F.smooth_l1_loss(
                predicted_aps, core_aps, beta=float(config["aps_huber_beta"])
            )
        event_loss = torch.zeros((), device=device)
        if event_model is not None:
            predicted_iwe = predict_iwe_bins(image, event_model.flow_xy_bins)
            event_loss = normalised_event_loss(
                predicted_iwe, event_model.observed_iwe_bins
            )
        regularizer = second_order_regularizer(image)
        if mode == "event":
            gauge = (image.mean() - float(config["event_only_mean"])) ** 2
            gauge = gauge + (
                image.std() - float(config["event_only_std"])
            ) ** 2
            total = (
                event_loss
                + float(config["event_only_regularization_weight"]) * regularizer
                + float(config["event_only_gauge_weight"]) * gauge
            )
        else:
            total = aps_loss + float(config["regularization_weight"]) * regularizer
        if mode == "joint":
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
    """Run event-only, APS-only, and joint coarse-to-fine reconstruction."""
    initial_numpy = interpolate_core_aps(observations)
    initial = torch.as_tensor(initial_numpy, device=device)
    scales = [tuple(map(int, value)) for value in config["multiscale_shapes"]]
    iterations = list(map(int, config["iterations_per_scale"]))
    if len(scales) != len(iterations):
        raise ValueError("multiscale_shapes and iterations_per_scale must match")

    event_iterations = list(map(int, config["event_only_iterations_per_scale"]))
    if len(scales) != len(event_iterations):
        raise ValueError(
            "multiscale_shapes and event_only_iterations_per_scale must match"
        )
    event_image = torch.full_like(initial, float(config["event_only_mean"]))
    event_histories = []
    for shape, count in zip(scales, event_iterations, strict=True):
        event_image, history = _optimise_scale(
            event_image, observations, motion, shape, config, "event", count, device
        )
        event_histories.append(history)

    aps_image = initial
    aps_histories = []
    for shape, count in zip(scales, iterations, strict=True):
        aps_image, history = _optimise_scale(
            aps_image, observations, motion, shape, config, "aps", count, device
        )
        aps_histories.append(history)
    joint_image = aps_image
    joint_histories = []
    for shape, count in zip(scales, iterations, strict=True):
        joint_image, history = _optimise_scale(
            joint_image, observations, motion, shape, config, "joint", count, device
        )
        joint_histories.append(history)

    final_shape = scales[-1]
    centres, _, _, _, trajectory = _scaled_geometry(
        observations, motion, final_shape, device
    )
    event_model = _event_forward_model(
        observations,
        motion,
        final_shape,
        float(config["iwe_sigma_px"]),
        int(config["observability_time_samples"]),
        int(config["event_time_bins"]),
        device,
    )
    predicted_iwe_bins = predict_iwe_bins(joint_image, event_model.flow_xy_bins)
    observed_iwe = event_model.observed_iwe_bins.sum(0)
    predicted_iwe = predicted_iwe_bins.sum(0)
    predicted_aps = predict_core_aps(
        joint_image, centres, trajectory, int(config["aps_time_samples"])
    )

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
        event_image.cpu().numpy().astype(np.float32),
        aps_image.cpu().numpy().astype(np.float32),
        joint_image.cpu().numpy().astype(np.float32),
        observed_iwe.cpu().numpy().astype(np.float32),
        predicted_iwe.cpu().numpy().astype(np.float32),
        event_model.observability.cpu().numpy().astype(np.float32),
        event_model.observed_iwe_bins.cpu().numpy().astype(np.float32),
        predicted_iwe_bins.cpu().numpy().astype(np.float32),
        event_model.flow_xy_bins.cpu().numpy().astype(np.float32),
        concatenate_histories(event_histories),
        concatenate_histories(aps_histories),
        concatenate_histories(joint_histories),
        np.column_stack((observations.core_aps, predicted_aps.cpu().numpy())).astype(
            np.float32
        ),
    )
