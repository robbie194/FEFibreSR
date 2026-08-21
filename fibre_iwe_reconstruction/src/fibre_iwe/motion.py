"""Blind low-dimensional trajectory estimation from core-domain events."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import CoreObservations, interpolate_core_aps
from .event_forward import (
    build_event_forward_model,
    normalised_event_loss,
    predict_iwe_bins,
)
from .image_forward import predict_core_aps, second_order_regularizer
from .render import render_iwe
from .trajectory import cubic_bspline_basis, fit_cubic_bspline


@dataclass(frozen=True)
class MotionEstimate:
    initial_control_positions_xy: np.ndarray
    control_positions_xy: np.ndarray
    coarse_scores: np.ndarray
    endpoint_validation_scores: np.ndarray
    path_validation_scores: np.ndarray
    model_name: str = "low_dimensional"
    spline_control_positions_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    refinement_history: np.ndarray = field(
        default_factory=lambda: np.empty((0, 6), dtype=np.float32)
    )
    joint_refinement_history: np.ndarray = field(
        default_factory=lambda: np.empty((0, 8), dtype=np.float32)
    )
    candidate_event_improvement_fraction: float = 0.0


def _numpy_iwe_score(
    xy: np.ndarray,
    time: np.ndarray,
    polarity: np.ndarray,
    centres_xy: np.ndarray,
    endpoint_xy: np.ndarray,
    shape: tuple[int, int],
    sigma: float,
    time_samples: int = 36,
) -> float:
    """Density-normalised CMax score that removes the fixed core-lattice bias."""
    warped = xy - time[:, None] * endpoint_xy[None]
    image, _, _ = np.histogram2d(
        warped[:, 1],
        warped[:, 0],
        bins=shape,
        range=((0, shape[0]), (0, shape[1])),
        weights=polarity,
    )
    times = np.linspace(0, 1, time_samples)
    support_xy = (
        centres_xy[None] - times[:, None, None] * endpoint_xy[None, None]
    ).reshape(-1, 2)
    support, _, _ = np.histogram2d(
        support_xy[:, 1],
        support_xy[:, 0],
        bins=shape,
        range=((0, shape[0]), (0, shape[1])),
    )
    focused = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    density = cv2.GaussianBlur(support.astype(np.float32), (0, 0), sigma)
    density /= float(time_samples)
    rate = focused / (density + 0.02)
    selected = density > 0.02
    weights = density[selected]
    values = rate[selected]
    mean = float(np.sum(weights * values) / np.sum(weights))
    return float(np.sum(weights * np.square(values - mean)) / np.sum(weights))


def coarse_linear_endpoint(
    observations: CoreObservations,
    search_radius_px: int,
    search_step_px: float,
    sigma: float,
    displacement_regularization: float,
    max_events: int = 60_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-search a constant-velocity endpoint using only IWE contrast."""
    count = len(observations.event_xy)
    stride = max(1, int(np.ceil(count / max_events)))
    xy = observations.event_xy[::stride]
    time = observations.event_time_normalized[::stride]
    polarity = observations.event_polarity[::stride]
    candidates = np.arange(
        -search_radius_px, search_radius_px + 0.5 * search_step_px, search_step_px
    )
    records = []
    best_score = -np.inf
    best = np.zeros(2, dtype=np.float32)
    for dy in candidates:
        for dx in candidates:
            endpoint = np.array((dx, dy), dtype=np.float32)
            score = _numpy_iwe_score(
                xy,
                time,
                polarity,
                observations.centres_xy,
                endpoint,
                observations.sensor_shape,
                sigma,
            )
            regularized_score = score - float(displacement_regularization) * float(
                dx * dx + dy * dy
            )
            records.append((dx, dy, score, regularized_score))
            if regularized_score > best_score:
                best_score = regularized_score
                best = endpoint
    coarse_best = best.copy()
    for dy in np.arange(coarse_best[1] - 1, coarse_best[1] + 1.01, 0.25):
        for dx in np.arange(coarse_best[0] - 1, coarse_best[0] + 1.01, 0.25):
            endpoint = np.array((dx, dy), dtype=np.float32)
            score = _numpy_iwe_score(
                xy,
                time,
                polarity,
                observations.centres_xy,
                endpoint,
                observations.sensor_shape,
                sigma,
            )
            regularized_score = score - float(displacement_regularization) * float(
                dx * dx + dy * dy
            )
            records.append((dx, dy, score, regularized_score))
            if regularized_score > best_score:
                best_score = regularized_score
                best = endpoint
    return best, np.asarray(records, dtype=np.float32)


def endpoint_from_observation_consistency(
    observations: CoreObservations,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a linear endpoint using only APS-to-event forward consistency."""
    shape = tuple(map(int, config["endpoint_validation_shape"]))
    scale = torch.tensor(
        (
            shape[1] / observations.sensor_shape[1],
            shape[0] / observations.sensor_shape[0],
        ),
        device=device,
    )
    centres = torch.as_tensor(observations.centres_xy, device=device) * scale
    event_xy = torch.as_tensor(observations.event_xy, device=device) * scale
    event_time = torch.as_tensor(observations.event_time_normalized, device=device)
    polarity = torch.as_tensor(observations.event_polarity, device=device)
    aps = torch.as_tensor(interpolate_core_aps(observations), device=device)
    aps = F.interpolate(
        aps[None, None], size=shape, mode="bilinear", align_corners=True
    )[0, 0]
    time_samples = int(config["endpoint_validation_time_samples"])
    bin_count = int(config["endpoint_validation_time_bins"])
    sigma = float(config["endpoint_validation_iwe_sigma_px"]) * shape[0] / float(
        observations.sensor_shape[0]
    )
    regularization = float(config["endpoint_validation_displacement_weight"])

    records: list[tuple[float, float, float, float]] = []
    best_loss = np.inf
    best = np.zeros(2, dtype=np.float32)

    def evaluate(endpoint: np.ndarray) -> None:
        nonlocal best_loss, best
        trajectory = torch.stack(
            (torch.zeros(2, device=device), torch.as_tensor(endpoint, device=device))
        ) * scale
        model = build_event_forward_model(
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
        predicted = predict_iwe_bins(aps, model.flow_xy_bins)
        data_loss = float(normalised_event_loss(predicted, model.observed_iwe_bins))
        total_loss = data_loss + regularization * float(np.square(endpoint).sum())
        records.append((float(endpoint[0]), float(endpoint[1]), data_loss, total_loss))
        if total_loss < best_loss:
            best_loss = total_loss
            best = endpoint.copy()

    radius = int(config["coarse_search_radius_px"])
    step = float(config["coarse_search_step_px"])
    candidates = np.arange(-radius, radius + 0.5 * step, step)
    with torch.no_grad():
        for dy in candidates:
            for dx in candidates:
                evaluate(np.array((dx, dy), dtype=np.float32))
        coarse_best = best.copy()
        refinement_radius = float(config["endpoint_validation_refinement_radius_px"])
        refinement_step = float(config["endpoint_validation_refinement_step_px"])
        refined_x = np.arange(
            coarse_best[0] - refinement_radius,
            coarse_best[0] + refinement_radius + 0.5 * refinement_step,
            refinement_step,
        )
        refined_y = np.arange(
            coarse_best[1] - refinement_radius,
            coarse_best[1] + refinement_radius + 0.5 * refinement_step,
            refinement_step,
        )
        for dy in refined_y:
            for dx in refined_x:
                evaluate(np.array((dx, dy), dtype=np.float32))
    return best, np.asarray(records, dtype=np.float32)


def trajectory_from_observation_consistency(
    observations: CoreObservations,
    endpoint_xy: np.ndarray,
    segment_count: int,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smooth 2-D arc and speed profile with two scalar parameters."""
    shape = tuple(map(int, config["path_validation_shape"]))
    scale = torch.tensor(
        (
            shape[1] / observations.sensor_shape[1],
            shape[0] / observations.sensor_shape[0],
        ),
        device=device,
    )
    centres = torch.as_tensor(observations.centres_xy, device=device) * scale
    event_xy = torch.as_tensor(observations.event_xy, device=device) * scale
    event_time = torch.as_tensor(observations.event_time_normalized, device=device)
    polarity = torch.as_tensor(observations.event_polarity, device=device)
    aps = torch.as_tensor(interpolate_core_aps(observations), device=device)
    aps = F.interpolate(
        aps[None, None], size=shape, mode="bilinear", align_corners=True
    )[0, 0]
    endpoint_norm = max(float(np.linalg.norm(endpoint_xy)), 1e-8)
    tangent = endpoint_xy / endpoint_norm
    normal = np.array((-tangent[1], tangent[0]), dtype=np.float32)
    times = np.linspace(0, 1, segment_count + 1, dtype=np.float32)
    bend_basis = np.sin(np.pi * times) ** 2
    easing_basis = np.sin(2 * np.pi * times)
    time_samples = int(config["path_validation_time_samples"])
    bin_count = int(config["path_validation_time_bins"])
    sigma = float(config["path_validation_iwe_sigma_px"]) * shape[0] / float(
        observations.sensor_shape[0]
    )

    records: list[tuple[float, float, float]] = []
    best_loss = np.inf
    best_positions = np.linspace(
        (0, 0), endpoint_xy, segment_count + 1, dtype=np.float32
    )
    best_parameters = np.zeros(2, dtype=np.float32)

    def evaluate(curvature: float, easing: float) -> None:
        nonlocal best_loss, best_positions, best_parameters
        positions = (
            times[:, None] * endpoint_xy
            + curvature * bend_basis[:, None] * normal
            + easing * easing_basis[:, None] * tangent
        ).astype(np.float32)
        trajectory = torch.as_tensor(positions, device=device) * scale
        model = build_event_forward_model(
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
        predicted = predict_iwe_bins(aps, model.flow_xy_bins)
        loss = float(normalised_event_loss(predicted, model.observed_iwe_bins))
        records.append((curvature, easing, loss))
        if loss < best_loss:
            best_loss = loss
            best_positions = positions
            best_parameters[:] = (curvature, easing)

    curvature_min, curvature_max, curvature_step = map(
        float, config["path_curvature_search_px"]
    )
    easing_min, easing_max, easing_step = map(
        float, config["path_easing_search_px"]
    )
    with torch.no_grad():
        for curvature in np.arange(
            curvature_min, curvature_max + 0.5 * curvature_step, curvature_step
        ):
            for easing in np.arange(
                easing_min, easing_max + 0.5 * easing_step, easing_step
            ):
                evaluate(float(curvature), float(easing))
        refinement_step = float(config["path_refinement_step_px"])
        for curvature in np.arange(
            best_parameters[0] - curvature_step,
            best_parameters[0] + curvature_step + 0.5 * refinement_step,
            refinement_step,
        ):
            for easing in np.arange(
                best_parameters[1] - easing_step,
                best_parameters[1] + easing_step + 0.5 * refinement_step,
                refinement_step,
            ):
                evaluate(float(curvature), float(easing))
    return best_positions, np.asarray(records, dtype=np.float32)


def _estimate_low_dimensional_motion(
    observations: CoreObservations,
    config: dict,
    device: torch.device,
) -> MotionEstimate:
    """Estimate a low-dimensional path from CMax and APS/event consistency."""
    segment_count = int(config["segment_count"])
    cmax_endpoint, scores = coarse_linear_endpoint(
        observations,
        int(config["coarse_search_radius_px"]),
        float(config["coarse_search_step_px"]),
        float(config["iwe_sigma_px"]),
        float(config["coarse_displacement_regularization"]),
    )
    if bool(config.get("use_aps_event_endpoint_validation", True)):
        validated_endpoint, endpoint_scores = endpoint_from_observation_consistency(
            observations, config, device
        )
        agreement_radius = float(config["endpoint_cmax_agreement_radius_px"])
        if np.linalg.norm(validated_endpoint - cmax_endpoint) <= agreement_radius:
            endpoint = cmax_endpoint
        else:
            endpoint = validated_endpoint
    else:
        endpoint = cmax_endpoint
        endpoint_scores = np.empty((0, 4), dtype=np.float32)
    initial_positions = np.linspace(
        (0, 0), endpoint, segment_count + 1, dtype=np.float32
    )
    final_positions, path_scores = trajectory_from_observation_consistency(
        observations, endpoint, segment_count, config, device
    )
    return MotionEstimate(
        initial_positions,
        final_positions,
        scores,
        endpoint_scores,
        path_scores,
    )


def _estimate_bspline_motion(
    observations: CoreObservations,
    initial: MotionEstimate,
    config: dict,
    device: torch.device,
) -> MotionEstimate:
    """Refine the low-dimensional initializer with a bounded cubic B-spline."""
    shape = tuple(map(int, config["spline_validation_shape"]))
    scale = torch.tensor(
        (
            shape[1] / observations.sensor_shape[1],
            shape[0] / observations.sensor_shape[0],
        ),
        device=device,
    )
    centres = torch.as_tensor(observations.centres_xy, device=device) * scale
    event_xy = torch.as_tensor(observations.event_xy, device=device) * scale
    event_time = torch.as_tensor(observations.event_time_normalized, device=device)
    polarity = torch.as_tensor(observations.event_polarity, device=device)
    aps = torch.as_tensor(interpolate_core_aps(observations), device=device)
    aps = F.interpolate(
        aps[None, None], size=shape, mode="bilinear", align_corners=True
    )[0, 0]

    control_count = int(config["spline_control_count"])
    base_controls_numpy = fit_cubic_bspline(
        initial.control_positions_xy, control_count
    )
    base_controls = torch.as_tensor(base_controls_numpy, device=device)
    sample_count = int(config["segment_count"]) + 1
    basis = torch.as_tensor(
        cubic_bspline_basis(np.linspace(0, 1, sample_count), control_count),
        device=device,
    )
    raw_offsets = torch.nn.Parameter(
        torch.zeros((control_count - 1, 2), device=device)
    )
    optimizer = torch.optim.Adam(
        [raw_offsets], lr=float(config["spline_learning_rate"])
    )
    max_offset = float(config["spline_max_control_offset_px"])
    smoothness_weight = float(config["spline_smoothness_weight"])
    deviation_weight = float(config["spline_deviation_weight"])
    endpoint_weight = float(config["spline_endpoint_weight"])
    sigma = float(config["spline_iwe_sigma_px"]) * shape[0] / float(
        observations.sensor_shape[0]
    )
    time_samples = int(config["spline_time_samples"])
    bin_count = int(config["spline_time_bins"])
    history: list[tuple[int, float, float, float, float, float]] = []
    best_loss = np.inf
    initial_event_loss = np.nan
    best_event_loss = np.inf
    best_controls = base_controls_numpy.copy()
    best_positions = initial.control_positions_xy.copy()
    endpoint_reference = torch.as_tensor(
        initial.control_positions_xy[-1], device=device
    )

    for iteration in range(int(config["spline_iterations"])):
        optimizer.zero_grad(set_to_none=True)
        bounded_offsets = max_offset * torch.tanh(raw_offsets)
        controls = base_controls + torch.cat(
            (torch.zeros((1, 2), device=device), bounded_offsets), dim=0
        )
        positions = basis @ controls
        event_model = build_event_forward_model(
            centres,
            event_xy,
            event_time,
            polarity,
            positions * scale,
            shape,
            sigma,
            time_samples,
            bin_count,
        )
        predicted = predict_iwe_bins(aps, event_model.flow_xy_bins)
        event_loss = normalised_event_loss(predicted, event_model.observed_iwe_bins)
        if iteration == 0:
            initial_event_loss = float(event_loss.detach())
        acceleration = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        smoothness = acceleration.square().mean()
        deviation = bounded_offsets.square().mean()
        endpoint = (positions[-1] - endpoint_reference).square().mean()
        total = (
            event_loss
            + smoothness_weight * smoothness
            + deviation_weight * deviation
            + endpoint_weight * endpoint
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_([raw_offsets], 2.0)
        optimizer.step()
        value = float(total.detach())
        if value < best_loss:
            best_loss = value
            best_event_loss = float(event_loss.detach())
            best_controls = controls.detach().cpu().numpy().astype(np.float32)
            best_positions = positions.detach().cpu().numpy().astype(np.float32)
        if iteration % 20 == 0 or iteration == int(config["spline_iterations"]) - 1:
            history.append(
                (
                    iteration,
                    value,
                    float(event_loss.detach()),
                    float(smoothness.detach()),
                    float(deviation.detach()),
                    float(endpoint.detach()),
                )
            )

    improvement = (initial_event_loss - best_event_loss) / max(initial_event_loss, 1e-8)
    history_array = np.asarray(history, dtype=np.float32)
    if improvement < float(config["spline_min_event_improvement_fraction"]):
        return MotionEstimate(
            initial.initial_control_positions_xy,
            initial.control_positions_xy,
            initial.coarse_scores,
            initial.endpoint_validation_scores,
            initial.path_validation_scores,
            "low_dimensional_fallback",
            best_controls,
            history_array,
            candidate_event_improvement_fraction=improvement,
        )
    return MotionEstimate(
        initial.control_positions_xy,
        best_positions,
        initial.coarse_scores,
        initial.endpoint_validation_scores,
        initial.path_validation_scores,
        "bspline",
        best_controls,
        history_array,
        candidate_event_improvement_fraction=improvement,
    )


def jointly_refine_bspline_motion(
    observations: CoreObservations,
    motion: MotionEstimate,
    motion_config: dict,
    reconstruction_config: dict,
    device: torch.device,
) -> MotionEstimate:
    """Jointly refine a B-spline trajectory and effective image from observations."""
    if motion.model_name != "bspline":
        return motion
    shape = tuple(map(int, motion_config["spline_joint_shape"]))
    scale = torch.tensor(
        (
            shape[1] / observations.sensor_shape[1],
            shape[0] / observations.sensor_shape[0],
        ),
        device=device,
    )
    centres = torch.as_tensor(observations.centres_xy, device=device) * scale
    event_xy = torch.as_tensor(observations.event_xy, device=device) * scale
    event_time = torch.as_tensor(observations.event_time_normalized, device=device)
    polarity = torch.as_tensor(observations.event_polarity, device=device)
    core_aps = torch.as_tensor(observations.core_aps, device=device)
    initial_image = torch.as_tensor(interpolate_core_aps(observations), device=device)
    image = F.interpolate(
        initial_image[None, None], size=shape, mode="bilinear", align_corners=True
    )[0, 0].detach().clone().requires_grad_(True)

    base_controls = torch.as_tensor(
        motion.spline_control_positions_xy, device=device
    )
    control_count = len(base_controls)
    sample_count = len(motion.control_positions_xy)
    basis = torch.as_tensor(
        cubic_bspline_basis(np.linspace(0, 1, sample_count), control_count),
        device=device,
    )
    raw_offsets = torch.nn.Parameter(
        torch.zeros((control_count - 1, 2), device=device)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [image], "lr": float(motion_config["spline_joint_image_lr"])},
            {
                "params": [raw_offsets],
                "lr": float(motion_config["spline_joint_motion_lr"]),
            },
        ]
    )
    max_offset = float(motion_config["spline_joint_max_control_offset_px"])
    warmup_iterations = int(motion_config["spline_joint_warmup_iterations"])
    iterations = int(motion_config["spline_joint_iterations"])
    sigma = float(motion_config["spline_joint_iwe_sigma_px"]) * shape[0] / float(
        observations.sensor_shape[0]
    )
    history: list[tuple[int, float, float, float, float, float, float, float]] = []
    best_loss = np.inf
    best_controls = motion.spline_control_positions_xy.copy()
    best_positions = motion.control_positions_xy.copy()
    endpoint_reference = torch.as_tensor(
        motion.control_positions_xy[-1], device=device
    )

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        bounded_offsets = max_offset * torch.tanh(raw_offsets)
        active_offsets = (
            bounded_offsets.detach()
            if iteration < warmup_iterations
            else bounded_offsets
        )
        controls = base_controls + torch.cat(
            (torch.zeros((1, 2), device=device), active_offsets), dim=0
        )
        positions = basis @ controls
        scaled_trajectory = positions * scale
        event_model = build_event_forward_model(
            centres,
            event_xy,
            event_time,
            polarity,
            scaled_trajectory,
            shape,
            sigma,
            int(motion_config["spline_joint_time_samples"]),
            int(motion_config["spline_joint_time_bins"]),
        )
        predicted_iwe = predict_iwe_bins(image, event_model.flow_xy_bins)
        event_loss = normalised_event_loss(
            predicted_iwe, event_model.observed_iwe_bins
        )
        predicted_aps = predict_core_aps(
            image,
            centres,
            scaled_trajectory,
            int(reconstruction_config["aps_time_samples"]),
        )
        aps_loss = F.smooth_l1_loss(
            predicted_aps,
            core_aps,
            beta=float(reconstruction_config["aps_huber_beta"]),
        )
        image_regularizer = second_order_regularizer(image)
        acceleration = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        motion_smoothness = acceleration.square().mean()
        correction = active_offsets.square().mean()
        endpoint = (positions[-1] - endpoint_reference).square().mean()
        total = (
            aps_loss
            + float(motion_config["spline_joint_event_weight"]) * event_loss
            + float(motion_config["spline_joint_image_regularization_weight"])
            * image_regularizer
            + float(motion_config["spline_joint_smoothness_weight"])
            * motion_smoothness
            + float(motion_config["spline_joint_deviation_weight"]) * correction
            + float(motion_config["spline_joint_endpoint_weight"]) * endpoint
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_([image, raw_offsets], 2.0)
        optimizer.step()
        with torch.no_grad():
            image.clamp_(0.01, 1.0)
        value = float(total.detach())
        if iteration >= warmup_iterations and value < best_loss:
            best_loss = value
            best_controls = controls.detach().cpu().numpy().astype(np.float32)
            best_positions = positions.detach().cpu().numpy().astype(np.float32)
        if iteration % 20 == 0 or iteration == iterations - 1:
            history.append(
                (
                    iteration,
                    value,
                    float(aps_loss.detach()),
                    float(event_loss.detach()),
                    float(image_regularizer.detach()),
                    float(motion_smoothness.detach()),
                    float(correction.detach()),
                    float(endpoint.detach()),
                )
            )

    return MotionEstimate(
        motion.initial_control_positions_xy,
        best_positions,
        motion.coarse_scores,
        motion.endpoint_validation_scores,
        motion.path_validation_scores,
        motion.model_name,
        best_controls,
        motion.refinement_history,
        np.asarray(history, dtype=np.float32),
        motion.candidate_event_improvement_fraction,
    )


def estimate_motion(
    observations: CoreObservations,
    config: dict,
    device: torch.device,
) -> MotionEstimate:
    """Estimate either the compact path or its regularized B-spline extension."""
    initial = _estimate_low_dimensional_motion(observations, config, device)
    model = str(config.get("model", "low_dimensional"))
    if model == "low_dimensional":
        return initial
    if model == "bspline":
        return _estimate_bspline_motion(observations, initial, config, device)
    raise ValueError(f"unknown motion estimation model: {model}")
