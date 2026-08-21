"""Blind low-dimensional trajectory estimation from core-domain events."""

from __future__ import annotations

from dataclasses import dataclass

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
from .render import render_iwe


@dataclass(frozen=True)
class MotionEstimate:
    initial_control_positions_xy: np.ndarray
    control_positions_xy: np.ndarray
    coarse_scores: np.ndarray
    endpoint_validation_scores: np.ndarray
    path_validation_scores: np.ndarray


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


def estimate_motion(
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
