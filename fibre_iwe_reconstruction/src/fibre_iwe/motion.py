"""Blind low-dimensional trajectory estimation from core-domain events."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .data import CoreObservations
from .render import render_iwe


@dataclass(frozen=True)
class MotionEstimate:
    initial_control_positions_xy: np.ndarray
    control_positions_xy: np.ndarray
    loss_history: np.ndarray
    coarse_scores: np.ndarray


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


def estimate_motion(
    observations: CoreObservations,
    config: dict,
    device: torch.device,
) -> MotionEstimate:
    """Estimate a smooth piecewise-linear path by core-IWE contrast maximization."""
    segment_count = int(config["segment_count"])
    endpoint, scores = coarse_linear_endpoint(
        observations,
        int(config["coarse_search_radius_px"]),
        float(config["coarse_search_step_px"]),
        float(config["iwe_sigma_px"]),
        float(config["coarse_displacement_regularization"]),
    )
    initial_positions = np.linspace((0, 0), endpoint, segment_count + 1).astype(np.float32)
    base_positions = torch.as_tensor(initial_positions, device=device)
    interior_offsets = torch.nn.Parameter(torch.zeros((segment_count - 1, 2), device=device))
    optimizer = torch.optim.Adam([interior_offsets], lr=float(config["learning_rate"]))
    xy = torch.as_tensor(observations.event_xy, device=device)
    time = torch.as_tensor(observations.event_time_normalized, device=device)
    polarity = torch.as_tensor(observations.event_polarity, device=device)
    centres = torch.as_tensor(observations.centres_xy, device=device)
    max_events = int(config["max_optimization_events"])
    if len(xy) > max_events:
        selected = torch.linspace(0, len(xy) - 1, max_events, device=device).long()
        xy, time, polarity = xy[selected], time[selected], polarity[selected]
    history: list[tuple[int, float, float, float]] = []
    iterations = int(config["iterations"])
    smooth_weight = float(config["smoothness_weight"])
    support_times = torch.linspace(0, 1, 36, device=device)
    support_time = support_times.repeat_interleave(len(centres))
    support_centres = centres.repeat(len(support_times), 1)
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        positions = base_positions + torch.cat(
            (
                torch.zeros((1, 2), device=device),
                interior_offsets,
                torch.zeros((1, 2), device=device),
            ),
            dim=0,
        )
        increments = positions[1:] - positions[:-1]
        iwe = render_iwe(
            xy,
            time,
            polarity,
            positions,
            observations.sensor_shape,
            float(config["iwe_sigma_px"]),
            signed=True,
        )
        density = render_iwe(
            support_centres,
            support_time,
            torch.ones(len(support_centres), device=device) / len(support_times),
            positions,
            observations.sensor_shape,
            float(config["iwe_sigma_px"]),
            signed=True,
        )
        rate = iwe / (density + 0.02)
        weight_sum = density.sum().clamp_min(1e-8)
        mean = (density * rate).sum() / weight_sum
        contrast = (density * (rate - mean).square()).sum() / weight_sum
        smoothness = (increments[1:] - increments[:-1]).square().mean()
        loss = -contrast + smooth_weight * smoothness
        loss.backward()
        torch.nn.utils.clip_grad_norm_([interior_offsets], 5.0)
        optimizer.step()
        with torch.no_grad():
            interior_offsets.clamp_(-2.0, 2.0)
        if iteration % 25 == 0 or iteration == iterations - 1:
            history.append(
                (iteration, float(loss), float(contrast), float(smoothness))
            )
    final_positions = base_positions + torch.cat(
        (
            torch.zeros((1, 2), device=device),
            interior_offsets.detach(),
            torch.zeros((1, 2), device=device),
        ),
        dim=0,
    )
    return MotionEstimate(
        initial_positions,
        final_positions.cpu().numpy().astype(np.float32),
        np.asarray(history, dtype=np.float64),
        scores,
    )
