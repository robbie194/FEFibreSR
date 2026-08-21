"""Shared temporal core-IWE forward model for motion selection and inversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .render import bilinear_splat, gaussian_blur, render_iwe, sample_trajectory


@dataclass(frozen=True)
class EventForwardModel:
    observed_iwe_bins: torch.Tensor
    flow_xy_bins: torch.Tensor
    observability: torch.Tensor


def build_event_forward_model(
    centres_xy: torch.Tensor,
    event_xy: torch.Tensor,
    event_time: torch.Tensor,
    polarity: torch.Tensor,
    trajectory_xy: torch.Tensor,
    shape: tuple[int, int],
    sigma: float,
    time_samples: int,
    bin_count: int,
) -> EventForwardModel:
    """Build temporal IWE targets and trajectory-dependent 2-D flow maps."""
    event_bins = torch.clamp((event_time * bin_count).long(), max=bin_count - 1)
    observed_bins = []
    for bin_index in range(bin_count):
        selected = event_bins == bin_index
        observed_bins.append(
            render_iwe(
                event_xy[selected],
                event_time[selected],
                polarity[selected],
                trajectory_xy,
                shape,
                sigma,
                signed=True,
            )
        )

    device = centres_xy.device
    time_edges = torch.linspace(0, 1, time_samples + 1, device=device)
    time_midpoints = 0.5 * (time_edges[:-1] + time_edges[1:])
    edge_positions = sample_trajectory(trajectory_xy, time_edges)
    midpoint_positions = sample_trajectory(trajectory_xy, time_midpoints)
    increments = edge_positions[1:] - edge_positions[:-1]
    reference = sample_trajectory(
        trajectory_xy, torch.tensor([0.5], device=device)
    )[0]
    interval_bins = torch.clamp(
        (time_midpoints * bin_count).long(), max=bin_count - 1
    )
    flow_bins = []
    density_bins = []
    for bin_index in range(bin_count):
        selected = interval_bins == bin_index
        positions = midpoint_positions[selected]
        warped = (
            centres_xy[None, :, :] - positions[:, None, :] + reference
        ).reshape(-1, 2)
        repeated_increments = increments[selected].repeat_interleave(
            len(centres_xy), dim=0
        )
        flow_components = []
        for axis in range(2):
            flow_components.append(
                gaussian_blur(
                    bilinear_splat(
                        warped[:, 0],
                        warped[:, 1],
                        repeated_increments[:, axis],
                        shape,
                    ),
                    sigma,
                )
            )
        density = gaussian_blur(
            bilinear_splat(
                warped[:, 0],
                warped[:, 1],
                torch.full(
                    (len(warped),),
                    1.0 / time_samples,
                    device=device,
                    dtype=warped.dtype,
                ),
                shape,
            ),
            sigma,
        )
        flow_bins.append(torch.stack(flow_components))
        density_bins.append(density)
    observability = torch.stack(density_bins).sum(0)
    observability /= observability.max().clamp_min(1e-8)
    return EventForwardModel(
        torch.stack(observed_bins), torch.stack(flow_bins), observability
    )


def predict_iwe_bins(
    image: torch.Tensor, flow_xy_bins: torch.Tensor
) -> torch.Tensor:
    """Predict each temporal IWE from the candidate log-image gradient."""
    log_image = torch.log(image.clamp_min(1e-3))
    dx = 0.5 * (
        torch.roll(log_image, -1, dims=1) - torch.roll(log_image, 1, dims=1)
    )
    dy = 0.5 * (
        torch.roll(log_image, -1, dims=0) - torch.roll(log_image, 1, dims=0)
    )
    dx[:, 0] = log_image[:, 1] - log_image[:, 0]
    dx[:, -1] = log_image[:, -1] - log_image[:, -2]
    dy[0, :] = log_image[1, :] - log_image[0, :]
    dy[-1, :] = log_image[-1, :] - log_image[-2, :]
    return -(flow_xy_bins[:, 0] * dx + flow_xy_bins[:, 1] * dy)


def normalised_event_loss(
    predicted: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    """Return an energy-weighted temporal cosine loss with unknown event scale."""
    predicted = predicted.flatten(1) - predicted.flatten(1).mean(1, keepdim=True)
    observed = observed.flatten(1) - observed.flatten(1).mean(1, keepdim=True)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=1).clamp_min(1e-8)
    observed_norm = torch.linalg.vector_norm(observed, dim=1)
    cosine = (predicted * observed).sum(1) / (
        predicted_norm * observed_norm.clamp_min(1e-8)
    )
    active = observed_norm > 1e-8
    weights = observed_norm[active]
    return ((1 - cosine[active]) * weights).sum() / weights.sum().clamp_min(1e-8)
