"""Shared differentiable APS forward model and image regularizer."""

from __future__ import annotations

import torch

from .render import sample_image_at_points, sample_trajectory, shift_image


def motion_average(
    image: torch.Tensor,
    trajectory_xy: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Average a reference-time image along an exposure trajectory."""
    times = torch.linspace(0, 1, sample_count, device=image.device)
    positions = sample_trajectory(trajectory_xy, times)
    reference = sample_trajectory(
        trajectory_xy, torch.tensor([0.5], device=image.device)
    )[0]
    frames = [shift_image(image, position - reference) for position in positions]
    return torch.stack(frames).mean(dim=0)


def predict_core_aps(
    image: torch.Tensor,
    centres_xy: torch.Tensor,
    trajectory_xy: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Predict one exposure-averaged scalar intensity per fibre core."""
    blurred = motion_average(image, trajectory_xy, sample_count)
    return sample_image_at_points(blurred, centres_xy)


def second_order_regularizer(image: torch.Tensor) -> torch.Tensor:
    """Penalize oscillation while retaining piecewise-smooth image structure."""
    dxx = image[:, 2:] - 2 * image[:, 1:-1] + image[:, :-2]
    dyy = image[2:, :] - 2 * image[1:-1, :] + image[:-2, :]
    horizontal = torch.sqrt(dxx.square() + 1e-6).mean()
    vertical = torch.sqrt(dyy.square() + 1e-6).mean()
    return horizontal + vertical
