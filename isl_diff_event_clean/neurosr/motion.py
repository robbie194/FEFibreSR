"""Piecewise-linear camera motion and blur-kernel construction."""

from __future__ import annotations

import numpy as np
import torch
from skimage.registration import phase_cross_correlation
from torch import nn

from .events import bilinear_splat, gaussian_splat


class PiecewiseLinearTrajectory(nn.Module):
    """Expand segment displacement increments into a per-microsecond path."""

    def __init__(self, duration_us: int, segment_count: int, device: torch.device):
        super().__init__()
        self.duration_us = duration_us
        self.segment_count = segment_count
        self.segment_widths = torch.full(
            (segment_count,), duration_us / segment_count, device=device
        )
        self.boundaries = torch.linspace(
            0, duration_us, segment_count + 1, device=device
        )
        times = torch.linspace(0, duration_us, duration_us + 1, device=device)
        self.times = times[None, :, None]
        masks = (self.times >= self.boundaries[:-1]) & (
            self.times < self.boundaries[1:]
        )
        masks[:, -1, -1] = True
        self.segment_masks = masks.float()
        self.current_increments: torch.Tensor | None = None

    def forward(self, increments_xy: torch.Tensor) -> torch.Tensor:
        self.current_increments = increments_xy
        local_times = (self.times - self.boundaries[:-1]) * self.segment_masks
        velocities = increments_xy / self.segment_widths[:, None]
        segment_starts = torch.cumsum(
            torch.cat(
                [torch.zeros((1, 2), device=increments_xy.device), increments_xy[:-1]],
                dim=0,
            ),
            dim=0,
        )
        positions = (
            local_times[..., None] * velocities[None, None, :, :]
            + segment_starts[None, None, :, :]
        )
        return (positions * self.segment_masks[..., None]).sum(dim=2)

    def regularization(self, smoothness: float, magnitude: float) -> torch.Tensor:
        if self.current_increments is None:
            raise RuntimeError("trajectory must be evaluated before regularization")
        increments = self.current_increments
        return smoothness * (increments[1:] - increments[:-1]).square().sum() + (
            magnitude * increments.square().sum()
        )


def register_event_frames(frames: torch.Tensor) -> np.ndarray:
    """Estimate every event frame's subpixel translation from the first frame."""
    # The established baseline performs registration in float64 even though
    # event accumulation is float32. Keep that precision boundary explicit.
    frames = frames.double()
    reference = frames[0, 10:-10, 10:-10].cpu().numpy()
    positions = np.zeros((2, len(frames)), dtype=np.float64)
    for index in range(1, len(frames)):
        moving = frames[index, 10:-10, 10:-10].cpu().numpy()
        positions[:, index] = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=100,
            space="real",
            normalization=None,
        )[0]
    return positions


def initialize_segment_increments(event_frames: torch.Tensor) -> torch.Tensor:
    """Convert registered frame positions into per-segment ``(dx, dy)`` increments."""
    positions_yx = register_event_frames(event_frames)
    positions_xy = -positions_yx[[1, 0]].T
    return torch.from_numpy(positions_xy[1:] - positions_xy[:-1]).float()


def dense_motion_blur_kernel(
    dense_trajectory_xy: torch.Tensor,
    scale: int,
    reference_time_us: int,
    minimum_size: int = 21,
) -> torch.Tensor:
    """Integrate a dense trajectory into a normalized Gaussian blur kernel."""
    path = dense_trajectory_xy
    path_extent = torch.linalg.vector_norm(path.abs().max(dim=1).values)
    size = max(minimum_size, 2 * int(path_extent) * scale + 1)
    center = size // 2
    relative = path - path[:, reference_time_us : reference_time_us + 1]
    x = center + relative[0] * scale
    y = center + relative[1] * scale
    kernel_size = scale + (scale + 1) % 2 + 2
    kernel = gaussian_splat(
        x,
        y,
        torch.ones_like(x),
        (size, size),
        sigma=0.5,
        kernel_size=kernel_size,
        center_mode="floor",
    )
    return kernel / kernel.sum()


def piecewise_motion_blur_kernel(
    increments_xy: torch.Tensor,
    scale: int,
    exposure_end_us: int,
    reference_time_us: int,
    total_duration_us: int,
    samples_per_segment: int = 20,
    minimum_size: int = 21,
) -> torch.Tensor:
    """Integrate the analytic piecewise path over one APS exposure."""
    device = increments_xy.device
    count = len(increments_xy)
    boundaries = torch.linspace(0, total_duration_us, count + 1, device=device)
    cumulative = torch.cat(
        [torch.zeros((1, 2), device=device), torch.cumsum(increments_xy, dim=0)]
    )

    def position_at(time_us: torch.Tensor | float) -> torch.Tensor:
        time = torch.as_tensor(time_us, device=device)
        segment = (torch.searchsorted(boundaries, time) - 1).clamp(0, count - 1)
        fraction = (time - boundaries[segment]) / (
            boundaries[segment + 1] - boundaries[segment]
        )
        return cumulative[segment] + fraction * increments_xy[segment]

    reference = position_at(reference_time_us)
    coordinates: list[torch.Tensor] = []
    for segment in range(count):
        start = boundaries[segment]
        end = boundaries[segment + 1]
        if end <= 0 or start >= exposure_end_us:
            continue
        sample_start = max(start, torch.tensor(0.0, device=device))
        sample_end = min(end, torch.tensor(float(exposure_end_us), device=device))
        times = torch.linspace(
            sample_start, sample_end, samples_per_segment, device=device
        )
        velocity = increments_xy[segment] / (end - start)
        positions = cumulative[segment] + (times - start)[:, None] * velocity
        coordinates.append(positions - reference)
    coordinates_xy = torch.cat(coordinates)
    path_length = torch.linalg.vector_norm(coordinates_xy.abs().max(dim=0).values).item()
    size = max(minimum_size, int(2 * path_length * scale + 1 + path_length))
    center = size // 2
    kernel = bilinear_splat(
        center + coordinates_xy[:, 0] * scale,
        center + coordinates_xy[:, 1] * scale,
        torch.ones(len(coordinates_xy), device=device),
        (size, size),
    )
    return kernel / kernel.sum()
