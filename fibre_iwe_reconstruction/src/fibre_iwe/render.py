"""Differentiable image sampling, event warping and splatting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bilinear_splat(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Accumulate subpixel samples while preserving coordinate gradients."""
    height, width = shape
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    dx, dy = x - x0, y - y0
    x0, y0 = x0.long(), y0.long()
    output = torch.zeros(shape, dtype=weights.dtype, device=weights.device)
    for ox, oy, factor in (
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ):
        target_x, target_y = x0 + ox, y0 + oy
        valid = (
            (target_x >= 0)
            & (target_x < width)
            & (target_y >= 0)
            & (target_y < height)
        )
        output.index_put_(
            (target_y[valid], target_x[valid]),
            weights[valid] * factor[valid],
            accumulate=True,
        )
    return output


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    radius = max(1, int(round(3 * sigma)))
    values = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-0.5 * (values / sigma).square())
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    return F.conv2d(
        image[None, None], kernel[None, None], padding=radius
    )[0, 0]


def sample_trajectory(
    control_positions: torch.Tensor, normalized_times: torch.Tensor
) -> torch.Tensor:
    """Linearly sample ``[K+1,2]`` positions at normalized event times."""
    segment_count = len(control_positions) - 1
    scaled = normalized_times.clamp(0, 1) * segment_count
    left = torch.floor(scaled).long().clamp(max=segment_count - 1)
    fraction = (scaled - left).unsqueeze(1)
    return control_positions[left] * (1 - fraction) + control_positions[left + 1] * fraction


def warp_events(
    xy: torch.Tensor,
    normalized_times: torch.Tensor,
    control_positions: torch.Tensor,
    reference_time: float = 0.5,
) -> torch.Tensor:
    event_position = sample_trajectory(control_positions, normalized_times)
    reference = sample_trajectory(
        control_positions,
        torch.as_tensor([reference_time], device=xy.device, dtype=xy.dtype),
    )[0]
    return xy - event_position + reference


def render_iwe(
    xy: torch.Tensor,
    normalized_times: torch.Tensor,
    polarity: torch.Tensor,
    control_positions: torch.Tensor,
    shape: tuple[int, int],
    sigma: float,
    signed: bool,
) -> torch.Tensor:
    warped = warp_events(xy, normalized_times, control_positions)
    weights = polarity if signed else torch.ones_like(polarity)
    return gaussian_blur(bilinear_splat(warped[:, 0], warped[:, 1], weights, shape), sigma)


def sample_image_at_points(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a 2-D image at pixel coordinates."""
    height, width = image.shape
    grid = torch.stack(
        (2 * xy[:, 0] / (width - 1) - 1, 2 * xy[:, 1] / (height - 1) - 1),
        dim=-1,
    ).reshape(1, 1, -1, 2)
    return F.grid_sample(
        image[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, 0, 0]


def shift_image(image: torch.Tensor, shift_xy: torch.Tensor) -> torch.Tensor:
    """Return ``image(x-shift_x, y-shift_y)``."""
    height, width = image.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    grid = torch.stack(
        (
            2 * (x - shift_xy[0]) / (width - 1) - 1,
            2 * (y - shift_xy[1]) / (height - 1) - 1,
        ),
        dim=-1,
    )[None]
    return F.grid_sample(
        image[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, 0]
