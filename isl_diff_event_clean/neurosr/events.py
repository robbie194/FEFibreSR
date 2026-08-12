"""Differentiable event warping and image formation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def signed_polarity(polarity: torch.Tensor) -> torch.Tensor:
    """Convert DAVIS polarity ``{0, 1}`` to signed weights ``{-1, +1}``."""
    return torch.where(polarity == 0, -torch.ones_like(polarity), polarity)


def bilinear_splat(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Accumulate subpixel samples into their four neighbouring pixels."""
    x = x.float()
    y = y.float()
    weights = weights.float()
    height, width = image_shape
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    dx = x - x0
    dy = y - y0
    x0 = x0.long()
    y0 = y0.long()
    valid = (x0 >= 0) & (x0 < width - 1) & (y0 >= 0) & (y0 < height - 1)

    x0, y0 = x0[valid], y0[valid]
    dx, dy, weights = dx[valid], dy[valid], weights[valid]
    image = torch.zeros(image_shape, device=x.device, dtype=torch.float32)
    image.index_put_((y0, x0), weights * (1 - dx) * (1 - dy), accumulate=True)
    image.index_put_((y0, x0 + 1), weights * dx * (1 - dy), accumulate=True)
    image.index_put_((y0 + 1, x0), weights * (1 - dx) * dy, accumulate=True)
    image.index_put_((y0 + 1, x0 + 1), weights * dx * dy, accumulate=True)
    return image


def gaussian_splat(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    image_shape: tuple[int, int],
    sigma: float,
    kernel_size: int,
    center_mode: str = "round",
) -> torch.Tensor:
    """Render each event with a normalized, subpixel-shifted Gaussian kernel."""
    x = x.float()
    y = y.float()
    weights = weights.float()
    if center_mode == "round":
        x_center = torch.round(x).long()
        y_center = torch.round(y).long()
    elif center_mode == "floor":
        x_center = torch.floor(x).long()
        y_center = torch.floor(y).long()
    else:
        raise ValueError(f"unsupported Gaussian center mode: {center_mode}")
    dx = x - x_center
    dy = y - y_center
    radius = (kernel_size - 1) // 2
    offsets = torch.arange(-radius, radius + 1, device=x.device)

    wx = torch.exp(-0.5 * (dx[:, None] - offsets[None, :]).square() / sigma**2)
    wy = torch.exp(-0.5 * (dy[:, None] - offsets[None, :]).square() / sigma**2)
    kernels = wx[:, :, None] * wy[:, None, :]
    kernels = kernels / (kernels.sum(dim=(1, 2), keepdim=True) + 1e-12)

    count = kernel_size * kernel_size
    target_x = (
        x_center[:, None, None] + offsets[None, :, None]
    ).expand(-1, -1, kernel_size).reshape(-1)
    target_y = (
        y_center[:, None, None] + offsets[None, None, :]
    ).expand(-1, kernel_size, -1).reshape(-1)
    values = kernels.reshape(-1) * weights.repeat_interleave(count)
    height, width = image_shape
    valid = (
        (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
    )
    image = torch.zeros(image_shape, device=x.device, dtype=torch.float32)
    image.index_put_(
        (target_y[valid], target_x[valid]), values[valid].float(), accumulate=True
    )
    return image


def event_image(
    x: torch.Tensor,
    y: torch.Tensor,
    polarity: torch.Tensor,
    image_shape: tuple[int, int],
    *,
    signed: bool,
) -> torch.Tensor:
    weights = signed_polarity(polarity) if signed else polarity.abs()
    return bilinear_splat(x, y, weights, image_shape)


def gaussian_event_image(
    x: torch.Tensor,
    y: torch.Tensor,
    polarity: torch.Tensor,
    sensor_shape: tuple[int, int],
    scale: int,
    sigma: float,
    *,
    signed: bool,
) -> torch.Tensor:
    """Render a high-resolution IWE using the reference Gaussian convention."""
    weights = signed_polarity(polarity) if signed else polarity.abs()
    output_shape = (sensor_shape[0] * scale, sensor_shape[1] * scale)
    kernel_size = int(3 * sigma * scale + 1) | 1
    return gaussian_splat(
        x * scale,
        y * scale,
        weights,
        output_shape,
        sigma=scale * sigma / 2,
        kernel_size=kernel_size,
    )


def warp_events_to_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    timestamps_us: torch.Tensor,
    dense_trajectory_xy: torch.Tensor,
    reference_time_us: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a dense trajectory at every event and move events to one time."""
    trajectory = dense_trajectory_xy.unsqueeze(-1)
    while trajectory.ndim < 4:
        trajectory = trajectory.unsqueeze(0)

    coordinates = torch.stack((timestamps_us, timestamps_us), dim=1)
    coordinates = coordinates[None, :, None, :]
    denominator = timestamps_us.max() - 1
    coordinates = coordinates / denominator * 2 - 1
    displacement = F.grid_sample(
        trajectory.double(), coordinates.double(), align_corners=True
    )
    reference_index = int(float(reference_time_us) - float(timestamps_us[0]))
    reference = trajectory[:, :, reference_index, :].squeeze()
    sampled = displacement.squeeze(0).squeeze(-1)
    return x - sampled[0] + reference[0], y - sampled[1] + reference[1]


def numpy_event_frame(
    x: np.ndarray,
    y: np.ndarray,
    timestamps_us: np.ndarray,
    polarity: np.ndarray,
    sensor_shape: tuple[int, int],
    start_us: float,
    duration_us: float,
) -> np.ndarray:
    """Build the unsigned event-count frame used for registration."""
    selected = (timestamps_us >= start_us) & (
        timestamps_us <= start_us + duration_us
    )
    image = event_image(
        torch.from_numpy(x[selected]).float(),
        torch.from_numpy(y[selected]).float(),
        torch.from_numpy(polarity[selected]).float(),
        sensor_shape,
        signed=True,
    ).numpy()
    image.flat[int(np.argmax(image))] = 0
    return image
