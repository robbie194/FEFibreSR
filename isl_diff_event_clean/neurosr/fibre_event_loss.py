"""Observation models and regularisers for fibre reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lin_log_sensor_response(intensity_dn: torch.Tensor, threshold_dn: float = 20.0) -> torch.Tensor:
    """Differentiable equivalent of v2e's linear/logarithmic response."""
    threshold = torch.as_tensor(
        threshold_dn, dtype=intensity_dn.dtype, device=intensity_dn.device
    )
    slope = torch.log(threshold) / threshold
    safe = torch.clamp(intensity_dn, min=torch.finfo(intensity_dn.dtype).tiny)
    return torch.where(intensity_dn <= threshold, intensity_dn * slope, torch.log(safe))


def trapezoidal_average(values: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
    """Time-average a ``[time, ...]`` tensor using trapezoidal integration."""
    duration = timestamps_s[-1] - timestamps_s[0]
    pair_areas = (values[:-1] + values[1:]) * 0.5
    dt_shape = (-1,) + (1,) * (values.ndim - 1)
    return (pair_areas * torch.diff(timestamps_s).reshape(dt_shape)).sum(dim=0) / duration


def predict_cumulative_event_change(
    core_signals: torch.Tensor,
    sensor_gain: torch.Tensor,
    input_white_dn: float,
) -> torch.Tensor:
    """Predict cumulative v2e contrast at each calibrated sensor pixel."""
    pixel_intensity = (
        core_signals[:, :, None] * sensor_gain[None] * float(input_white_dn)
    )
    response = lin_log_sensor_response(pixel_intensity)
    return response - response[0:1]


def robust_event_loss(
    predicted_change: torch.Tensor,
    observed_change: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Huber loss for threshold-quantised cumulative event observations."""
    return F.smooth_l1_loss(predicted_change, observed_change, beta=float(beta))


def isotropic_total_variation(image: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Edge-preserving spatial regularity on the latent object grid."""
    dx = image[:, 1:] - image[:, :-1]
    dy = image[1:, :] - image[:-1, :]
    dx = dx[:-1, :]
    dy = dy[:, :-1]
    return torch.sqrt(dx.square() + dy.square() + epsilon).mean()
