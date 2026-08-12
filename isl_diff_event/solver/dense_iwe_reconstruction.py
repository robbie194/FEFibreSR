from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


def as_image_tensor(x: ArrayLike, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x))
    if device is not None:
        x = x.to(device=torch.device(device))
    x = x.float()
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
        x = x[0, 0]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError("image tensor must have shape [H,W], [1,H,W], or [1,1,H,W]")
    return x


def as_flow_tensor(
    flow: ArrayLike,
    image_shape: tuple[int, int],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if not torch.is_tensor(flow):
        flow = torch.from_numpy(np.asarray(flow))
    if device is not None:
        flow = flow.to(device=torch.device(device))
    flow = flow.float()
    if flow.ndim == 4 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError("flow tensor must have shape [2,H,W] or [1,2,H,W]")
    if tuple(flow.shape[-2:]) != tuple(image_shape):
        flow = F.interpolate(flow[None], size=image_shape, mode="bilinear", align_corners=True)[0]
    return flow


def lin_log_intensity(
    intensity: torch.Tensor,
    threshold: Union[float, torch.Tensor] = 1.000001,
    input_scale: float = 255.0,
    output_scale: float = float(np.log(255.0)),
) -> torch.Tensor:
    x = intensity * float(input_scale)
    threshold_t = torch.as_tensor(threshold, device=x.device, dtype=x.dtype).clamp_min(1e-6)
    slope = torch.log(threshold_t) / threshold_t
    log_image = torch.where(x <= threshold_t, x * slope, torch.log(x + 1e-8))
    return log_image / float(output_scale)


def dense_flow_grid(flow_xy: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    flow_xy = as_flow_tensor(flow_xy, image_shape)
    _, h, w = flow_xy.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow_xy.device, dtype=flow_xy.dtype),
        torch.arange(w, device=flow_xy.device, dtype=flow_xy.dtype),
        indexing="ij",
    )
    sample_x = xx + flow_xy[0]
    sample_y = yy + flow_xy[1]
    grid_x = sample_x / max(w - 1, 1) * 2.0 - 1.0
    grid_y = sample_y / max(h - 1, 1) * 2.0 - 1.0
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def unit_flow_mag_mask(
    flow_xy: torch.Tensor,
    eps: float = 1e-1,
    border: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if flow_xy.ndim != 3 or flow_xy.shape[0] != 2:
        raise ValueError("flow_xy must have shape [2,H,W]")
    mag = torch.linalg.vector_norm(flow_xy, dim=0)
    safe_mag = torch.where(mag > float(eps), mag, torch.ones_like(mag))
    mask = torch.isfinite(mag) & (mag > float(eps))
    if border > 0:
        border_mask = torch.zeros_like(mask)
        border_mask[border:-border, border:-border] = True
        #mask = border_mask
        mask = mask & border_mask
    mask = mask.float()
    unit_flow = flow_xy / safe_mag.unsqueeze(0)
    unit_flow = unit_flow * mask.unsqueeze(0)
    return unit_flow.float(), safe_mag.float(), mask

from function.optimizer import forward_grad

def make_iwe_from_log_image_eklt(
    log_image: torch.Tensor,
    flow_map: torch.Tensor,
) -> torch.Tensor:
    grad_y, grad_x = forward_grad(log_image)

    predicted_iwe = -(
        grad_x * flow_map[0]
        + grad_y * flow_map[1]
    )

    predicted_iwe = predicted_iwe.clone()
    predicted_iwe[-1, :] = 0.0
    predicted_iwe[:, -1] = 0.0

    return predicted_iwe

def prepare_iwe_for_unit_flow(
    iwe: torch.Tensor,
    flow_xy: torch.Tensor,
    eps: float = 1e-6,
    border: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    iwe = as_image_tensor(iwe)
    flow_xy = as_flow_tensor(flow_xy, tuple(iwe.shape), iwe.device)
    unit_flow, flow_mag, mask = unit_flow_mag_mask(flow_xy, eps=eps, border=border)
    target = torch.nan_to_num(iwe / flow_mag, nan=0.0, posinf=0.0, neginf=0.0)
    target = target * mask
    return target.float(), unit_flow.float(), mask.float()


def warp_iwe_from_log_image(
    log_image: torch.Tensor,
    flow_map: torch.Tensor,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    grid = dense_flow_grid(flow_map, tuple(log_image.shape)).to(device=log_image.device, dtype=log_image.dtype)
    warped_log_image = F.grid_sample(
        log_image[None, None],
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )[0, 0]
    return log_image - warped_log_image


def warp_iwe_from_intensity(
    intensity: torch.Tensor,
    flow_xy: torch.Tensor,
    lin_log_threshold: Union[float, torch.Tensor] = 1.000001,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    log_image = lin_log_intensity(intensity, threshold=lin_log_threshold)
    return warp_iwe_from_log_image(log_image, flow_xy, padding_mode=padding_mode)


def valid_flow_mask(flow_xy: torch.Tensor, eps: float = 1e-6, border: int = 0) -> torch.Tensor:
    if flow_xy.ndim != 3 or flow_xy.shape[0] != 2:
        raise ValueError("flow_xy must have shape [2,H,W]")
    mag = torch.linalg.vector_norm(flow_xy, dim=0)
    mask = torch.isfinite(mag) & (mag > float(eps))
    if border > 0:
        border_mask = torch.zeros_like(mask)
        border_mask[border:-border, border:-border] = True
        mask = mask & border_mask
    return mask.float()


def normalize_iwe(iwe: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    iwe = as_image_tensor(iwe)
    if mask is None:
        masked_iwe = iwe
    else:
        masked_iwe = iwe * as_image_tensor(mask, device=iwe.device)
    norm = torch.linalg.vector_norm(masked_iwe.reshape(-1)).clamp_min(float(eps))
    return iwe / norm


def iwe_l2_loss(pred_iwe: torch.Tensor, target_iwe: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred_iwe = as_image_tensor(pred_iwe)
    target_iwe = as_image_tensor(target_iwe, device=pred_iwe.device).detach()
    if mask is None:
        return (pred_iwe - target_iwe).square().mean()
    mask = as_image_tensor(mask, device=pred_iwe.device)
    denom = mask.sum().clamp_min(1.0)
    return ((pred_iwe - target_iwe).square() * mask).sum() / denom


def best_iwe_scale(
    pred_iwe: torch.Tensor,
    target_iwe: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_iwe = as_image_tensor(pred_iwe)
    target_iwe = as_image_tensor(target_iwe, device=pred_iwe.device).detach()
    if mask is not None:
        mask = as_image_tensor(mask, device=pred_iwe.device)
        pred_iwe = pred_iwe * mask
        target_iwe = target_iwe * mask
    numerator = (pred_iwe * target_iwe).sum()
    denominator = pred_iwe.square().sum().clamp_min(float(eps))
    return numerator / denominator


def normalized_iwe_l2_loss(
    pred_iwe: torch.Tensor,
    target_iwe: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_iwe = as_image_tensor(pred_iwe)
    target_iwe = as_image_tensor(target_iwe, device=pred_iwe.device).detach()
    if mask is not None:
        mask = as_image_tensor(mask, device=pred_iwe.device)
    pred_n = normalize_iwe(pred_iwe, mask)
    target_n = normalize_iwe(target_iwe, mask)
    if mask is None:
        return (pred_n - target_n).square().mean()
    denom = mask.sum().clamp_min(1.0)
    return ((pred_n - target_n).square() * mask).sum() / denom


def prepare_iwe_target_and_flow(
    target_iwe: ArrayLike,
    flow_xy: ArrayLike,
    device: Optional[Union[str, torch.device]] = None,
    border: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = as_image_tensor(target_iwe, device=device)
    flow = as_flow_tensor(flow_xy, tuple(target.shape), target.device)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return prepare_iwe_for_unit_flow(target, flow, border=border)
