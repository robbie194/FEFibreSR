"""Image model, losses, and optimizers used by NeuroSR."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer


class AdamP(Optimizer):
    """Adam with projection, used by the reference second-scale reconstruction."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        weight_decay_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "delta": delta,
                "weight_decay_ratio": weight_decay_ratio,
                "nesterov": nesterov,
            },
        )

    def _project(
        self, parameter: Tensor, gradient: Tensor, update: Tensor, group: dict
    ) -> tuple[Tensor, float]:
        view_functions = (
            lambda value: value.view(value.size(0), -1),
            lambda value: value.reshape(1, -1),
        )
        for view in view_functions:
            parameter_view = view(parameter)
            gradient_view = view(gradient)
            cosine = F.cosine_similarity(
                gradient_view.abs(), parameter_view.abs(), dim=1, eps=group["eps"]
            ).abs()
            threshold = group["delta"] / math.sqrt(parameter_view.size(1))
            if cosine.max() < threshold:
                shape = [-1] + [1] * (parameter.ndim - 1)
                direction = parameter / (
                    parameter_view.norm(dim=1).view(shape) + group["eps"]
                )
                update = update - direction * (
                    view(direction * update).sum(dim=1)
                ).view(shape)
                return update, group["weight_decay_ratio"]
        return update, 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["mean"] = torch.zeros_like(parameter)
                    state["variance"] = torch.zeros_like(parameter)
                state["step"] += 1
                mean, variance = state["mean"], state["variance"]
                mean.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient.conj(), value=1 - beta2)
                denominator = variance.sqrt() / math.sqrt(1 - beta2 ** state["step"])
                denominator.add_(group["eps"])
                update = mean / denominator
                if group["nesterov"]:
                    update = (beta1 * mean + (1 - beta1) * gradient) / denominator
                decay_ratio = 1.0
                if parameter.ndim > 1:
                    update, decay_ratio = self._project(
                        parameter, gradient, update, group
                    )
                if group["weight_decay"]:
                    parameter.mul_(
                        1 - group["lr"] * group["weight_decay"] * decay_ratio
                    )
                step_size = group["lr"] / (1 - beta1 ** state["step"])
                parameter.add_(update, alpha=-step_size)
        return loss


def linear_log_intensity(
    intensity: torch.Tensor, threshold: torch.Tensor | float
) -> torch.Tensor:
    """Continuous linear-to-log response used near zero intensity."""
    # The reference sensor response is evaluated in float64. Besides preserving
    # its rounding behaviour, ``+1e-8`` keeps the unselected log branch finite
    # when ``torch.where`` receives a zero-intensity pixel.
    intensity = intensity.double()
    threshold_tensor = torch.as_tensor(
        threshold, device=intensity.device, dtype=intensity.dtype
    )
    slope = torch.log(threshold_tensor) / threshold_tensor
    return torch.where(
        intensity <= threshold_tensor,
        intensity * slope,
        torch.log(intensity + 1e-8),
    )


def forward_gradient(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return forward differences as ``(dy, dx)`` with zeroed wrap boundaries."""
    dx = torch.roll(image, -1, dims=1) - image
    dy = torch.roll(image, -1, dims=0) - image
    dx[:, -1] = 0
    dy[-1, :] = 0
    return dy, dx


def predicted_iwe(log_image: torch.Tensor, displacement_xy: torch.Tensor) -> torch.Tensor:
    """EKLT brightness-change prediction ``-grad(L) dot displacement``."""
    dy, dx = forward_gradient(log_image)
    return -dx * displacement_xy[0] - dy * displacement_xy[1]


def mean_square(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean()


def smooth_l1_to_zero(value: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=beta)


def directional_total_variation(
    image: torch.Tensor,
    normalized_displacement_xy: np.ndarray,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Penalize variation perpendicular to the observed motion direction."""
    dx = image - torch.roll(image, -1, dims=1)
    dy = image - torch.roll(image, -1, dims=0)
    dx[:, -1] = 0
    dy[-1, :] = 0
    directional = -dx * normalized_displacement_xy[1] + dy * normalized_displacement_xy[0]
    return torch.sqrt(directional.abs().square() + epsilon).mean()


def blur_image(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Replicate-pad and convolve while preserving gradients to the kernel."""
    height, width = kernel.shape
    padding = (
        (width - 1) // 2,
        width // 2,
        (height - 1) // 2,
        height // 2,
    )
    padded = F.pad(image[None, None], padding, mode="replicate")
    weight = torch.flip(kernel, dims=(0, 1))[None, None]
    weight = weight / weight.sum()
    return F.conv2d(padded, weight).squeeze()


def block_average(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Average non-overlapping ``scale x scale`` sensor footprints."""
    patches = image.unfold(0, scale, scale).unfold(1, scale, scale)
    return patches.mean(dim=-1).mean(dim=-1)
