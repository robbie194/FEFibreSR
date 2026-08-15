"""Differentiable distal-object to fibre-core forward model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FibreCoreForward(nn.Module):
    """Reproduce GRIN imaging and circular core-aperture integration.

    The model intentionally stops in the core domain. DAVIS spot positions do
    not represent additional distal coordinates; fixed sensor gains are dealt
    with later by the event observation model.
    """

    def __init__(
        self,
        *,
        source_shape: tuple[int, int],
        source_pixel_size_um: float,
        fibre_shape: tuple[int, int],
        fibre_pixel_size_um: float,
        grin_magnification: float,
        grin_sigma_um: float,
        grin_transmission: float,
        fibre_transmission: float,
        core_centres_xy_um: np.ndarray,
        core_diameter_um: float,
        aperture_supersample: int,
    ) -> None:
        super().__init__()
        from fibre_sim.fibre import circular_pixel_coverage_kernel

        source_h, source_w = source_shape
        fibre_h, fibre_w = fibre_shape
        x_um = (
            torch.arange(fibre_w, dtype=torch.float32) - (fibre_w - 1) / 2
        ) * float(fibre_pixel_size_um)
        y_um = (
            torch.arange(fibre_h, dtype=torch.float32) - (fibre_h - 1) / 2
        ) * float(fibre_pixel_size_um)
        grid_y_um, grid_x_um = torch.meshgrid(y_um, x_um, indexing="ij")
        source_x_px = (source_w - 1) / 2 + (
            grid_x_um / float(grin_magnification)
        ) / float(source_pixel_size_um)
        source_y_px = (source_h - 1) / 2 + (
            grid_y_um / float(grin_magnification)
        ) / float(source_pixel_size_um)
        self.register_buffer("base_source_x_px", source_x_px)
        self.register_buffer("base_source_y_px", source_y_px)

        centres = torch.as_tensor(core_centres_xy_um, dtype=torch.float32)
        core_x_px = (fibre_w - 1) / 2 + centres[:, 0] / float(fibre_pixel_size_um)
        core_y_px = (fibre_h - 1) / 2 + centres[:, 1] / float(fibre_pixel_size_um)
        core_grid = torch.stack(
            (
                2 * core_x_px / (fibre_w - 1) - 1,
                2 * core_y_px / (fibre_h - 1) - 1,
            ),
            dim=-1,
        ).reshape(1, 1, -1, 2)
        self.register_buffer("core_grid", core_grid)

        aperture = circular_pixel_coverage_kernel(
            float(core_diameter_um),
            float(fibre_pixel_size_um),
            int(aperture_supersample),
        )
        aperture = aperture / aperture.sum()
        self.register_buffer(
            "aperture_kernel",
            torch.as_tensor(aperture, dtype=torch.float32)[None, None],
        )
        sigma_px = float(grin_sigma_um) / float(fibre_pixel_size_um)
        if sigma_px > 0:
            radius = int(round(3 * sigma_px))
            coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
            gaussian_1d = torch.exp(-0.5 * (coordinates / sigma_px).square())
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        else:
            gaussian_2d = torch.empty(0, dtype=torch.float32)
        self.register_buffer("grin_kernel", gaussian_2d[None, None])
        self.source_shape = tuple(map(int, source_shape))
        self.source_pixel_size_um = float(source_pixel_size_um)
        self.transmission = float(grin_transmission) * float(fibre_transmission)

    def forward(self, object_image: torch.Tensor, shifts_xy_um: torch.Tensor) -> torch.Tensor:
        """Return scalar core intensities with shape ``[time, core]``."""
        if object_image.ndim != 2:
            raise ValueError("object_image must have shape [height, width]")
        if tuple(object_image.shape) != self.source_shape:
            raise ValueError(
                f"expected object shape {self.source_shape}, got {tuple(object_image.shape)}"
            )
        shifts = shifts_xy_um.to(device=object_image.device, dtype=object_image.dtype)
        source_h, source_w = self.source_shape
        x_px = self.base_source_x_px[None] - (
            shifts[:, 0, None, None] / self.source_pixel_size_um
        )
        y_px = self.base_source_y_px[None] - (
            shifts[:, 1, None, None] / self.source_pixel_size_um
        )
        sampling_grid = torch.stack(
            (2 * x_px / (source_w - 1) - 1, 2 * y_px / (source_h - 1) - 1),
            dim=-1,
        )
        source_batch = object_image[None, None].expand(len(shifts), -1, -1, -1)
        fibre_input = F.grid_sample(
            source_batch,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        if self.grin_kernel.numel():
            radius = self.grin_kernel.shape[-1] // 2
            fibre_input = F.conv2d(
                F.pad(fibre_input, (radius, radius, radius, radius), mode="replicate"),
                self.grin_kernel,
            )
        kernel_h, kernel_w = self.aperture_kernel.shape[-2:]
        padding = (kernel_w // 2, kernel_w // 2, kernel_h // 2, kernel_h // 2)
        aperture_average = F.conv2d(
            F.pad(fibre_input, padding, mode="replicate"), self.aperture_kernel
        )
        core_grid = self.core_grid.expand(len(shifts), -1, -1, -1)
        signals = F.grid_sample(
            aperture_average,
            core_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return signals[:, 0, 0] * self.transmission


def expand_latent_image(latent: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
    """Bilinearly expand the reconstruction grid to the simulator source grid."""
    return F.interpolate(
        latent[None, None], size=output_shape, mode="bilinear", align_corners=True
    )[0, 0]
