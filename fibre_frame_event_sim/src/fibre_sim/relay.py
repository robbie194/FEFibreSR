from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def relay_to_sensor_sequence(
    fibre_frames: np.ndarray,
    *,
    fibre_pixel_size_um: float,
    sensor_shape_px: tuple[int, int],
    sensor_pixel_pitch_um: float,
    magnification: float,
    psf_sigma_sensor_um: float = 0.0,
    pixel_integration_supersample: int = 2,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Relay the distal fibre plane to area-integrating sensor pixels."""
    frames = np.asarray(fibre_frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"fibre_frames must have shape [T,H,W], got {frames.shape}")
    if min(fibre_pixel_size_um, sensor_pixel_pitch_um, magnification) <= 0:
        raise ValueError("pixel sizes and magnification must be positive")
    ss = int(pixel_integration_supersample)
    if ss < 1:
        raise ValueError("pixel_integration_supersample must be >= 1")

    sensor_h, sensor_w = map(int, sensor_shape_px)
    sub_h, sub_w = sensor_h * ss, sensor_w * ss
    sub_pitch = sensor_pixel_pitch_um / ss
    sensor_x = (np.arange(sub_w, dtype=np.float32) - (sub_w - 1) / 2) * sub_pitch
    sensor_y = (np.arange(sub_h, dtype=np.float32) - (sub_h - 1) / 2) * sub_pitch
    sx, sy = np.meshgrid(sensor_x, sensor_y)
    fibre_h, fibre_w = frames.shape[1:]
    map_x = ((fibre_w - 1) / 2 + sx / magnification / fibre_pixel_size_um).astype(np.float32)
    map_y = ((fibre_h - 1) / 2 + sy / magnification / fibre_pixel_size_um).astype(np.float32)

    output = np.empty((frames.shape[0], sensor_h, sensor_w), dtype=np.float32)
    sigma_subpx = psf_sigma_sensor_um / sub_pitch
    for index, frame in enumerate(frames):
        relayed = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if sigma_subpx > 0:
            relayed = cv2.GaussianBlur(
                relayed, (0, 0), sigmaX=sigma_subpx, sigmaY=sigma_subpx,
                borderType=cv2.BORDER_CONSTANT,
            )
        output[index] = relayed.reshape(sensor_h, ss, sensor_w, ss).mean(axis=(1, 3))
        if progress is not None:
            progress(index + 1, frames.shape[0])
    return output

