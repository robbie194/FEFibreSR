from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def simulate_grin_sequence(
    object_image: np.ndarray,
    shifts_um: np.ndarray,
    *,
    object_pixel_size_um: float,
    fibre_shape_px: tuple[int, int],
    fibre_pixel_size_um: float,
    magnification: float = 1.0,
    sigma_um: float = 0.0,
    transmission: float = 1.0,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Image a moving object onto the proximal fibre plane.

    Coordinates are centred on both arrays. ``shifts_um[t]`` is the object
    translation (x, y); positive x therefore moves image features to the
    right at the fibre plane. The output is ``[T, H_f, W_f]`` float32.
    """
    source = np.asarray(object_image, dtype=np.float32)
    shifts = np.asarray(shifts_um, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"object_image must be 2-D, got {source.shape}")
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError(f"shifts_um must have shape [T,2], got {shifts.shape}")
    if object_pixel_size_um <= 0 or fibre_pixel_size_um <= 0:
        raise ValueError("pixel sizes must be positive")
    if magnification <= 0 or transmission < 0:
        raise ValueError("magnification must be positive and transmission nonnegative")

    out_h, out_w = map(int, fibre_shape_px)
    src_h, src_w = source.shape
    x_f = (np.arange(out_w, dtype=np.float32) - (out_w - 1) / 2) * fibre_pixel_size_um
    y_f = (np.arange(out_h, dtype=np.float32) - (out_h - 1) / 2) * fibre_pixel_size_um
    grid_x, grid_y = np.meshgrid(x_f, y_f)

    sequence = np.empty((len(shifts), out_h, out_w), dtype=np.float32)
    for index, (shift_x, shift_y) in enumerate(shifts):
        map_x = (
            (src_w - 1) / 2
            + (grid_x / magnification - shift_x) / object_pixel_size_um
        ).astype(np.float32)
        map_y = (
            (src_h - 1) / 2
            + (grid_y / magnification - shift_y) / object_pixel_size_um
        ).astype(np.float32)
        frame = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if sigma_um > 0:
            sigma_px = float(sigma_um) / float(fibre_pixel_size_um)
            frame = cv2.GaussianBlur(
                frame, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px,
                borderType=cv2.BORDER_REPLICATE,
            )
        sequence[index] = np.asarray(frame * transmission, dtype=np.float32)
        if progress is not None:
            progress(index + 1, len(shifts))
    return sequence

