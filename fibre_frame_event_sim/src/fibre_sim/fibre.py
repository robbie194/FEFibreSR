from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def generate_hex_core_centres(
    field_size_um: float,
    pitch_um: float,
    core_diameter_um: float,
) -> np.ndarray:
    """Return an approximately centred hexagonal lattice inside a square ROI.

    Each returned core disk lies fully inside the local simulated field. The
    array columns are physical ``(x_um, y_um)`` coordinates.
    """
    if field_size_um <= 0 or pitch_um <= 0 or core_diameter_um <= 0:
        raise ValueError("field size, pitch and diameter must be positive")
    radius = core_diameter_um / 2
    limit = field_size_um / 2 - radius
    row_pitch = np.sqrt(3.0) * pitch_um / 2
    row_indices = np.arange(
        int(np.floor(-limit / row_pitch)) - 1,
        int(np.ceil(limit / row_pitch)) + 2,
    )
    points: list[tuple[float, float]] = []
    for row in row_indices:
        y = row * row_pitch
        if abs(y) > limit + 1e-9:
            continue
        offset = (row & 1) * pitch_um / 2
        col_indices = np.arange(
            int(np.floor((-limit - offset) / pitch_um)) - 1,
            int(np.ceil((limit - offset) / pitch_um)) + 2,
        )
        for col in col_indices:
            x = col * pitch_um + offset
            if abs(x) <= limit + 1e-9:
                points.append((float(x), float(y)))
    centres = np.asarray(points, dtype=np.float32)
    order = np.lexsort((centres[:, 0], centres[:, 1]))
    return centres[order]


def circular_pixel_coverage_kernel(
    diameter_um: float,
    pixel_size_um: float,
    supersample: int = 32,
) -> np.ndarray:
    """Subpixel area coverage of a centred circular core for nearby pixels."""
    radius_px = diameter_um / (2 * pixel_size_um)
    half = int(np.ceil(radius_px + 0.5))
    coords = np.arange(-half, half + 1, dtype=np.float64)
    sub = (np.arange(supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    kernel = np.empty((len(coords), len(coords)), dtype=np.float32)
    for iy, cy in enumerate(coords):
        for ix, cx in enumerate(coords):
            sx, sy = np.meshgrid(cx + sub, cy + sub)
            kernel[iy, ix] = np.mean(sx * sx + sy * sy <= radius_px * radius_px)
    nonzero_rows = np.flatnonzero(np.any(kernel > 0, axis=1))
    nonzero_cols = np.flatnonzero(np.any(kernel > 0, axis=0))
    return kernel[
        nonzero_rows[0] : nonzero_rows[-1] + 1,
        nonzero_cols[0] : nonzero_cols[-1] + 1,
    ]


def _centres_to_image_coordinates(
    centres_xy_um: np.ndarray,
    shape: tuple[int, int],
    pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x = (width - 1) / 2 + centres_xy_um[:, 0] / pixel_size_um
    y = (height - 1) / 2 + centres_xy_um[:, 1] / pixel_size_um
    return x.astype(np.float32), y.astype(np.float32)


def _bilinear_scatter(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = x - x0
    dy = y - y0
    for ox, oy, weight in (
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xx, yy = x0 + ox, y0 + oy
        valid = (xx >= 0) & (xx < shape[1]) & (yy >= 0) & (yy < shape[0])
        np.add.at(image, (yy[valid], xx[valid]), values[valid] * weight[valid])
    return image


def simulate_fibre_sequence(
    grin_frames: np.ndarray,
    core_centres_xy_um: np.ndarray,
    *,
    pixel_size_um: float,
    core_diameter_um: float,
    aperture_supersample: int = 32,
    transmission: float = 1.0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample each circular core aperture and render circular distal spots.

    Returns ``(distal_frames[T,H,W], core_signals[T,N])``. The average over
    each core entrance aperture is used as its scalar intensity.
    """
    frames = np.asarray(grin_frames, dtype=np.float32)
    centres = np.asarray(core_centres_xy_um, dtype=np.float32)
    if frames.ndim != 3 or centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("expected frames [T,H,W] and centres [N,2]")
    kernel = circular_pixel_coverage_kernel(
        core_diameter_um, pixel_size_um, aperture_supersample
    )
    averaging_kernel = kernel / np.sum(kernel)
    x, y = _centres_to_image_coordinates(centres, frames.shape[1:], pixel_size_um)
    map_x = x.reshape(1, -1)
    map_y = y.reshape(1, -1)
    distal = np.empty_like(frames, dtype=np.float32)
    signals = np.empty((frames.shape[0], len(centres)), dtype=np.float32)
    for index, frame in enumerate(frames):
        aperture_average = cv2.filter2D(
            frame, -1, averaging_kernel, borderType=cv2.BORDER_REPLICATE
        )
        signal = cv2.remap(
            aperture_average,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)
        signal = np.asarray(signal * transmission, dtype=np.float32)
        impulses = _bilinear_scatter(signal, x, y, frames.shape[1:])
        distal[index] = cv2.filter2D(
            impulses, -1, kernel, borderType=cv2.BORDER_CONSTANT
        )
        signals[index] = signal
        if progress is not None:
            progress(index + 1, frames.shape[0])
    return distal, signals

