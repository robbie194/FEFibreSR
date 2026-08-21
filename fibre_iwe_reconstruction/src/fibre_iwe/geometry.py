"""Core-mask generation and mask-derived geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .io import CoreMask


@dataclass(frozen=True)
class SimulatedCoreGeometry:
    """Core labels plus a private, simulation-only proximal response map."""

    core_mask: CoreMask
    proximal_response: np.ndarray


def hexagonal_centres(
    shape: tuple[int, int], pitch_px: float, margin_px: float
) -> np.ndarray:
    """Return an approximately centred hexagonal lattice in sensor pixels."""
    height, width = shape
    row_pitch = np.sqrt(3.0) * pitch_px / 2.0
    points: list[tuple[float, float]] = []
    row = 0
    y = margin_px
    while y <= height - 1 - margin_px:
        offset = (row % 2) * pitch_px / 2.0
        x = margin_px + offset
        while x <= width - 1 - margin_px:
            points.append((x, y))
            x += pitch_px
        row += 1
        y = margin_px + row * row_pitch
    return np.asarray(points, dtype=np.float32)


def generate_irregular_core_mask(
    shape: tuple[int, int],
    centres_xy: np.ndarray,
    radius_px: float,
    seed: int,
) -> SimulatedCoreGeometry:
    """Create non-circular, nonuniform proximal spots without overlap."""
    rng = np.random.default_rng(seed)
    height, width = shape
    labels = np.zeros(shape, dtype=np.int32)
    response = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    for core_index, (cx, cy) in enumerate(centres_xy, start=1):
        angle = rng.uniform(0, np.pi)
        axis_x = radius_px * rng.uniform(0.82, 1.14)
        axis_y = radius_px * rng.uniform(0.82, 1.14)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = xx - cx, yy - cy
        rotated_x = cos_a * dx + sin_a * dy
        rotated_y = -sin_a * dx + cos_a * dy
        distance = (rotated_x / axis_x) ** 2 + (rotated_y / axis_y) ** 2
        boundary_noise = cv2.GaussianBlur(
            rng.normal(0, 1, shape).astype(np.float32), (0, 0), 0.8
        )
        candidate = (distance + 0.12 * boundary_noise < 1.0) & (labels == 0)
        if candidate.sum() < 4:
            raise RuntimeError(f"core {core_index} produced too small a mask")
        labels[candidate] = core_index
        local_gain = np.exp(-0.65 * distance[candidate])
        local_gain *= rng.lognormal(mean=0.0, sigma=0.12, size=candidate.sum())
        # Equal median throughput keeps pixel gain unknown but avoids injecting a
        # separate per-core calibration problem into this baseline experiment.
        local_gain /= np.median(local_gain)
        response[candidate] = local_gain.astype(np.float32)
    return SimulatedCoreGeometry(CoreMask(labels), response)


def centres_from_mask(labels: np.ndarray) -> np.ndarray:
    """Compute core centres using only the calibrated label image."""
    core_count = int(labels.max())
    centres = np.empty((core_count, 2), dtype=np.float32)
    for label in range(1, core_count + 1):
        y, x = np.nonzero(labels == label)
        if len(x) == 0:
            raise ValueError(f"core label {label} is empty")
        centres[label - 1] = (float(x.mean()), float(y.mean()))
    return centres


def core_pixel_lists(mask: CoreMask) -> list[np.ndarray]:
    """Return sensor ``(x,y)`` candidates for each core."""
    pixels: list[np.ndarray] = []
    for label in range(1, int(mask.labels.max()) + 1):
        y, x = np.nonzero(mask.labels == label)
        pixels.append(np.column_stack((x, y)).astype(np.int32))
    return pixels
