from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def prepare_source(
    image_path: str | Path,
    *,
    crop_center_px: tuple[int, int],
    crop_size_px: int,
    output_shape_px: tuple[int, int],
    intensity_floor: float,
) -> tuple[np.ndarray, Image.Image, dict[str, Any]]:
    """Crop the USAF image and return a linear float intensity image.

    ``crop_center_px`` uses (x, y); ``output_shape_px`` uses (height, width).
    The source target is essentially binary, so gamma decoding has negligible
    effect. A nonzero dark floor keeps the event-camera log response finite.
    """
    path = Path(image_path)
    image = Image.open(path).convert("L")
    width, height = image.size
    center_x, center_y = map(int, crop_center_px)
    size = int(crop_size_px)
    left = center_x - size // 2
    top = center_y - size // 2
    right = left + size
    bottom = top + size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"crop {(left, top, right, bottom)} exceeds image {(width, height)}"
        )

    crop = image.crop((left, top, right, bottom))
    out_h, out_w = map(int, output_shape_px)
    resized = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)
    normalized = np.asarray(resized, dtype=np.float32) / 255.0
    floor = float(intensity_floor)
    if not (0 <= floor < 1):
        raise ValueError("intensity_floor must be in [0,1)")
    linear = floor + (1.0 - floor) * normalized
    linear = np.clip(linear, floor, 1.0).astype(np.float32)

    meta = {
        "source_path": str(path.resolve()),
        "original_shape_px": [height, width],
        "crop_box_xyxy_px": [left, top, right, bottom],
        "crop_size_px": size,
        "output_shape_px": [out_h, out_w],
        "intensity_floor": floor,
        "intensity_min": float(linear.min()),
        "intensity_max": float(linear.max()),
    }
    return linear, crop, meta

