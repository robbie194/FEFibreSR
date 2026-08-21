"""Convert raw sensor observations into calibrated core-domain measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

from .geometry import centres_from_mask
from .io import load_core_mask, load_recording


@dataclass(frozen=True)
class CoreObservations:
    sensor_shape: tuple[int, int]
    exposure_start_s: float
    exposure_end_s: float
    centres_xy: np.ndarray
    core_aps: np.ndarray
    event_xy: np.ndarray
    event_time_normalized: np.ndarray
    event_polarity: np.ndarray
    event_core_index: np.ndarray


def load_core_observations(observations_dir: Path) -> CoreObservations:
    """Load only the two files allowed at reconstruction time."""
    mask = load_core_mask(observations_dir / "core_mask.npz")
    recording = load_recording(observations_dir / "recording.h5")
    if recording.sensor_shape != mask.labels.shape:
        raise ValueError("recording and core mask shapes differ")
    centres = centres_from_mask(mask.labels)
    background = float(np.median(recording.aps_frame[mask.labels == 0]))
    core_aps = np.empty(len(centres), dtype=np.float32)
    for core in range(len(centres)):
        selected = mask.labels == core + 1
        corrected = (recording.aps_frame[selected] - background) / np.clip(
            mask.flat_response[selected], 0.1, None
        )
        core_aps[core] = np.median(corrected)
    core_aps = np.clip(core_aps, 0, None)

    events = recording.events
    x = np.rint(events[:, 1]).astype(np.int32)
    y = np.rint(events[:, 2]).astype(np.int32)
    valid_sensor = (
        (x >= 0)
        & (x < recording.sensor_shape[1])
        & (y >= 0)
        & (y < recording.sensor_shape[0])
    )
    labels = np.zeros(len(events), dtype=np.int32)
    labels[valid_sensor] = mask.labels[y[valid_sensor], x[valid_sensor]]
    valid = labels > 0
    core_index = labels[valid] - 1
    duration = recording.exposure_end_s - recording.exposure_start_s
    normalized_time = np.clip(
        (events[valid, 0] - recording.exposure_start_s) / duration, 0, 1
    )
    return CoreObservations(
        recording.sensor_shape,
        recording.exposure_start_s,
        recording.exposure_end_s,
        centres,
        core_aps,
        centres[core_index],
        normalized_time.astype(np.float32),
        events[valid, 3].astype(np.float32),
        core_index.astype(np.int32),
    )


def interpolate_core_aps(observations: CoreObservations) -> np.ndarray:
    """Create the long-exposure, honeycomb-free initialization."""
    height, width = observations.sensor_shape
    y, x = np.mgrid[:height, :width]
    linear = griddata(
        observations.centres_xy,
        observations.core_aps,
        (x, y),
        method="linear",
        fill_value=np.nan,
    )
    nearest = griddata(
        observations.centres_xy,
        observations.core_aps,
        (x, y),
        method="nearest",
    )
    return np.where(np.isfinite(linear), linear, nearest).astype(np.float32)
