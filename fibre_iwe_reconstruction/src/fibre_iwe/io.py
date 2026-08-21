"""Small, explicit file formats at the simulation/reconstruction boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class CoreMask:
    """Measured sensor-pixel assignment and flat-field response for each core."""

    labels: np.ndarray
    flat_response: np.ndarray


@dataclass(frozen=True)
class Recording:
    """Only measurements available to a future real-data reconstruction."""

    aps_frame: np.ndarray
    events: np.ndarray
    exposure_start_s: float
    exposure_end_s: float
    sensor_shape: tuple[int, int]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_core_mask(path: Path, mask: CoreMask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(mask.labels, dtype=np.int32),
        flat_response=np.asarray(mask.flat_response, dtype=np.float32),
    )


def load_core_mask(path: Path) -> CoreMask:
    with np.load(path) as values:
        labels = values["labels"].astype(np.int32)
        response = values["flat_response"].astype(np.float32)
    if labels.shape != response.shape or labels.ndim != 2:
        raise ValueError("core mask arrays must have the same 2-D shape")
    if labels.min() < 0:
        raise ValueError("core labels must use 0 for background and positive core IDs")
    return CoreMask(labels, response)


def save_recording(path: Path, recording: Recording) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("aps_frame", data=recording.aps_frame, compression="gzip")
        handle.create_dataset(
            "events_t_s_x_y_p",
            data=recording.events.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
        handle.attrs["exposure_start_s"] = recording.exposure_start_s
        handle.attrs["exposure_end_s"] = recording.exposure_end_s
        handle.attrs["sensor_height"] = recording.sensor_shape[0]
        handle.attrs["sensor_width"] = recording.sensor_shape[1]


def load_recording(path: Path) -> Recording:
    with h5py.File(path, "r") as handle:
        aps = handle["aps_frame"][:].astype(np.float32)
        events = handle["events_t_s_x_y_p"][:].astype(np.float32)
        start = float(handle.attrs["exposure_start_s"])
        end = float(handle.attrs["exposure_end_s"])
        shape = (int(handle.attrs["sensor_height"]), int(handle.attrs["sensor_width"]))
    if aps.shape != shape or events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("invalid recording shape")
    if len(events) and np.any(np.diff(events[:, 0]) < 0):
        raise ValueError("events must be timestamp sorted")
    return Recording(aps, events, start, end, shape)
