"""DAVIS recording loading and exposure-aligned sample selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dv import AedatFile


@dataclass(frozen=True)
class Recording:
    """A DAVIS stream on a microsecond timeline starting at zero."""

    timestamps_us: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray
    frames: np.ndarray
    exposure_windows_us: np.ndarray


@dataclass(frozen=True)
class ExposureSample:
    """Events and APS frames selected for one reconstruction run."""

    timestamps_us: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray
    sharp_frame: np.ndarray
    blurred_frame: np.ndarray
    frame_index: int
    frame_exposure_us: int
    event_window_us: int


def load_aedat4(path: Path) -> Recording:
    """Load events, frames, and exposure bounds from a DAVIS AEDAT4 file."""
    if not path.is_file():
        raise FileNotFoundError(f"DAVIS recording not found: {path}")

    with AedatFile(str(path)) as stream:
        # ``dv`` streams deliberately do not implement ``len``; a comprehension
        # consumes them without Python trying to preallocate via length_hint.
        packets = [packet for packet in stream["events"].numpy()]
        event_table = np.hstack(packets)
        frame_packets = [packet for packet in stream["frames"]]

    origin_us = int(event_table["timestamp"].min())
    frames = np.stack([packet.image for packet in frame_packets]).squeeze()
    exposure_windows = np.asarray(
        [
            [packet.timestamp_start_of_exposure, packet.timestamp_end_of_exposure]
            for packet in frame_packets
        ],
        dtype=np.int64,
    )
    return Recording(
        timestamps_us=event_table["timestamp"] - origin_us,
        x=event_table["x"],
        y=event_table["y"],
        polarity=event_table["polarity"],
        frames=frames,
        exposure_windows_us=exposure_windows - origin_us,
    )


def select_time_window(
    timestamps_us: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    start_us: float,
    duration_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return events in the inclusive interval ``[start, start + duration]``."""
    selected = (timestamps_us >= start_us) & (
        timestamps_us <= start_us + duration_us
    )
    return x[selected], y[selected], timestamps_us[selected], polarity[selected]


def select_exposure_sample(
    recording: Recording,
    requested_start_us: int,
    use_two_exposures: bool,
) -> ExposureSample:
    """Match the reference script's APS frame and event-window selection."""
    later_frames = np.flatnonzero(
        recording.exposure_windows_us[:, 0] > requested_start_us
    )
    if later_frames.size == 0:
        raise ValueError("requested time lies after the final APS exposure")
    frame_index = int(later_frames[0] - 1)
    if frame_index < 0:
        raise ValueError("requested time lies before the first APS exposure")

    start_us, end_us = recording.exposure_windows_us[frame_index]
    frame_exposure_us = int(end_us - start_us)
    event_window_us = frame_exposure_us
    if use_two_exposures and frame_index + 1 < len(recording.exposure_windows_us):
        event_window_us = int(
            recording.exposure_windows_us[frame_index + 1, 1] - start_us
        )
    event_window_us = max(event_window_us, frame_exposure_us)

    x, y, timestamps, polarity = select_time_window(
        recording.timestamps_us,
        recording.x,
        recording.y,
        recording.polarity,
        float(start_us),
        float(event_window_us),
    )
    timestamps = timestamps - timestamps.min()

    frame = recording.frames[frame_index]
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp = recording.frames[0]
    if sharp.ndim == 3:
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)

    return ExposureSample(
        timestamps_us=timestamps,
        x=x,
        y=y,
        polarity=polarity,
        sharp_frame=np.asarray(sharp, dtype=np.float64),
        blurred_frame=np.asarray(frame, dtype=np.float64),
        frame_index=frame_index,
        frame_exposure_us=frame_exposure_us,
        event_window_us=event_window_us,
    )
