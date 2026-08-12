from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .io_utils import ensure_dir


def _import_event_emulator(v2e_root: str | Path):
    root = Path(v2e_root).resolve()
    if not (root / "v2ecore" / "emulator.py").is_file():
        raise FileNotFoundError(f"v2e checkout not found at {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from v2ecore.emulator import EventEmulator

    return EventEmulator


def generate_v2e_events(
    sensor_frames: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    v2e_root: str | Path,
    pos_threshold: float,
    neg_threshold: float,
    threshold_sigma: float = 0.0,
    cutoff_hz: float = 0.0,
    leak_rate_hz: float = 0.0,
    shot_noise_rate_hz: float = 0.0,
    refractory_period_s: float = 0.0,
    input_white_dn: float = 255.0,
    device: str = "cuda",
    seed: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate ideal/non-ideal DVS events from high-rate intensity frames.

    The returned columns are ``[t_s, x, y, polarity]`` where polarity is
    +1 for ON and -1 for OFF. Frames stay float32 and are merely scaled to
    v2e's 0--255 linear intensity convention.
    """
    frames = np.asarray(sensor_frames, dtype=np.float32)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if frames.ndim != 3 or len(frames) != len(times):
        raise ValueError("expected sensor_frames [T,H,W] and matching timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

    EventEmulator = _import_event_emulator(v2e_root)
    height, width = frames.shape[1:]
    emulator = EventEmulator(
        pos_thres=float(pos_threshold),
        neg_thres=float(neg_threshold),
        sigma_thres=float(threshold_sigma),
        cutoff_hz=float(cutoff_hz),
        leak_rate_hz=float(leak_rate_hz),
        refractory_period_s=float(refractory_period_s),
        shot_noise_rate_hz=float(shot_noise_rate_hz),
        photoreceptor_noise=False,
        seed=int(seed),
        output_width=width,
        output_height=height,
        device=device,
    )
    logger = logging.getLogger("v2ecore.emulator")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    batches: list[np.ndarray] = []
    try:
        scaled = np.clip(frames, 0.0, 1.0).astype(np.float32) * float(input_white_dn)
        for frame, timestamp in zip(scaled, times, strict=True):
            events = emulator.generate_events(frame, float(timestamp))
            if events is not None and len(events):
                batches.append(np.asarray(events, dtype=np.float32))
    finally:
        logger.setLevel(previous_level)
        cleanup = getattr(emulator, "cleanup", None)
        if callable(cleanup):
            cleanup()
    events = np.concatenate(batches, axis=0) if batches else np.empty((0, 4), np.float32)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    stats = {
        "count": int(len(events)),
        "on_count": int(np.sum(events[:, 3] > 0)) if len(events) else 0,
        "off_count": int(np.sum(events[:, 3] < 0)) if len(events) else 0,
        "first_timestamp_s": float(events[0, 0]) if len(events) else None,
        "last_timestamp_s": float(events[-1, 0]) if len(events) else None,
        "device_used": device,
        "coordinate_order": "t_s,x,y,polarity",
    }
    return events, stats


def write_events_h5(
    path: str | Path,
    events: np.ndarray,
    *,
    sensor_shape_px: tuple[int, int],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write float-second and conventional uint32-microsecond event tables."""
    target = Path(path)
    ensure_dir(target.parent)
    value = np.asarray(events, dtype=np.float32).reshape(-1, 4)
    events_us = value.copy()
    if len(events_us):
        events_us[:, 0] *= 1e6
        events_us[:, 3] = (events_us[:, 3] > 0).astype(np.float32)
    with h5py.File(target, "w") as handle:
        handle.create_dataset(
            "events_t_s_x_y_p",
            data=value,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        handle.create_dataset(
            "events_t_us_x_y_p01",
            data=events_us.astype(np.uint32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        handle.attrs["sensor_height_px"] = int(sensor_shape_px[0])
        handle.attrs["sensor_width_px"] = int(sensor_shape_px[1])
        if metadata:
            import json

            handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

