from __future__ import annotations

import numpy as np


def integrate_aps_frame(
    sensor_frames: np.ndarray,
    timestamps_s: np.ndarray,
    exposure_start_s: float,
    exposure_end_s: float,
) -> np.ndarray:
    """Return exposure-time average using endpoint-aware trapezoidal integration."""
    frames = np.asarray(sensor_frames, dtype=np.float32)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if frames.ndim != 3 or times.ndim != 1 or len(frames) != len(times):
        raise ValueError("expected sensor_frames [T,H,W] and matching timestamps [T]")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    start, end = float(exposure_start_s), float(exposure_end_s)
    if not (times[0] <= start < end <= times[-1]):
        raise ValueError("exposure lies outside sequence")

    inside = times[(times > start) & (times < end)]
    sample_times = np.concatenate(([start], inside, [end]))
    sample_frames: list[np.ndarray] = []
    for time in sample_times:
        exact = np.flatnonzero(np.isclose(times, time, rtol=0, atol=1e-12))
        if exact.size:
            sample_frames.append(frames[exact[0]])
            continue
        right = int(np.searchsorted(times, time))
        left = right - 1
        alpha = (time - times[left]) / (times[right] - times[left])
        sample_frames.append((1 - alpha) * frames[left] + alpha * frames[right])
    stack = np.stack(sample_frames, axis=0)
    integrated = np.trapz(stack, x=sample_times, axis=0) / (end - start)
    return np.asarray(integrated, dtype=np.float32)

