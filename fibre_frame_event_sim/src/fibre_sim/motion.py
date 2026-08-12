from __future__ import annotations

import numpy as np


def uniform_motion(
    duration_s: float,
    dt_s: float,
    velocity_um_s: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate endpoint-inclusive uniform motion samples.

    Returns timestamps ``[T]`` and shifts ``[T,2]`` in (x, y) micrometres.
    """
    intervals = int(round(float(duration_s) / float(dt_s)))
    if intervals <= 0:
        raise ValueError("duration must contain at least one time interval")
    if abs(intervals * float(dt_s) - float(duration_s)) > 1e-12:
        raise ValueError("duration_s must be an integer multiple of dt_s")
    timestamps = np.linspace(0.0, float(duration_s), intervals + 1, dtype=np.float64)
    velocity = np.asarray(velocity_um_s, dtype=np.float64)
    if velocity.shape != (2,):
        raise ValueError("velocity_um_s must have two values (vx, vy)")
    shifts = timestamps[:, None] * velocity[None, :]
    return timestamps, shifts.astype(np.float32)

