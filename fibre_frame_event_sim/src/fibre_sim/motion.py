from __future__ import annotations

from typing import Any

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


def piecewise_linear_motion(
    duration_s: float,
    dt_s: float,
    waypoint_times_s: list[float] | np.ndarray,
    waypoints_xy_um: list[list[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate an endpoint-inclusive trajectory through 2-D waypoints."""
    intervals = int(round(float(duration_s) / float(dt_s)))
    if intervals <= 0 or abs(intervals * float(dt_s) - float(duration_s)) > 1e-12:
        raise ValueError("duration_s must contain an integer number of time steps")

    waypoint_times = np.asarray(waypoint_times_s, dtype=np.float64)
    waypoints = np.asarray(waypoints_xy_um, dtype=np.float64)
    if waypoint_times.ndim != 1 or len(waypoint_times) < 2:
        raise ValueError("piecewise motion needs at least two waypoint times")
    if waypoints.shape != (len(waypoint_times), 2):
        raise ValueError("waypoints_xy_um must have shape [waypoint, 2]")
    if not np.all(np.diff(waypoint_times) > 0):
        raise ValueError("waypoint_times_s must be strictly increasing")
    if not np.isclose(waypoint_times[0], 0.0, atol=1e-12):
        raise ValueError("the first waypoint time must be zero")
    if not np.isclose(waypoint_times[-1], duration_s, atol=1e-12):
        raise ValueError("the last waypoint time must equal duration_s")

    timestamps = np.linspace(0.0, float(duration_s), intervals + 1, dtype=np.float64)
    shifts = np.column_stack(
        [np.interp(timestamps, waypoint_times, waypoints[:, axis]) for axis in range(2)]
    )
    return timestamps, shifts.astype(np.float32)


def motion_from_config(motion_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build either a legacy uniform or a general piecewise-linear trajectory."""
    trajectory = str(motion_cfg.get("trajectory", "uniform")).lower()
    duration = float(motion_cfg["duration_s"])
    dt = float(motion_cfg["dt_s"])
    if trajectory == "uniform":
        return uniform_motion(duration, dt, tuple(motion_cfg["velocity_um_s"]))
    if trajectory == "piecewise_linear":
        return piecewise_linear_motion(
            duration,
            dt,
            motion_cfg["waypoint_times_s"],
            motion_cfg["waypoints_xy_um"],
        )
    raise ValueError(f"unsupported motion trajectory: {trajectory}")
