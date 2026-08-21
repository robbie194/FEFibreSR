"""Continuous trajectory bases shared by blind motion estimators."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def cubic_bspline_basis(times: np.ndarray, control_count: int) -> np.ndarray:
    """Evaluate a clamped uniform cubic B-spline basis on ``[0, 1]``."""
    degree = 3
    if control_count <= degree:
        raise ValueError("a cubic B-spline needs at least four control points")
    normalized_times = np.asarray(times, dtype=np.float64)
    if np.any((normalized_times < 0) | (normalized_times > 1)):
        raise ValueError("B-spline sample times must lie in [0, 1]")
    internal_count = control_count - degree - 1
    internal_knots = np.linspace(0, 1, internal_count + 2)[1:-1]
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            internal_knots,
            np.ones(degree + 1),
        )
    )
    basis = BSpline.design_matrix(
        normalized_times, knots, degree, extrapolate=False
    ).toarray()
    return basis.astype(np.float32)


def fit_cubic_bspline(
    sample_positions_xy: np.ndarray,
    control_count: int,
) -> np.ndarray:
    """Fit controls while preserving the first and final positions exactly."""
    positions = np.asarray(sample_positions_xy, dtype=np.float64)
    times = np.linspace(0, 1, len(positions))
    basis = cubic_bspline_basis(times, control_count).astype(np.float64)
    controls = np.empty((control_count, 2), dtype=np.float64)
    controls[0] = positions[0]
    controls[-1] = positions[-1]
    residual = positions - basis[:, :1] * controls[0] - basis[:, -1:] * controls[-1]
    controls[1:-1] = np.linalg.lstsq(basis[:, 1:-1], residual, rcond=None)[0]
    return controls.astype(np.float32)


def sample_cubic_bspline(
    control_positions_xy: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Sample B-spline controls at uniformly spaced normalized times."""
    controls = np.asarray(control_positions_xy, dtype=np.float32)
    times = np.linspace(0, 1, sample_count, dtype=np.float32)
    return cubic_bspline_basis(times, len(controls)) @ controls
