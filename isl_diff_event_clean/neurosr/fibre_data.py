"""Load sensor observations and reduce them to one temporal channel per core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class CoreCalibration:
    """DAVIS pixels that provide clean, high-gain readout of each fibre core."""

    pixel_xy: np.ndarray  # [core, readout, (x, y)]
    gain: np.ndarray  # [core, readout]


@dataclass(frozen=True)
class FibreObservations:
    """Measurements used by the inverse problem, with no simulation truth."""

    timestamps_s: np.ndarray
    shifts_xy_um: np.ndarray
    core_centres_xy_um: np.ndarray
    core_aps: np.ndarray
    cumulative_event_change: np.ndarray
    calibration: CoreCalibration


def build_core_calibration(
    simulation_cfg: dict,
    core_centres_xy_um: np.ndarray,
    pixels_per_core: int,
) -> CoreCalibration:
    """Select non-overlapping high-response sensor pixels for every core.

    A uniform fibre input is a flat-field calibration: every core signal is
    exactly one, so the relayed sensor value at a selected pixel is its gain.
    This uses only known optical geometry, not the unknown test object.
    """
    from fibre_sim.config import derived_parameters
    from fibre_sim.fibre import simulate_fibre_sequence
    from fibre_sim.relay import relay_to_sensor_sequence

    derived = derived_parameters(simulation_cfg)
    fibre_cfg = simulation_cfg["fibre"]
    relay_cfg = simulation_cfg["relay"]
    uniform = np.ones((1, *derived["fibre_shape_px"]), dtype=np.float32)
    fibre_frame, signals = simulate_fibre_sequence(
        uniform,
        core_centres_xy_um,
        pixel_size_um=float(simulation_cfg["grin"]["fibre_grid_pixel_size_um"]),
        core_diameter_um=float(fibre_cfg["core_diameter_um"]),
        aperture_supersample=int(fibre_cfg["aperture_supersample"]),
        transmission=float(fibre_cfg["transmission"]),
    )
    if not np.allclose(signals, 1.0, atol=1e-6):
        raise RuntimeError("uniform-input core calibration did not produce unit signals")
    sensor = relay_to_sensor_sequence(
        fibre_frame,
        fibre_pixel_size_um=float(simulation_cfg["grin"]["fibre_grid_pixel_size_um"]),
        sensor_shape_px=tuple(relay_cfg["sensor_shape_px"]),
        sensor_pixel_pitch_um=float(relay_cfg["sensor_pixel_pitch_um"]),
        magnification=float(derived["relay_magnification"]),
        psf_sigma_sensor_um=float(relay_cfg["psf_sigma_sensor_um"]),
        pixel_integration_supersample=int(relay_cfg["pixel_integration_supersample"]),
    )[0]

    sensor_h, sensor_w = sensor.shape
    magnification = float(derived["relay_magnification"])
    sensor_pitch = float(relay_cfg["sensor_pixel_pitch_um"])
    centre_x = (sensor_w - 1) / 2 + core_centres_xy_um[:, 0] * magnification / sensor_pitch
    centre_y = (sensor_h - 1) / 2 + core_centres_xy_um[:, 1] * magnification / sensor_pitch

    chosen = np.empty((len(core_centres_xy_um), pixels_per_core, 2), dtype=np.int32)
    gains = np.empty((len(core_centres_xy_um), pixels_per_core), dtype=np.float32)
    occupied: set[tuple[int, int]] = set()
    spot_radius = float(derived["fibre_diameter_on_sensor_px"]) / 2
    search_radius = int(np.ceil(spot_radius + np.sqrt(0.5)))
    for core_index, (cx, cy) in enumerate(zip(centre_x, centre_y, strict=True)):
        x0, y0 = int(round(float(cx))), int(round(float(cy)))
        candidates: list[tuple[float, int, int]] = []
        for y in range(max(0, y0 - search_radius), min(sensor_h, y0 + search_radius + 1)):
            for x in range(max(0, x0 - search_radius), min(sensor_w, x0 + search_radius + 1)):
                distance = float(np.hypot(x - cx, y - cy))
                if (
                    (x, y) not in occupied
                    and distance <= spot_radius + np.sqrt(0.5)
                    and sensor[y, x] > 0
                ):
                    # Prefer gain first; distance resolves the many exactly-flat
                    # pixels without accidentally drifting toward another spot.
                    candidates.append((float(sensor[y, x]), -distance, x, y))
        candidates.sort(reverse=True)
        if len(candidates) < pixels_per_core:
            raise RuntimeError(f"core {core_index} has too few unambiguous sensor pixels")
        for readout, (gain, _negative_distance, x, y) in enumerate(
            candidates[:pixels_per_core]
        ):
            chosen[core_index, readout] = (x, y)
            gains[core_index, readout] = gain
            occupied.add((x, y))
    return CoreCalibration(chosen, gains)


def extract_core_aps(aps_frame: np.ndarray, calibration: CoreCalibration) -> np.ndarray:
    """Undo fixed sensor gain and robustly combine redundant core readouts."""
    xy = calibration.pixel_xy
    values = aps_frame[xy[..., 1], xy[..., 0]] / calibration.gain
    return np.median(values, axis=1).astype(np.float32)


def aggregate_cumulative_events(
    events: np.ndarray,
    timestamps_s: np.ndarray,
    calibration: CoreCalibration,
    sensor_shape: tuple[int, int],
    positive_threshold: float,
    negative_threshold: float,
) -> np.ndarray:
    """Return cumulative lin-log change for every selected core readout.

    Output shape is ``[time, core, readout]``. Readouts remain separate so
    their threshold quantisation errors can be robustly averaged in the loss.
    """
    height, width = sensor_shape
    lookup_core = np.full((height, width), -1, dtype=np.int32)
    lookup_readout = np.full((height, width), -1, dtype=np.int16)
    for core in range(calibration.pixel_xy.shape[0]):
        for readout, (x, y) in enumerate(calibration.pixel_xy[core]):
            lookup_core[y, x] = core
            lookup_readout[y, x] = readout

    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    core = lookup_core[y, x]
    selected = core >= 0
    selected_events = events[selected]
    core = core[selected]
    readout = lookup_readout[y[selected], x[selected]]

    # Event timestamps are stored as float32 while motion timestamps are
    # float64. Remove only their representation-scale positive drift so an
    # event generated at a frame boundary stays on that boundary.
    time_tolerance = float(np.min(np.diff(timestamps_s))) * 1e-5
    event_times = selected_events[:, 0].astype(np.float64) - time_tolerance
    time_index = np.searchsorted(timestamps_s, event_times, side="left")
    time_index = np.clip(time_index, 0, len(timestamps_s) - 1)
    change = np.where(
        selected_events[:, 3] > 0,
        float(positive_threshold),
        -float(negative_threshold),
    ).astype(np.float32)
    increments = np.zeros(
        (len(timestamps_s), calibration.pixel_xy.shape[0], calibration.pixel_xy.shape[1]),
        dtype=np.float32,
    )
    np.add.at(increments, (time_index, core, readout), change)
    return np.cumsum(increments, axis=0)


def load_fibre_observations(
    simulation_cfg: dict,
    output_root: Path,
    pixels_per_core: int,
) -> FibreObservations:
    """Load APS/events/motion and construct the core-domain inverse data."""
    from fibre_sim.fibre import generate_hex_core_centres

    motion = np.load(output_root / "01_motion" / "motion.npz")
    timestamps = motion["timestamps_s"].astype(np.float64)
    shifts = motion["shifts_xy_um"].astype(np.float32)
    fibre_cfg = simulation_cfg["fibre"]
    centres = generate_hex_core_centres(
        float(simulation_cfg["grin"]["fibre_field_size_um"]),
        float(fibre_cfg["core_pitch_um"]),
        float(fibre_cfg["core_diameter_um"]),
    )
    calibration = build_core_calibration(simulation_cfg, centres, pixels_per_core)
    aps = np.load(output_root / "05_aps" / "aps_frame.npy").astype(np.float32)
    core_aps = extract_core_aps(aps, calibration)
    with h5py.File(output_root / "06_events" / "events.h5", "r") as handle:
        events = handle["events_t_s_x_y_p"][:]
    event_cfg = simulation_cfg["events"]
    cumulative = aggregate_cumulative_events(
        events,
        timestamps,
        calibration,
        tuple(simulation_cfg["relay"]["sensor_shape_px"]),
        float(event_cfg["pos_threshold"]),
        float(event_cfg["neg_threshold"]),
    )
    return FibreObservations(
        timestamps,
        shifts,
        centres,
        core_aps,
        cumulative,
        calibration,
    )
