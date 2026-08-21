"""Generate raw fibre APS/events while keeping truth outside the inverse input."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import ExperimentConfig
from .geometry import (
    core_pixel_lists,
    generate_irregular_core_mask,
    hexagonal_centres,
)
from .io import CoreMask, Recording, save_core_mask, save_recording, write_json
from .render import sample_image_at_points, shift_image


def _prepare_object(config: ExperimentConfig) -> np.ndarray:
    source = config.values["source"]
    path = Path(source["path"])
    if not path.is_absolute():
        path = (config.project_root / path).resolve()
    image = Image.open(path).convert("L")
    center_x, center_y = map(int, source["crop_center_px"])
    size = int(source["crop_size_px"])
    crop = image.crop(
        (
            center_x - size // 2,
            center_y - size // 2,
            center_x + size // 2,
            center_y + size // 2,
        )
    )
    height, width = map(int, config.values["sensor"]["shape_px"])
    resized = crop.resize((width, height), Image.Resampling.LANCZOS)
    normalized = np.asarray(resized, dtype=np.float32) / 255.0
    floor = float(source["intensity_floor"])
    return np.clip(floor + (1 - floor) * normalized, floor, 1).astype(np.float32)


def _motion(times: np.ndarray, motion_cfg: dict) -> np.ndarray:
    """A smooth, non-constant trajectory that the inverse never reads."""
    u = (times - times[0]) / (times[-1] - times[0])
    end_x, end_y = map(float, motion_cfg["end_shift_px"])
    curvature = float(motion_cfg["curvature_px"])
    easing = u + 0.10 * np.sin(2 * np.pi * u) / (2 * np.pi)
    x = end_x * easing
    y = end_y * easing + curvature * np.sin(np.pi * u) ** 2
    return np.column_stack((x, y)).astype(np.float32)


def _effective_sequence(
    object_image: np.ndarray,
    centres_xy: np.ndarray,
    shifts_xy: np.ndarray,
    sigma_px: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    image = torch.as_tensor(object_image, device=device)
    if sigma_px > 0:
        radius = max(1, int(round(3 * sigma_px)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=image.dtype)
        kernel_1d = torch.exp(-0.5 * (coords / sigma_px).square())
        kernel_1d /= kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        image = torch.nn.functional.conv2d(
            image[None, None], kernel[None, None], padding=radius
        )[0, 0]
    centres = torch.as_tensor(centres_xy, device=device)
    signals: list[np.ndarray] = []
    for shift in torch.as_tensor(shifts_xy, device=device):
        signals.append(sample_image_at_points(image, centres - shift).cpu().numpy())
    return image.cpu().numpy().astype(np.float32), np.asarray(signals, dtype=np.float32)


def _generate_core_events(
    core_signals: np.ndarray,
    timestamps_s: np.ndarray,
    mask: CoreMask,
    threshold: float,
    threshold_jitter: float,
    noise_fraction: float,
    seed: int,
) -> np.ndarray:
    """Generate core-level threshold crossings, then scatter them inside spots."""
    rng = np.random.default_rng(seed)
    core_count = core_signals.shape[1]
    thresholds = threshold * rng.lognormal(0.0, threshold_jitter, core_count)
    log_signal = np.log(np.clip(core_signals, 1e-4, None))
    reference = log_signal[0].copy()
    pixels = core_pixel_lists(mask)
    pixel_probabilities = []
    for core_index, candidates in enumerate(pixels, start=1):
        gain = mask.flat_response[candidates[:, 1], candidates[:, 0]].astype(np.float64)
        pixel_probabilities.append(gain / gain.sum())

    batches: list[np.ndarray] = []
    for time_index in range(1, len(timestamps_s)):
        delta = log_signal[time_index] - reference
        positive = np.floor(np.maximum(delta, 0) / thresholds).astype(np.int32)
        negative = np.floor(np.maximum(-delta, 0) / thresholds).astype(np.int32)
        active = np.flatnonzero((positive + negative) > 0)
        rows: list[tuple[float, float, float, float]] = []
        for core in active:
            polarity = 1 if positive[core] else -1
            count = int(positive[core] + negative[core])
            candidates = pixels[core]
            selected = rng.choice(
                len(candidates), size=count, replace=True, p=pixel_probabilities[core]
            )
            fractions = (np.arange(count, dtype=np.float64) + 1) / (count + 1)
            times = timestamps_s[time_index - 1] + fractions * (
                timestamps_s[time_index] - timestamps_s[time_index - 1]
            )
            for event_time, pixel_index in zip(times, selected, strict=True):
                x, y = candidates[pixel_index]
                rows.append((event_time, float(x), float(y), float(polarity)))
            reference[core] += polarity * count * thresholds[core]
        if rows:
            batches.append(np.asarray(rows, dtype=np.float32))

    events = np.concatenate(batches) if batches else np.empty((0, 4), np.float32)
    noise_count = int(round(len(events) * noise_fraction))
    if noise_count:
        height, width = mask.labels.shape
        noise = np.column_stack(
            (
                rng.uniform(timestamps_s[0], timestamps_s[-1], noise_count),
                rng.integers(0, width, noise_count),
                rng.integers(0, height, noise_count),
                rng.choice((-1, 1), noise_count),
            )
        ).astype(np.float32)
        events = np.concatenate((events, noise))
    return events[np.argsort(events[:, 0], kind="stable")]


def _sensor_aps(
    core_aps: np.ndarray, mask: CoreMask, noise_sigma: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = mask.labels
    frame = np.full(labels.shape, 0.015, dtype=np.float32)
    foreground = labels > 0
    frame[foreground] += (
        core_aps[labels[foreground] - 1] * mask.flat_response[foreground]
    )
    frame += rng.normal(0, noise_sigma, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 1).astype(np.float32)


def generate_observations(config: ExperimentConfig, device: torch.device) -> dict:
    """Write public observations and isolated private truth."""
    values = config.values
    project_cfg = values["project"]
    sensor_shape = tuple(map(int, values["sensor"]["shape_px"]))
    seed = int(project_cfg["seed"])
    object_image = _prepare_object(config)
    centres = hexagonal_centres(
        sensor_shape,
        float(values["fibre"]["core_pitch_px"]),
        float(values["fibre"]["margin_px"]),
    )
    mask = generate_irregular_core_mask(
        sensor_shape,
        centres,
        float(values["fibre"]["proximal_spot_radius_px"]),
        seed,
    )
    duration = float(values["recording"]["duration_s"])
    dt = float(values["recording"]["sample_interval_s"])
    timestamps = np.linspace(0, duration, int(round(duration / dt)) + 1)
    shifts = _motion(timestamps, values["motion"])
    effective_object, core_signals = _effective_sequence(
        object_image,
        centres,
        shifts,
        float(values["simulation_only"]["effective_blur_sigma_px"]),
        device,
    )
    core_aps = np.trapz(core_signals, timestamps, axis=0) / duration
    aps_frame = _sensor_aps(
        core_aps,
        mask,
        float(values["simulation_only"]["aps_noise_sigma"]),
        seed + 1,
    )
    event_cfg = values["simulation_only"]["events"]
    events = _generate_core_events(
        core_signals,
        timestamps,
        mask,
        float(event_cfg["contrast_threshold"]),
        float(event_cfg["threshold_log_sigma"]),
        float(event_cfg["background_noise_fraction"]),
        seed + 2,
    )
    recording = Recording(aps_frame, events, 0.0, duration, sensor_shape)
    config.observations_dir.mkdir(parents=True, exist_ok=True)
    save_core_mask(config.observations_dir / "core_mask.npz", mask)
    save_recording(config.observations_dir / "recording.h5", recording)
    write_json(
        config.observations_dir / "README.json",
        {
            "reconstruction_contract": [
                "core_mask.npz: measured pixel-to-core labels and flat-field response",
                "recording.h5: raw APS, raw events, exposure timestamps",
            ],
            "forbidden_inverse_inputs": ["object", "motion", "PSF", "event thresholds"],
            "event_columns": ["timestamp_s", "sensor_x", "sensor_y", "polarity"],
        },
    )

    reference_index = len(timestamps) // 2
    reference_truth = shift_image(
        torch.as_tensor(effective_object), torch.as_tensor(shifts[reference_index])
    ).numpy()
    config.private_truth_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.private_truth_dir / "object_input.npy", object_image)
    np.save(config.private_truth_dir / "object_effective_reference.npy", reference_truth)
    np.savez(
        config.private_truth_dir / "motion_truth.npz",
        timestamps_s=timestamps,
        shifts_xy_px=shifts,
    )
    np.save(config.private_truth_dir / "core_signals.npy", core_signals)
    write_json(
        config.private_truth_dir / "README.json",
        {"warning": "Evaluation only. Reconstruction must never read this directory."},
    )
    return {
        "core_count": int(len(centres)),
        "event_count": int(len(events)),
        "sensor_shape": list(sensor_shape),
        "duration_s": duration,
    }
