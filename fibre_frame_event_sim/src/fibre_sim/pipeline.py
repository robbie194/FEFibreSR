from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from .aps import integrate_aps_frame
from .config import derived_parameters, output_root, resolve_project_path
from .events import generate_v2e_events, write_events_h5
from .fibre import generate_hex_core_centres, simulate_fibre_sequence
from .grin import simulate_grin_sequence
from .io_utils import ensure_dir, read_sequence_h5, write_json, write_sequence_h5
from .motion import motion_from_config
from .relay import relay_to_sensor_sequence
from .source import prepare_source
from .visualize import (
    save_aps_comparison,
    save_aps_difference,
    save_event_plots,
    save_event_segment_previews,
    save_float_image,
    save_sequence_contact_sheet,
    save_trajectory_plot,
)


def _progress(label: str):
    last_bucket = -1

    def report(done: int, total: int) -> None:
        nonlocal last_bucket
        bucket = int(done * 10 / total)
        if bucket != last_bucket or done == total:
            last_bucket = bucket
            print(f"[{label}] {done}/{total} ({done / total:.0%})", flush=True)

    return report


def prepare_source_step(cfg: dict[str, Any]) -> Path:
    folder = ensure_dir(output_root(cfg) / "00_source")
    source_cfg = cfg["source"]
    image, crop, metadata = prepare_source(
        resolve_project_path(cfg, source_cfg["path"]),
        crop_center_px=tuple(source_cfg["crop_center_px"]),
        crop_size_px=int(source_cfg["crop_size_px"]),
        output_shape_px=tuple(source_cfg["output_shape_px"]),
        intensity_floor=float(source_cfg["intensity_floor"]),
    )
    np.save(folder / "object_intensity.npy", image)
    save_float_image(folder / "object_intensity.png", image)
    crop.save(folder / "source_crop.png")
    write_json(folder / "metadata.json", metadata)
    print(f"[00_source] wrote {folder}")
    return folder / "object_intensity.npy"


def generate_motion_step(cfg: dict[str, Any]) -> Path:
    folder = ensure_dir(output_root(cfg) / "01_motion")
    motion_cfg = cfg["motion"]
    timestamps, shifts = motion_from_config(motion_cfg)
    np.savez(folder / "motion.npz", timestamps_s=timestamps, shifts_xy_um=shifts)
    with (folder / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp_s", "shift_x_um", "shift_y_um"))
        writer.writerows(zip(timestamps, shifts[:, 0], shifts[:, 1], strict=True))
    save_trajectory_plot(folder / "trajectory.png", timestamps, shifts)
    write_json(
        folder / "metadata.json",
        {
            "trajectory": str(motion_cfg.get("trajectory", "uniform")),
            "sample_count": len(timestamps),
            "interval_count": len(timestamps) - 1,
            "dt_s": float(motion_cfg["dt_s"]),
            "shift_min_xy_um": shifts.min(axis=0).tolist(),
            "shift_max_xy_um": shifts.max(axis=0).tolist(),
            "final_shift_xy_um": shifts[-1].tolist(),
        },
    )
    print(f"[01_motion] wrote {folder}")
    return folder / "motion.npz"


def generate_grin_step(cfg: dict[str, Any]) -> Path:
    root = output_root(cfg)
    folder = ensure_dir(root / "02_grin")
    object_image = np.load(root / "00_source" / "object_intensity.npy")
    motion = np.load(root / "01_motion" / "motion.npz")
    timestamps = motion["timestamps_s"]
    shifts = motion["shifts_xy_um"]
    derived = derived_parameters(cfg)
    grin_cfg = cfg["grin"]
    frames = simulate_grin_sequence(
        object_image,
        shifts,
        object_pixel_size_um=float(cfg["source"]["pixel_size_um"]),
        fibre_shape_px=derived["fibre_shape_px"],
        fibre_pixel_size_um=float(grin_cfg["fibre_grid_pixel_size_um"]),
        magnification=float(grin_cfg["magnification"]),
        sigma_um=float(grin_cfg["sigma_um"]),
        transmission=float(grin_cfg["transmission"]),
        progress=_progress("02_grin"),
    )
    path = folder / "grin_sequence.h5"
    write_sequence_h5(path, frames, timestamps, metadata={"plane": "proximal_fibre_input"})
    save_sequence_contact_sheet(folder / "contact_sheet.png", frames, timestamps, title="GRIN output")
    print(f"[02_grin] wrote {path}")
    return path


def generate_fibre_step(cfg: dict[str, Any]) -> Path:
    root = output_root(cfg)
    folder = ensure_dir(root / "03_fibre")
    grin_frames, timestamps, _ = read_sequence_h5(root / "02_grin" / "grin_sequence.h5")
    fibre_cfg = cfg["fibre"]
    centres = generate_hex_core_centres(
        float(cfg["grin"]["fibre_field_size_um"]),
        float(fibre_cfg["core_pitch_um"]),
        float(fibre_cfg["core_diameter_um"]),
    )
    frames, signals = simulate_fibre_sequence(
        grin_frames,
        centres,
        pixel_size_um=float(cfg["grin"]["fibre_grid_pixel_size_um"]),
        core_diameter_um=float(fibre_cfg["core_diameter_um"]),
        aperture_supersample=int(fibre_cfg["aperture_supersample"]),
        transmission=float(fibre_cfg["transmission"]),
        progress=_progress("03_fibre"),
    )
    path = folder / "fibre_sequence.h5"
    write_sequence_h5(
        path,
        frames,
        timestamps,
        metadata={"plane": "distal_fibre_output", "core_count": len(centres)},
        extra_datasets={"core_centres_xy_um": centres, "core_signals": signals},
    )
    save_sequence_contact_sheet(folder / "contact_sheet.png", frames, timestamps, title="MCF distal output")
    write_json(folder / "metadata.json", {"core_count": len(centres)})
    print(f"[03_fibre] wrote {path} ({len(centres)} local cores)")
    return path


def generate_sensor_step(cfg: dict[str, Any]) -> Path:
    root = output_root(cfg)
    folder = ensure_dir(root / "04_sensor")
    fibre_frames, timestamps, _ = read_sequence_h5(root / "03_fibre" / "fibre_sequence.h5")
    relay_cfg = cfg["relay"]
    derived = derived_parameters(cfg)
    frames = relay_to_sensor_sequence(
        fibre_frames,
        fibre_pixel_size_um=float(cfg["grin"]["fibre_grid_pixel_size_um"]),
        sensor_shape_px=tuple(relay_cfg["sensor_shape_px"]),
        sensor_pixel_pitch_um=float(relay_cfg["sensor_pixel_pitch_um"]),
        magnification=float(derived["relay_magnification"]),
        psf_sigma_sensor_um=float(relay_cfg["psf_sigma_sensor_um"]),
        pixel_integration_supersample=int(relay_cfg["pixel_integration_supersample"]),
        progress=_progress("04_sensor"),
    )
    metadata = {
        "plane": "DAVIS346_sensor",
        "relay_magnification": derived["relay_magnification"],
        "core_pitch_on_sensor_px": derived["fibre_pitch_on_sensor_px"],
        "core_diameter_on_sensor_px": derived["fibre_diameter_on_sensor_px"],
    }
    path = folder / "sensor_sequence.h5"
    write_sequence_h5(path, frames, timestamps, metadata=metadata)
    save_sequence_contact_sheet(folder / "contact_sheet.png", frames, timestamps, title="DAVIS irradiance")
    write_json(folder / "metadata.json", metadata)
    print(f"[04_sensor] wrote {path}")
    return path


def generate_aps_step(cfg: dict[str, Any]) -> Path:
    root = output_root(cfg)
    folder = ensure_dir(root / "05_aps")
    frames, timestamps, _ = read_sequence_h5(root / "04_sensor" / "sensor_sequence.h5")
    aps_cfg = cfg["aps"]
    aps = integrate_aps_frame(
        frames,
        timestamps,
        float(aps_cfg["exposure_start_s"]),
        float(aps_cfg["exposure_end_s"]),
    )
    np.save(folder / "aps_frame.npy", aps)
    save_float_image(folder / "aps_frame.png", aps)
    save_aps_comparison(folder / "comparison.png", frames[0], aps, frames[-1])
    save_aps_difference(
        folder / "difference_vs_middle.png",
        aps,
        frames[len(frames) // 2],
        reference_label="instantaneous frame at 12.5 ms",
    )
    write_json(
        folder / "metadata.json",
        {
            "shape_px": list(aps.shape),
            "exposure_start_s": float(aps_cfg["exposure_start_s"]),
            "exposure_end_s": float(aps_cfg["exposure_end_s"]),
            "integration": "trapezoidal_time_average",
            "sample_count": len(timestamps),
            "intensity_min": float(aps.min()),
            "intensity_max": float(aps.max()),
        },
    )
    print(f"[05_aps] wrote {folder / 'aps_frame.npy'}")
    return folder / "aps_frame.npy"


def generate_events_step(cfg: dict[str, Any]) -> Path:
    root = output_root(cfg)
    folder = ensure_dir(root / "06_events")
    frames, timestamps, _ = read_sequence_h5(root / "04_sensor" / "sensor_sequence.h5")
    event_cfg = cfg["events"]
    v2e_root = Path(cfg["_project_root"]).parent / "v2e"
    events, stats = generate_v2e_events(
        frames,
        timestamps,
        v2e_root=v2e_root,
        pos_threshold=float(event_cfg["pos_threshold"]),
        neg_threshold=float(event_cfg["neg_threshold"]),
        threshold_sigma=float(event_cfg["threshold_sigma"]),
        cutoff_hz=float(event_cfg["cutoff_hz"]),
        leak_rate_hz=float(event_cfg["leak_rate_hz"]),
        shot_noise_rate_hz=float(event_cfg["shot_noise_rate_hz"]),
        refractory_period_s=float(event_cfg["refractory_period_s"]),
        input_white_dn=float(event_cfg["input_white_dn"]),
        device=str(event_cfg["device"]),
        seed=int(cfg["project"]["seed"]),
    )
    path = folder / "events.h5"
    write_events_h5(path, events, sensor_shape_px=tuple(cfg["relay"]["sensor_shape_px"]), metadata=stats)
    on_map, off_map = save_event_plots(
        folder / "event_accumulation.png",
        folder / "event_rate.png",
        events,
        tuple(cfg["relay"]["sensor_shape_px"]),
        duration_s=float(cfg["motion"]["duration_s"]),
    )
    np.savez(folder / "event_count_maps.npz", on=on_map, off=off_map)
    segment_summaries = {
        "5ms": save_event_segment_previews(
            folder / "segments_5ms",
            folder / "event_segments_5ms.png",
            events,
            tuple(cfg["relay"]["sensor_shape_px"]),
            start_s=0.0,
            end_s=float(cfg["motion"]["duration_s"]),
            segment_width_s=0.005,
            columns=5,
        ),
        "1ms": save_event_segment_previews(
            folder / "segments_1ms",
            folder / "event_segments_1ms.png",
            events,
            tuple(cfg["relay"]["sensor_shape_px"]),
            start_s=0.0,
            end_s=float(cfg["motion"]["duration_s"]),
            segment_width_s=0.001,
            columns=5,
        ),
    }
    write_json(folder / "segment_stats.json", segment_summaries)
    write_json(folder / "stats.json", stats)
    print(f"[06_events] wrote {path} ({len(events)} events)")
    return path


def validate_outputs_step(cfg: dict[str, Any]) -> Path:
    root = output_root(cfg)
    folder = ensure_dir(root / "07_validation")
    derived = derived_parameters(cfg)
    checks: dict[str, dict[str, Any]] = {}

    source = np.load(root / "00_source" / "object_intensity.npy")
    checks["source"] = {
        "passed": source.shape == derived["source_shape_px"] and np.isfinite(source).all(),
        "shape": list(source.shape), "min": float(source.min()), "max": float(source.max()),
    }
    motion = np.load(root / "01_motion" / "motion.npz")
    expected_times, expected_shifts = motion_from_config(cfg["motion"])
    checks["motion"] = {
        "passed": (
            np.allclose(motion["timestamps_s"], expected_times)
            and np.allclose(motion["shifts_xy_um"], expected_shifts)
        ),
        "sample_count": len(motion["timestamps_s"]), "final_shift_xy_um": motion["shifts_xy_um"][-1].tolist(),
    }
    for key, relative, shape in (
        ("grin", "02_grin/grin_sequence.h5", (derived["time_samples"], *derived["fibre_shape_px"])),
        ("fibre", "03_fibre/fibre_sequence.h5", (derived["time_samples"], *derived["fibre_shape_px"])),
        ("sensor", "04_sensor/sensor_sequence.h5", (derived["time_samples"], *map(int, cfg["relay"]["sensor_shape_px"]))),
    ):
        with h5py.File(root / relative, "r") as handle:
            dataset = handle["frames"]
            finite = np.isfinite(dataset[:]).all()
            checks[key] = {
                "passed": tuple(dataset.shape) == shape and finite,
                "shape": list(dataset.shape), "min": float(np.min(dataset)), "max": float(np.max(dataset)),
            }
    aps = np.load(root / "05_aps" / "aps_frame.npy")
    checks["aps"] = {
        "passed": aps.shape == tuple(cfg["relay"]["sensor_shape_px"]) and np.isfinite(aps).all(),
        "shape": list(aps.shape), "min": float(aps.min()), "max": float(aps.max()),
    }
    with h5py.File(root / "06_events" / "events.h5", "r") as handle:
        events = handle["events_t_s_x_y_p"][:]
    height, width = map(int, cfg["relay"]["sensor_shape_px"])
    event_valid = len(events) > 0
    if len(events):
        event_valid = bool(
            np.all(np.diff(events[:, 0]) >= 0)
            and events[:, 0].min() >= 0
            and events[:, 0].max() <= float(cfg["motion"]["duration_s"]) + 1e-7
            and events[:, 1].min() >= 0 and events[:, 1].max() < width
            and events[:, 2].min() >= 0 and events[:, 2].max() < height
            and set(np.unique(events[:, 3])).issubset({-1.0, 1.0})
        )
    checks["events"] = {
        "passed": event_valid,
        "count": len(events),
        "on_count": int(np.sum(events[:, 3] > 0)) if len(events) else 0,
        "off_count": int(np.sum(events[:, 3] < 0)) if len(events) else 0,
    }
    report = {
        "all_passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "derived_parameters": {key: list(value) if isinstance(value, tuple) else value for key, value in derived.items()},
    }
    path = folder / "validation_report.json"
    write_json(path, report)
    if not report["all_passed"]:
        raise RuntimeError(f"validation failed; see {path}")
    print(f"[07_validation] all checks passed; wrote {path}")
    return path


ALL_STEPS = (
    prepare_source_step,
    generate_motion_step,
    generate_grin_step,
    generate_fibre_step,
    generate_sensor_step,
    generate_aps_step,
    generate_events_step,
    validate_outputs_step,
)


def run_all(cfg: dict[str, Any]) -> list[Path]:
    return [step(cfg) for step in ALL_STEPS]
