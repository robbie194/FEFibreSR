from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a simulation YAML file.

    Relative source/output paths are resolved against the simulation project
    directory (the parent of ``configs``), not against the caller's cwd.
    """
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration must be a mapping: {path}")

    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent.parent)
    validate_config(cfg)
    return cfg


def resolve_project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(cfg["_project_root"]) / path).resolve()


def output_root(cfg: dict[str, Any]) -> Path:
    return resolve_project_path(cfg, cfg["project"]["output_dir"])


def derived_parameters(cfg: dict[str, Any]) -> dict[str, Any]:
    source = cfg["source"]
    motion = cfg["motion"]
    grin = cfg["grin"]
    relay = cfg["relay"]

    source_h, source_w = map(int, source["output_shape_px"])
    fibre_px = float(grin["fibre_grid_pixel_size_um"])
    fibre_size = float(grin["fibre_field_size_um"])
    fibre_n = int(round(fibre_size / fibre_px))

    duration = float(motion["duration_s"])
    dt = float(motion["dt_s"])
    intervals = int(round(duration / dt))
    times = intervals + 1

    sensor_h, sensor_w = map(int, relay["sensor_shape_px"])
    sensor_pitch = float(relay["sensor_pixel_pitch_um"])
    requested_mag = relay["magnification"]
    if isinstance(requested_mag, str) and requested_mag.lower() == "fit":
        magnification = min(
            sensor_w * sensor_pitch / fibre_size,
            sensor_h * sensor_pitch / fibre_size,
        )
    else:
        magnification = float(requested_mag)

    return {
        "source_shape_px": (source_h, source_w),
        "source_field_size_um": (
            source_h * float(source["pixel_size_um"]),
            source_w * float(source["pixel_size_um"]),
        ),
        "fibre_shape_px": (fibre_n, fibre_n),
        "time_intervals": intervals,
        "time_samples": times,
        "relay_magnification": magnification,
        "fibre_pitch_on_sensor_px": (
            float(cfg["fibre"]["core_pitch_um"])
            * magnification
            / sensor_pitch
        ),
        "fibre_diameter_on_sensor_px": (
            float(cfg["fibre"]["core_diameter_um"])
            * magnification
            / sensor_pitch
        ),
    }


def validate_config(cfg: dict[str, Any]) -> None:
    required = (
        "project",
        "source",
        "motion",
        "grin",
        "fibre",
        "relay",
        "aps",
        "events",
    )
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"missing config sections: {missing}")

    source_path = resolve_project_path(cfg, cfg["source"]["path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"source image does not exist: {source_path}")

    duration = float(cfg["motion"]["duration_s"])
    dt = float(cfg["motion"]["dt_s"])
    if duration <= 0 or dt <= 0:
        raise ValueError("motion duration_s and dt_s must be positive")
    intervals = duration / dt
    if abs(intervals - round(intervals)) > 1e-9:
        raise ValueError("duration_s must be an integer multiple of dt_s")

    pixel_size = float(cfg["source"]["pixel_size_um"])
    source_h, source_w = map(int, cfg["source"]["output_shape_px"])
    source_h_um = source_h * pixel_size
    source_w_um = source_w * pixel_size
    fibre_size = float(cfg["grin"]["fibre_field_size_um"])
    vx, vy = map(abs, map(float, cfg["motion"]["velocity_um_s"]))
    required_h = fibre_size + 2 * vy * duration
    required_w = fibre_size + 2 * vx * duration
    if source_h_um + 1e-9 < required_h or source_w_um + 1e-9 < required_w:
        raise ValueError(
            "source physical field is too small for fibre field plus motion margin"
        )

    fibre_px = float(cfg["grin"]["fibre_grid_pixel_size_um"])
    if abs(fibre_size / fibre_px - round(fibre_size / fibre_px)) > 1e-9:
        raise ValueError("fibre field size must be divisible by fibre grid pixel size")

    exposure_start = float(cfg["aps"]["exposure_start_s"])
    exposure_end = float(cfg["aps"]["exposure_end_s"])
    if not (0 <= exposure_start < exposure_end <= duration + 1e-12):
        raise ValueError("APS exposure must lie inside the simulated time range")

    if float(cfg["fibre"]["core_diameter_um"]) >= float(
        cfg["fibre"]["core_pitch_um"]
    ):
        raise ValueError("core diameter must be smaller than core pitch")

