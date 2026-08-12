from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=_numpy_json_default,
        )


def _numpy_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_sequence_h5(
    path: str | Path,
    frames: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    dataset_name: str = "frames",
    metadata: dict[str, Any] | None = None,
    extra_datasets: dict[str, np.ndarray] | None = None,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    frames = np.asarray(frames, dtype=np.float32)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if frames.ndim != 3:
        raise ValueError(f"frames must have shape [T,H,W], got {frames.shape}")
    if len(frames) != len(timestamps_s):
        raise ValueError("frame and timestamp counts differ")

    with h5py.File(target, "w") as handle:
        handle.create_dataset(
            dataset_name,
            data=frames,
            chunks=(1, frames.shape[1], frames.shape[2]),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        handle.create_dataset("timestamps_s", data=timestamps_s)
        if extra_datasets:
            for name, array in extra_datasets.items():
                value = np.asarray(array)
                kwargs: dict[str, Any] = {}
                if value.ndim > 0 and value.size > 1024:
                    kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
                handle.create_dataset(name, data=value, **kwargs)
        if metadata:
            handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)


def read_sequence_h5(
    path: str | Path, dataset_name: str = "frames"
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        frames = handle[dataset_name][:]
        timestamps = handle["timestamps_s"][:]
        raw_meta = handle.attrs.get("metadata_json", "{}")
        metadata = json.loads(raw_meta)
    return frames, timestamps, metadata


def h5_dataset(path: str | Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle[name][:]
