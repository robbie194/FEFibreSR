from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io_utils import ensure_dir


def save_float_image(
    path: str | Path,
    image: np.ndarray,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    scaled = np.clip((np.asarray(image) - vmin) / (vmax - vmin), 0, 1)
    # PNGs are diagnostic previews; keep them 8-bit for broad Typora/browser
    # compatibility. Numerical outputs remain float32 in NPY/HDF5 files.
    Image.fromarray(np.round(scaled * 255).astype(np.uint8), mode="L").save(target)


def save_sequence_contact_sheet(
    path: str | Path,
    frames: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    values = np.asarray(frames)
    indices = [0, len(values) // 2, len(values) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, index in zip(axes, indices, strict=True):
        axis.imshow(values[index], cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(f"t = {timestamps_s[index] * 1e3:.1f} ms")
        axis.axis("off")
    fig.suptitle(title)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def save_trajectory_plot(
    path: str | Path, timestamps_s: np.ndarray, shifts_um: np.ndarray
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(timestamps_s * 1e3, shifts_um[:, 0], label="x")
    axis.plot(timestamps_s * 1e3, shifts_um[:, 1], label="y")
    axis.set(xlabel="time (ms)", ylabel="object shift (µm)", title="Uniform object motion")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def event_count_maps(
    events: np.ndarray, sensor_shape_px: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, sensor_shape_px)
    on = np.zeros((height, width), dtype=np.uint32)
    off = np.zeros((height, width), dtype=np.uint32)
    value = np.asarray(events)
    if len(value):
        x = value[:, 1].astype(np.int64)
        y = value[:, 2].astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.add.at(on, (y[valid & (value[:, 3] > 0)], x[valid & (value[:, 3] > 0)]), 1)
        np.add.at(off, (y[valid & (value[:, 3] < 0)], x[valid & (value[:, 3] < 0)]), 1)
    return on, off


def _render_event_count_maps(
    on: np.ndarray,
    off: np.ndarray,
    *,
    log_scale: float | None = None,
) -> np.ndarray:
    magnitude = np.log1p(on + off).astype(np.float32)
    scale = log_scale
    if scale is None:
        scale = float(magnitude.max()) if magnitude.size and magnitude.max() > 0 else 1.0
    scale = max(float(scale), np.finfo(np.float32).eps)
    rgb = np.ones((*on.shape, 3), dtype=np.float32)
    rgb[..., 1] -= magnitude / scale
    rgb[..., 2] -= np.log1p(on) / scale
    rgb[..., 0] -= np.log1p(off) / scale
    return np.clip(rgb, 0, 1)


def event_accumulation_image(
    events: np.ndarray, sensor_shape_px: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    on, off = event_count_maps(events, sensor_shape_px)
    return _render_event_count_maps(on, off), on, off


def split_events_by_time(
    events: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    segment_width_s: float,
) -> list[tuple[float, float, np.ndarray]]:
    """Split events into non-overlapping half-open time windows.

    The final window includes its right endpoint, so every event within the
    requested interval appears exactly once.
    """
    value = np.asarray(events)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("events must have shape [N,4]")
    if not (start_s < end_s and segment_width_s > 0):
        raise ValueError("invalid event segmentation interval")
    edges = np.arange(start_s, end_s, segment_width_s, dtype=np.float64)
    edges = np.append(edges, end_s)
    segments: list[tuple[float, float, np.ndarray]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        right_test = value[:, 0] <= right + 1e-10 if index == len(edges) - 2 else value[:, 0] < right
        mask = (value[:, 0] >= left) & right_test
        segments.append((float(left), float(right), value[mask]))
    return segments


def save_event_segment_previews(
    output_dir: str | Path,
    contact_sheet_path: str | Path,
    events: np.ndarray,
    sensor_shape_px: tuple[int, int],
    *,
    start_s: float,
    end_s: float,
    segment_width_s: float,
    columns: int = 5,
) -> list[dict[str, int | float | str]]:
    """Save individual and contact-sheet event images for time windows."""
    folder = ensure_dir(output_dir)
    segments = split_events_by_time(
        events, start_s=start_s, end_s=end_s, segment_width_s=segment_width_s
    )
    maps = [event_count_maps(segment, sensor_shape_px) for _, _, segment in segments]
    common_scale = max(
        (float(np.log1p(on + off).max()) for on, off in maps),
        default=1.0,
    )
    rows = int(np.ceil(len(segments) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.3 * columns, 2.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    summaries: list[dict[str, int | float | str]] = []
    for index, ((left, right, segment), (on, off)) in enumerate(zip(segments, maps, strict=True)):
        rgb = _render_event_count_maps(on, off, log_scale=common_scale)
        left_ms, right_ms = left * 1e3, right * 1e3
        filename = f"segment_{index:02d}_{left_ms:06.1f}_{right_ms:06.1f}ms.png"
        Image.fromarray(np.round(rgb * 255).astype(np.uint8), mode="RGB").save(folder / filename)
        axis = axes.flat[index]
        axis.imshow(rgb)
        axis.set_title(
            f"{left_ms:g}–{right_ms:g} ms\n"
            f"ON {int(on.sum()):,} / OFF {int(off.sum()):,}",
            fontsize=9,
        )
        axis.axis("off")
        summaries.append(
            {
                "start_s": left,
                "end_s": right,
                "count": int(len(segment)),
                "on_count": int(on.sum()),
                "off_count": int(off.sum()),
                "preview": str(folder / filename),
            }
        )
    for axis in axes.flat[len(segments) :]:
        axis.axis("off")
    width_ms = segment_width_s * 1e3
    fig.suptitle(
        f"Event stream split into {width_ms:g} ms windows "
        "(ON red, OFF blue; common log scale)"
    )
    contact = Path(contact_sheet_path)
    ensure_dir(contact.parent)
    fig.savefig(contact, dpi=160)
    plt.close(fig)
    return summaries


def save_event_plots(
    accumulation_path: str | Path,
    rate_path: str | Path,
    events: np.ndarray,
    sensor_shape_px: tuple[int, int],
    *,
    duration_s: float,
    bin_width_s: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, on, off = event_accumulation_image(events, sensor_shape_px)
    target = Path(accumulation_path)
    ensure_dir(target.parent)
    Image.fromarray(np.round(rgb * 255).astype(np.uint8), mode="RGB").save(target)

    bins = np.arange(0, duration_s + bin_width_s * 1.001, bin_width_s)
    if bins[-1] < duration_s:
        bins = np.append(bins, duration_s)
    value = np.asarray(events)
    on_times = value[value[:, 3] > 0, 0] if len(value) else np.empty(0)
    off_times = value[value[:, 3] < 0, 0] if len(value) else np.empty(0)
    on_hist, edges = np.histogram(on_times, bins=bins)
    off_hist, _ = np.histogram(off_times, bins=bins)
    centres_ms = (edges[:-1] + edges[1:]) * 500
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.step(centres_ms, on_hist, where="mid", color="red", label="ON")
    axis.step(centres_ms, off_hist, where="mid", color="blue", label="OFF")
    axis.set(xlabel="time (ms)", ylabel=f"events / {bin_width_s * 1e3:g} ms", title="Event rate")
    axis.grid(True, alpha=0.3)
    axis.legend()
    rate_target = Path(rate_path)
    ensure_dir(rate_target.parent)
    fig.savefig(rate_target, dpi=160)
    plt.close(fig)
    return on, off


def save_aps_comparison(
    path: str | Path, first: np.ndarray, aps: np.ndarray, last: np.ndarray
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, image, label in zip(
        axes, (first, aps, last), ("instantaneous 0 ms", "APS 25 ms average", "instantaneous 25 ms"), strict=True
    ):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(label)
        axis.axis("off")
    fig.savefig(target, dpi=160)
    plt.close(fig)


def save_aps_difference(
    path: str | Path, aps: np.ndarray, reference: np.ndarray, *, reference_label: str
) -> None:
    """Save an autoscaled signed difference that exposes subtle APS blur."""
    target = Path(path)
    ensure_dir(target.parent)
    difference = np.asarray(aps) - np.asarray(reference)
    limit = max(float(np.max(np.abs(difference))), np.finfo(np.float32).eps)
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_title(f"APS minus {reference_label} (display autoscaled)")
    axis.axis("off")
    fig.colorbar(image, ax=axis, label="linear intensity difference")
    fig.savefig(target, dpi=160)
    plt.close(fig)
