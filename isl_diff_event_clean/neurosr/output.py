"""Result serialization, visualization, and reference comparison."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize_u8(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    span = float(value.max() - value.min())
    if span == 0:
        return np.ones_like(value, dtype=np.uint8)
    # The legacy path gives a floating-point image to OpenCV, whose saturating
    # conversion rounds to the nearest integer. Make that conversion explicit.
    normalized = (value - value.min()) / span * 255
    return np.rint(normalized).astype(np.uint8)


def save_results(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    summary: dict,
) -> None:
    """Save raw numerical products, readable PNGs, and a compact comparison panel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)
        if array.ndim == 2 and not name.startswith("trajectory"):
            cv2.imwrite(str(output_dir / f"{name}.png"), normalize_u8(array))
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 4, figsize=(12, 7), dpi=200)
    panels = (
        ("frame_sharp", "Reference APS", "gray"),
        ("frame_blurred", "Blurred APS", "gray"),
        ("reconstruction", "NeuroSR reconstruction", "gray"),
        ("event_iwe_target", "Warped-event IWE", "seismic"),
        ("background", "Learned background", "inferno"),
        ("event_iwe_predicted", "Image-predicted IWE", "seismic"),
        ("motion_blur_kernel", "Motion PSF", "gray"),
    )
    for axis, (key, title, color_map) in zip(axes.flat, panels, strict=False):
        axis.imshow(arrays[key], cmap=color_map)
        axis.set_title(title)
        axis.axis("off")
    axes.flat[-1].plot(arrays["loss_history"])
    axes.flat[-1].set_title("Reconstruction loss")
    axes.flat[-1].set_xlabel("checkpoint (100 iterations)")
    figure.tight_layout()
    figure.savefig(output_dir / "final_reconstruction_comparison.png")
    plt.close(figure)


def compare_result_directories(
    candidate_dir: Path, reference_dir: Path
) -> dict[str, dict[str, float | bool]]:
    """Compare shared arrays and account for nondeterministic GPU accumulation.

    ``allclose`` deliberately keeps strict element-wise tolerances.
    ``numerically_equivalent`` is the end-to-end reproducibility criterion: it
    accepts the small ordering error introduced by CUDA atomic accumulation,
    while still rejecting changes that alter the reconstructed signal.
    """
    report: dict[str, dict[str, float | bool]] = {}
    for candidate_path in sorted(candidate_dir.glob("*.npy")):
        reference_path = reference_dir / candidate_path.name
        if not reference_path.exists():
            continue
        candidate = np.load(candidate_path)
        reference = np.load(reference_path)
        if candidate.shape != reference.shape:
            report[candidate_path.stem] = {
                "shape_match": False,
                "allclose": False,
                "numerically_equivalent": False,
                "max_abs_error": float("inf"),
                "rmse": float("inf"),
                "normalized_rmse": float("inf"),
                "correlation": float("nan"),
            }
            continue
        error = candidate.astype(np.float64) - reference.astype(np.float64)
        value_range = float(
            max(candidate.max(), reference.max())
            - min(candidate.min(), reference.min())
        )
        rmse = float(np.sqrt(np.mean(error**2)))
        signal_scale = max(
            value_range,
            float(np.max(np.abs(candidate))),
            float(np.max(np.abs(reference))),
        )
        normalized_rmse = rmse / signal_scale if signal_scale else rmse
        if np.array_equal(candidate, reference):
            correlation = 1.0
        else:
            correlation = float(
                np.corrcoef(candidate.reshape(-1), reference.reshape(-1))[0, 1]
            )
        equivalent = normalized_rmse <= 5e-3 and correlation >= 0.999
        report[candidate_path.stem] = {
            "shape_match": True,
            "allclose": bool(np.allclose(candidate, reference, rtol=1e-5, atol=1e-6)),
            "numerically_equivalent": bool(equivalent),
            "max_abs_error": float(np.max(np.abs(error))),
            "rmse": rmse,
            "normalized_rmse": normalized_rmse,
            "correlation": correlation,
        }
    return report
