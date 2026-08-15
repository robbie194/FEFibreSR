"""Metrics and result serialization for fibre reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(reconstruction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return full-reference metrics on equally shaped, linear images."""
    return {
        "psnr_db": float(peak_signal_noise_ratio(truth, reconstruction, data_range=1.0)),
        "ssim": float(structural_similarity(truth, reconstruction, data_range=1.0)),
        "mae": float(np.mean(np.abs(reconstruction - truth))),
        "rmse": float(np.sqrt(np.mean((reconstruction - truth) ** 2))),
        "correlation": float(np.corrcoef(reconstruction.ravel(), truth.ravel())[0, 1]),
    }


def save_mode_result(
    output_dir: Path,
    *,
    reconstruction: np.ndarray,
    observable_reconstruction: np.ndarray,
    loss_history: np.ndarray,
    aps_reprojection: np.ndarray,
    event_residual: np.ndarray,
    summary: dict,
) -> None:
    """Save numerical products and compact diagnostic figures for one mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "reconstruction": reconstruction,
        "observable_reconstruction": observable_reconstruction,
        "loss_history": loss_history,
        "aps_reprojection": aps_reprojection,
        "event_residual": event_residual,
    }
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)

    plt.imsave(output_dir / "reconstruction.png", reconstruction, cmap="gray", vmin=0, vmax=1)
    plt.imsave(
        output_dir / "observable_reconstruction.png",
        observable_reconstruction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=180)
    axes[0].plot(loss_history[:, 0], label="total")
    axes[0].plot(loss_history[:, 1], label="APS")
    axes[0].plot(loss_history[:, 2], label="events")
    axes[0].set_title("Optimisation losses")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].scatter(aps_reprojection[:, 0], aps_reprojection[:, 1], s=3, alpha=0.5)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(xlabel="observed core APS", ylabel="predicted core APS", title="APS reprojection")
    axes[2].hist(event_residual.ravel(), bins=80)
    axes[2].set(xlabel="predicted - observed lin-log change", title="Event residual")
    figure.tight_layout()
    figure.savefig(output_dir / "diagnostics.png")
    plt.close(figure)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def save_comparison(
    output_dir: Path,
    truth: np.ndarray,
    initial: np.ndarray,
    results: dict[str, np.ndarray],
    crop_slices: tuple[slice, slice],
) -> None:
    """Save one common-scale panel for visual comparison of all methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [("Truth (evaluation only)", truth[crop_slices]), ("APS interpolation", initial[crop_slices])]
    titles = {
        "aps_only": "APS only",
        "events_only": "Events only",
        "joint": "APS + core events",
    }
    panels.extend((titles.get(name, name), image[crop_slices]) for name, image in results.items())
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=200)
    for axis, (title, image) in zip(np.atleast_1d(axes), panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction_comparison.png")
    plt.close(figure)
