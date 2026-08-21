from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibre_iwe.data import load_core_observations
from fibre_iwe.geometry import centres_from_mask, generate_irregular_core_mask
from fibre_iwe.io import CoreMask, Recording, save_core_mask, save_recording
from fibre_iwe.output import save_reconstruction_results
from fibre_iwe.reconstruction import ReconstructionResult
from fibre_iwe.render import render_iwe, warp_events


class CoreMaskTests(unittest.TestCase):
    def test_centres_are_derived_from_labels(self) -> None:
        labels = np.zeros((8, 10), dtype=np.int32)
        labels[1:3, 2:4] = 1
        labels[5:8, 7:9] = 2
        centres = centres_from_mask(labels)
        np.testing.assert_allclose(centres, [[2.5, 1.5], [7.5, 6.0]])

    def test_irregular_spots_do_not_overlap(self) -> None:
        centres = np.array([[6, 6], [14, 6], [10, 13]], dtype=np.float32)
        geometry = generate_irregular_core_mask((20, 20), centres, 2.2, seed=3)
        mask = geometry.core_mask
        self.assertEqual(int(mask.labels.max()), 3)
        self.assertTrue(
            np.all(geometry.proximal_response[mask.labels > 0] > 0)
        )
        for label in range(1, 4):
            self.assertGreater(np.sum(mask.labels == label), 5)

    def test_saved_mask_contains_labels_only(self) -> None:
        labels = np.array([[0, 1], [0, 1]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core_mask.npz"
            save_core_mask(path, CoreMask(labels))
            with np.load(path) as values:
                self.assertEqual(values.files, ["labels"])


class ObservationBoundaryTests(unittest.TestCase):
    def test_loader_uses_mask_to_map_events_to_core_centres(self) -> None:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[5:7, 5:7] = 2
        mask = CoreMask(labels)
        aps = np.zeros((8, 8), dtype=np.float32)
        aps[labels == 1] = 0.3
        aps[labels == 2] = 0.8
        events = np.array(
            [[0.1, 1, 2, 1], [0.2, 6, 5, -1], [0.3, 0, 7, 1]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_core_mask(root / "core_mask.npz", mask)
            save_recording(root / "recording.h5", Recording(aps, events, 0, 0.4, (8, 8)))
            observations = load_core_observations(root)
        np.testing.assert_allclose(observations.core_aps, [0.3, 0.8])
        np.testing.assert_allclose(observations.event_xy, [[1.5, 1.5], [5.5, 5.5]])
        np.testing.assert_array_equal(observations.event_polarity, [1, -1])


class RealDataOutputTests(unittest.TestCase):
    def test_results_do_not_require_ground_truth(self) -> None:
        shape = (32, 32)
        image = np.linspace(0.05, 0.95, np.prod(shape), dtype=np.float32).reshape(
            shape
        )
        iwe = image - image.mean()
        history = np.array([[0, 1.0, 0.5, 0.25, 0.1]], dtype=np.float64)
        aps_pairs = np.array([[0.2, 0.21], [0.8, 0.79]], dtype=np.float32)
        result = ReconstructionResult(
            image,
            image,
            image,
            iwe,
            iwe,
            np.ones(shape, dtype=np.float32),
            history,
            history,
            aps_pairs,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = save_reconstruction_results(
                output_dir, result, None, {"status": "real data"}, {}
            )
            self.assertTrue((output_dir / "03_reconstruction_comparison.png").is_file())
        self.assertFalse(summary["ground_truth_available"])
        self.assertIsNone(summary["metrics"])
        self.assertAlmostEqual(summary["data_fidelity"]["iwe_cosine_similarity"], 1.0)


class EventWarpTests(unittest.TestCase):
    def test_correct_trajectory_focuses_moving_events(self) -> None:
        time = torch.linspace(0, 1, 80)
        base_x = torch.where(torch.arange(80) % 2 == 0, 10.0, 18.0)
        base_y = torch.where(torch.arange(80) % 3 == 0, 12.0, 20.0)
        xy = torch.stack((base_x + 5 * time, base_y + 2 * time), dim=1)
        polarity = torch.ones(80)
        zero = torch.zeros((2, 2))
        known = torch.tensor([[0.0, 0.0], [5.0, 2.0]])
        blurred = render_iwe(xy, time, polarity, zero, (32, 32), 0.8, signed=False)
        focused = render_iwe(xy, time, polarity, known, (32, 32), 0.8, signed=False)
        self.assertGreater(float(focused.var()), float(blurred.var()) * 1.4)
        warped = warp_events(xy, time, known)
        np.testing.assert_allclose(
            (warped - torch.stack((base_x, base_y), dim=1)).mean(0).numpy(),
            [2.5, 1.0],
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
