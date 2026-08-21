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
        mask = generate_irregular_core_mask((20, 20), centres, 2.2, seed=3)
        self.assertEqual(int(mask.labels.max()), 3)
        self.assertTrue(np.all(mask.flat_response[mask.labels > 0] > 0))
        for label in range(1, 4):
            self.assertGreater(np.sum(mask.labels == label), 5)


class ObservationBoundaryTests(unittest.TestCase):
    def test_loader_uses_mask_to_map_events_to_core_centres(self) -> None:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[5:7, 5:7] = 2
        response = (labels > 0).astype(np.float32)
        mask = CoreMask(labels, response)
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
