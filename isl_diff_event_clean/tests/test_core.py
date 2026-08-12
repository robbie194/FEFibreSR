from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from neurosr.events import bilinear_splat, gaussian_splat
from neurosr.motion import PiecewiseLinearTrajectory
from neurosr.optimization import block_average, predicted_iwe
from neurosr.output import compare_result_directories


class EventRenderingTests(unittest.TestCase):
    def test_bilinear_splat_conserves_interior_weight(self) -> None:
        image = bilinear_splat(
            torch.tensor([2.25]),
            torch.tensor([3.75]),
            torch.tensor([2.0]),
            (8, 8),
        )
        self.assertAlmostEqual(float(image.sum()), 2.0, places=6)

    def test_gaussian_splat_conserves_interior_weight(self) -> None:
        image = gaussian_splat(
            torch.tensor([4.2]),
            torch.tensor([4.3]),
            torch.tensor([3.0]),
            (10, 10),
            sigma=0.8,
            kernel_size=5,
        )
        self.assertAlmostEqual(float(image.sum()), 3.0, places=5)


class MotionAndImageTests(unittest.TestCase):
    def test_piecewise_trajectory_reaches_increment_sum(self) -> None:
        model = PiecewiseLinearTrajectory(8, 2, torch.device("cpu"))
        increments = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        dense = model(increments).squeeze()
        np.testing.assert_allclose(dense[-1].numpy(), [2.0, 3.0], atol=1e-6)

    def test_block_average(self) -> None:
        image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        expected = torch.tensor([[2.5, 4.5], [10.5, 12.5]])
        torch.testing.assert_close(block_average(image, 2), expected)

    def test_constant_image_has_zero_predicted_iwe(self) -> None:
        output = predicted_iwe(torch.ones((5, 6)), torch.tensor([3.0, -2.0]))
        torch.testing.assert_close(output, torch.zeros_like(output))


class ResultComparisonTests(unittest.TestCase):
    def test_small_numerical_drift_is_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            reference = root / "reference"
            candidate.mkdir()
            reference.mkdir()
            baseline = np.linspace(0, 1, 100, dtype=np.float32)
            np.save(reference / "image.npy", baseline)
            np.save(candidate / "image.npy", baseline + 1e-4)

            report = compare_result_directories(candidate, reference)["image"]

        self.assertTrue(report["shape_match"])
        self.assertTrue(report["numerically_equivalent"])


if __name__ == "__main__":
    unittest.main()
