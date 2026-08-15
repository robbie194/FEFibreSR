from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_SRC = ROOT / "fibre_frame_event_sim" / "src"
if str(SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SRC))

from fibre_sim.fibre import simulate_fibre_sequence
from fibre_sim.grin import simulate_grin_sequence
from neurosr.fibre_data import (
    CoreCalibration,
    aggregate_cumulative_events,
    extract_core_aps,
)
from neurosr.fibre_event_loss import predict_cumulative_event_change
from neurosr.fibre_forward import FibreCoreForward


class CoreObservationTests(unittest.TestCase):
    def test_core_aps_undoes_gain_and_takes_median(self) -> None:
        calibration = CoreCalibration(
            pixel_xy=np.array([[[0, 0], [1, 0], [2, 0]]], dtype=np.int32),
            gain=np.array([[0.5, 1.0, 0.25]], dtype=np.float32),
        )
        aps = np.array([[0.2, 0.4, 0.11]], dtype=np.float32)
        np.testing.assert_allclose(extract_core_aps(aps, calibration), [0.4])

    def test_event_aggregation_preserves_readouts_and_polarity(self) -> None:
        calibration = CoreCalibration(
            pixel_xy=np.array([[[1, 1], [2, 1]]], dtype=np.int32),
            gain=np.ones((1, 2), dtype=np.float32),
        )
        events = np.array(
            [[0.1, 1, 1, 1], [0.2, 2, 1, -1], [0.2, 1, 1, 1]],
            dtype=np.float32,
        )
        cumulative = aggregate_cumulative_events(
            events,
            np.array([0.0, 0.1, 0.2]),
            calibration,
            (3, 4),
            0.2,
            0.3,
        )
        expected = np.array(
            [[[0.0, 0.0]], [[0.2, 0.0]], [[0.4, -0.3]]], dtype=np.float32
        )
        np.testing.assert_allclose(cumulative, expected)


class FibreForwardTests(unittest.TestCase):
    def test_torch_forward_matches_numpy_simulator(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.uniform(0.05, 1.0, (40, 40)).astype(np.float32)
        shifts = np.array([[0.0, 0.0], [0.4, -0.25], [0.8, 0.1]], dtype=np.float32)
        centres = np.array([[-4.0, -3.0], [0.0, 0.0], [4.0, 3.0]], dtype=np.float32)
        for sigma_um in (0.0, 0.8):
            with self.subTest(sigma_um=sigma_um):
                grin = simulate_grin_sequence(
                    source,
                    shifts,
                    object_pixel_size_um=0.5,
                    fibre_shape_px=(24, 24),
                    fibre_pixel_size_um=0.5,
                    sigma_um=sigma_um,
                )
                _, expected = simulate_fibre_sequence(
                    grin,
                    centres,
                    pixel_size_um=0.5,
                    core_diameter_um=2.9,
                )
                model = FibreCoreForward(
                    source_shape=source.shape,
                    source_pixel_size_um=0.5,
                    fibre_shape=(24, 24),
                    fibre_pixel_size_um=0.5,
                    grin_magnification=1.0,
                    grin_sigma_um=sigma_um,
                    grin_transmission=1.0,
                    fibre_transmission=1.0,
                    core_centres_xy_um=centres,
                    core_diameter_um=2.9,
                    aperture_supersample=32,
                )
                actual = model(
                    torch.from_numpy(source), torch.from_numpy(shifts)
                ).detach().numpy()
                np.testing.assert_allclose(
                    actual, expected, atol=2.5e-3, rtol=2.5e-3
                )

    def test_constant_core_signal_has_no_events(self) -> None:
        signal = torch.full((5, 2), 0.4)
        gain = torch.tensor([[1.0, 0.8], [0.9, 0.7]])
        predicted = predict_cumulative_event_change(signal, gain, 255.0)
        torch.testing.assert_close(predicted, torch.zeros_like(predicted))


if __name__ == "__main__":
    unittest.main(verbosity=2)
