from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fibre_sim.aps import integrate_aps_frame
from fibre_sim.config import derived_parameters, load_config
from fibre_sim.events import generate_v2e_events
from fibre_sim.fibre import generate_hex_core_centres, simulate_fibre_sequence
from fibre_sim.motion import motion_from_config, piecewise_linear_motion, uniform_motion
from fibre_sim.relay import relay_to_sensor_sequence
from fibre_sim.visualize import split_events_by_time


class ConfigAndMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(PROJECT_ROOT / "configs" / "phase1_usaf.yaml")

    def test_derived_baseline(self):
        derived = derived_parameters(self.cfg)
        self.assertEqual(derived["source_shape_px"], (400, 400))
        self.assertEqual(derived["fibre_shape_px"], (320, 320))
        self.assertEqual(derived["time_samples"], 251)
        self.assertAlmostEqual(derived["relay_magnification"], 30.0625)
        self.assertAlmostEqual(derived["fibre_pitch_on_sensor_px"], 7.3125)

    def test_motion_endpoint(self):
        times, shifts = uniform_motion(0.025, 0.0001, (180.0, 0.0))
        self.assertEqual(len(times), 251)
        np.testing.assert_allclose(shifts[-1], (4.5, 0.0))

    def test_piecewise_xy_motion_hits_both_legs(self):
        times, shifts = piecewise_linear_motion(
            0.05,
            0.0001,
            [0.0, 0.025, 0.05],
            [[0.0, 0.0], [4.5, 0.0], [4.5, 4.5]],
        )
        self.assertEqual(len(times), 501)
        np.testing.assert_allclose(shifts[250], (4.5, 0.0), atol=1e-7)
        np.testing.assert_allclose(shifts[-1], (4.5, 4.5), atol=1e-7)
        np.testing.assert_allclose(np.diff(shifts[:251, 1]), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.diff(shifts[250:, 0]), 0.0, atol=1e-7)

    def test_motion_config_dispatch_is_backward_compatible(self):
        times, shifts = motion_from_config(
            {
                "duration_s": 0.025,
                "dt_s": 0.0001,
                "velocity_um_s": [180.0, 0.0],
            }
        )
        self.assertEqual(len(times), 251)
        np.testing.assert_allclose(shifts[-1], (4.5, 0.0))

    def test_xy_sigma_configs_share_the_same_trajectory(self):
        sigma0 = load_config(PROJECT_ROOT / "configs" / "phase2_xy_usaf_sigma0.yaml")
        sigma08 = load_config(PROJECT_ROOT / "configs" / "phase2_xy_usaf_sigma08.yaml")
        times0, shifts0 = motion_from_config(sigma0["motion"])
        times08, shifts08 = motion_from_config(sigma08["motion"])
        np.testing.assert_array_equal(times0, times08)
        np.testing.assert_array_equal(shifts0, shifts08)
        self.assertEqual(float(sigma0["grin"]["sigma_um"]), 0.0)
        self.assertEqual(float(sigma08["grin"]["sigma_um"]), 0.8)


class OpticalModuleTests(unittest.TestCase):
    def test_core_lattice_and_uniform_response(self):
        centres = generate_hex_core_centres(40.0, 4.5, 2.9)
        self.assertGreater(len(centres), 70)
        distance = np.sqrt(np.sum((centres[None, :, :] - centres[:, None, :]) ** 2, axis=2))
        distance[distance == 0] = np.inf
        self.assertAlmostEqual(float(distance.min()), 4.5, places=4)
        frames = np.ones((2, 80, 80), dtype=np.float32) * 0.7
        distal, signals = simulate_fibre_sequence(
            frames,
            centres,
            pixel_size_um=0.5,
            core_diameter_um=2.9,
            aperture_supersample=16,
        )
        self.assertEqual(distal.shape, frames.shape)
        np.testing.assert_allclose(signals, 0.7, atol=2e-5)
        self.assertGreater(float(distal.max()), 0.5)

    def test_relay_shape_and_constant_interior(self):
        frame = np.ones((2, 32, 32), dtype=np.float32)
        output = relay_to_sensor_sequence(
            frame,
            fibre_pixel_size_um=0.5,
            sensor_shape_px=(12, 16),
            sensor_pixel_pitch_um=1.0,
            magnification=1.0,
            pixel_integration_supersample=2,
        )
        self.assertEqual(output.shape, (2, 12, 16))
        np.testing.assert_allclose(output[:, 2:-2, 2:-2], 1.0, atol=1e-6)

    def test_aps_constant_and_ramp(self):
        times = np.linspace(0, 1, 11)
        constant = np.ones((11, 3, 4), dtype=np.float32) * 0.4
        np.testing.assert_allclose(integrate_aps_frame(constant, times, 0, 1), 0.4)
        ramp = np.broadcast_to(times[:, None, None], (11, 3, 4)).astype(np.float32)
        np.testing.assert_allclose(integrate_aps_frame(ramp, times, 0.15, 0.85), 0.5, atol=1e-6)


class EventModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2e_root = PROJECT_ROOT.parent / "v2e"

    def test_static_sequence_has_no_events(self):
        frames = np.ones((4, 8, 8), dtype=np.float32) * 0.5
        times = np.arange(4, dtype=np.float64) * 0.001
        events, stats = generate_v2e_events(
            frames,
            times,
            v2e_root=self.v2e_root,
            pos_threshold=0.2,
            neg_threshold=0.2,
            device="cpu",
        )
        self.assertEqual(events.shape, (0, 4))
        self.assertEqual(stats["count"], 0)

    def test_bright_then_dark_generates_both_polarities(self):
        frames = np.stack(
            (
                np.ones((8, 8), np.float32) * 0.2,
                np.ones((8, 8), np.float32) * 0.8,
                np.ones((8, 8), np.float32) * 0.2,
            )
        )
        events, _ = generate_v2e_events(
            frames,
            np.array((0.0, 0.001, 0.002)),
            v2e_root=self.v2e_root,
            pos_threshold=0.2,
            neg_threshold=0.2,
            device="cpu",
        )
        self.assertGreater(np.sum(events[:, 3] > 0), 0)
        self.assertGreater(np.sum(events[:, 3] < 0), 0)

    def test_time_segments_are_disjoint_and_complete(self):
        events = np.array(
            (
                (0.0, 1, 1, 1),
                (0.001, 1, 1, 1),
                (0.002, 1, 1, -1),
                (0.003, 1, 1, -1),
            ),
            dtype=np.float32,
        )
        segments = split_events_by_time(
            events, start_s=0.0, end_s=0.003, segment_width_s=0.001
        )
        self.assertEqual([len(segment) for _, _, segment in segments], [1, 1, 2])
        self.assertEqual(sum(len(segment) for _, _, segment in segments), len(events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
