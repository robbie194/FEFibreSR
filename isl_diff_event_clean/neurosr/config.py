"""Configuration for the reproducible DAVIS reconstruction experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Every value needed to reproduce the reference NeuroSR run."""

    input_path: Path = Path(
        "/home/robbie/tyf_data/2025-01-05_DVS_multicore_fibre_nolen_40x/"
        "move_50pics/dvSave-2025_01_04_21_56_09.aedat4"
    )
    output_dir: Path = Path("results/fig/tyf_test")
    sensor_height: int = 260
    sensor_width: int = 346
    requested_start_us: int = 2_450_000
    use_two_exposures: bool = True
    trajectory_segments: int = 12
    reference_count: int = 3
    motion_iterations: int = 2_100
    reconstruction_iterations: int = 2_100
    super_resolution_scale: int = 2
    event_splat_sigma: float = 0.849
    random_seed: int = 0
    cpu_threads: int = 10

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return self.sensor_height, self.sensor_width
