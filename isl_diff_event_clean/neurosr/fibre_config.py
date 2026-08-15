"""Configuration for fibre-core event reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FibreReconstructionConfig:
    """All numerical choices needed for a reproducible reconstruction run."""

    simulation_config: Path = Path(
        "../fibre_frame_event_sim/configs/phase1_usaf.yaml"
    )
    output_dir: Path = Path("results/fibre_neurosr/phase1_usaf")
    latent_shape: tuple[int, int] = (112, 112)
    event_time_stride: int = 5
    calibration_pixels_per_core: int = 4
    iterations: int = 1_200
    learning_rate: float = 0.03
    event_weight: float = 0.03
    tv_weight: float = 1.5e-3
    huber_beta: float = 0.1
    checkpoint_interval: int = 25
    random_seed: int = 7
    device: str = "cuda"

    def resolve(self, project_root: Path) -> "FibreReconstructionConfig":
        """Return a copy whose two filesystem paths are absolute."""
        config_path = self.simulation_config
        output_path = self.output_dir
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()
        return FibreReconstructionConfig(
            **{
                **self.__dict__,
                "simulation_config": config_path,
                "output_dir": output_path,
            }
        )
