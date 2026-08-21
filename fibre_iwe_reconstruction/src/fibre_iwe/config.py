"""Configuration loading for the standalone core-IWE experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved configuration shared by generation, reconstruction and evaluation."""

    project_root: Path
    output_root: Path
    values: dict[str, Any]

    @property
    def observations_dir(self) -> Path:
        return self.output_root / "observations"

    @property
    def private_truth_dir(self) -> Path:
        return self.output_root / "private_truth"

    @property
    def results_dir(self) -> Path:
        return self.output_root / "results"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML and resolve paths relative to this experiment directory."""
    config_path = Path(path).resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    project_root = config_path.parents[1]
    output_root = Path(values["project"]["output_dir"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    return ExperimentConfig(project_root, output_root, values)
