# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Dataset loader for estimated wind-signal extraction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TauTrajectory:
    """Single trajectory input for the Stage 2 wind pipeline."""

    trajectory_id: str
    values: np.ndarray


class TauTrajectoryDataset:
    """Load PAHM trajectories with columns [sin_theta, cos_theta, theta_dot, u]."""

    def __init__(self, config: dict):
        self.config = config
        self.input_dir = self._resolve_input_dir(config)
        self.file_pattern = config["data"].get("file_pattern", "*.npy")
        self._trajectories = self._load_trajectories()

    def __len__(self) -> int:
        return len(self._trajectories)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray]:
        item = self._trajectories[index]
        return item.trajectory_id, item.values

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        for item in self._trajectories:
            yield item.trajectory_id, item.values

    @staticmethod
    def _resolve_input_dir(config: dict) -> Path:
        source = config["data"]["source"]
        if source == "synthetic":
            return Path(config["data"]["synthetic_input_dir"])
        if source == "real":
            return Path(config["data"]["real_input_dir"])
        raise ValueError("data.source must be 'synthetic' or 'real'")

    def _load_trajectories(self) -> list[TauTrajectory]:
        files = sorted(self.input_dir.glob(self.file_pattern))
        trajectories = []
        for path in files:
            trajectory = np.load(path)
            self._validate_trajectory(path, trajectory)
            trajectories.append(
                TauTrajectory(
                    trajectory_id=path.stem,
                    values=np.asarray(trajectory, dtype=np.float32),
                )
            )
        return trajectories

    @staticmethod
    def _validate_trajectory(path: Path, trajectory: np.ndarray) -> None:
        if trajectory.ndim != 2 or trajectory.shape[1] != 4:
            raise ValueError(
                f"Trajectory {path} must have shape (T, 4), got {trajectory.shape}"
            )
