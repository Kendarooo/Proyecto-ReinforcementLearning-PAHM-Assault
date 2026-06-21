# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Estimator interfaces for PAHM wind-signal generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WindEstimatorInterface(ABC):
    """Common interface for wind estimators consumed by the Stage 2 pipeline."""

    @abstractmethod
    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        """
        trajectory: shape (T, 4) with columns [sin_theta, cos_theta, theta_dot, u]
        returns:    shape (T,) estimated wind torque signal tau_w(t)
        """


class DummyWindEstimator(WindEstimatorInterface):
    """Returns zeros, simulating absence of perturbation (NFR-6c)."""

    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        return np.zeros(trajectory.shape[0])


class PAHMWindEstimator(WindEstimatorInterface):
    """Load the G1 GRU/LSTM checkpoint once estimador_wind.pth is available."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        raise NotImplementedError(
            "PAHMWindEstimator will be implemented when G1 provides the checkpoint."
        )

    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def build_estimator(config: dict) -> WindEstimatorInterface:
    """Build the configured wind estimator without changing the pipeline."""
    estimator_config = config["estimator"]
    estimator_type = estimator_config["type"]
    if estimator_type == "dummy":
        return DummyWindEstimator()
    if estimator_type == "pahm":
        return PAHMWindEstimator(
            checkpoint_path=estimator_config["checkpoint_path"],
            device=estimator_config.get("device", "cpu"),
        )
    raise ValueError(f"Unsupported estimator.type: {estimator_type}")
