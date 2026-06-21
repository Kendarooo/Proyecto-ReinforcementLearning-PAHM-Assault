# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Unsupervised wind-perturbation representation with Gaussian mixtures."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class GMMFitResult:
    """Training summary for BIC-based GMM selection."""

    selected_n_components: int
    bic_scores: dict[int, float]


class GMMWindModel:
    """GMM wrapper that selects the number of components using BIC."""

    def __init__(
        self,
        n_components_range: tuple[int, int],
        covariance_type: str = "full",
        random_state: int | None = None,
    ):
        self.n_components_range = n_components_range
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: GaussianMixture | None = None
        self.selected_n_components: int | None = None
        self.bic_scores: dict[int, float] = {}

    def fit(self, feature_matrix: np.ndarray) -> GMMFitResult:
        """Fit candidate GMMs and keep the one with the lowest BIC."""
        features = self._validate_feature_matrix(feature_matrix)
        start, stop = self.n_components_range
        if start > stop:
            raise ValueError("n_components_range must be ordered as (min, max)")
        if start < 1:
            raise ValueError("n_components_range minimum must be >= 1")
        if stop > features.shape[0]:
            raise ValueError("n_components_range maximum cannot exceed sample count")

        candidates: dict[int, GaussianMixture] = {}
        self.bic_scores = {}
        for n_components in range(start, stop + 1):
            candidate = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=5,
            )
            candidate.fit(features)
            candidates[n_components] = candidate
            self.bic_scores[n_components] = float(candidate.bic(features))

        self.selected_n_components = min(self.bic_scores, key=self.bic_scores.get)
        self.model = candidates[self.selected_n_components]
        return GMMFitResult(
            selected_n_components=self.selected_n_components,
            bic_scores=dict(self.bic_scores),
        )

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return the latent component assignment for each feature row."""
        self._require_fitted()
        features = self._validate_feature_matrix(feature_matrix)
        return self.model.predict(features)

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample feature vectors from the learned perturbation representation."""
        self._require_fitted()
        samples, _ = self.model.sample(n_samples)
        return np.asarray(samples, dtype=np.float32)

    def save(self, checkpoint_path: str) -> Path:
        """Persist the GMM checkpoint for validation or WindSampler use."""
        self._require_fitted()
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components_range": self.n_components_range,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
            "selected_n_components": self.selected_n_components,
            "bic_scores": self.bic_scores,
            "model": self.model,
        }
        with path.open("wb") as checkpoint:
            pickle.dump(payload, checkpoint)
        return path

    @classmethod
    def load(cls, checkpoint_path: str) -> "GMMWindModel":
        """Load a previously saved GMM checkpoint."""
        with Path(checkpoint_path).open("rb") as checkpoint:
            payload = pickle.load(checkpoint)
        instance = cls(
            n_components_range=tuple(payload["n_components_range"]),
            covariance_type=payload["covariance_type"],
            random_state=payload["random_state"],
        )
        instance.selected_n_components = payload["selected_n_components"]
        instance.bic_scores = payload["bic_scores"]
        instance.model = payload["model"]
        return instance

    def _require_fitted(self) -> None:
        if self.model is None or self.selected_n_components is None:
            raise RuntimeError("GMMWindModel must be fitted before use")

    @staticmethod
    def _validate_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
        features = np.asarray(feature_matrix, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"feature_matrix must have shape (N, F), got {features.shape}")
        if features.shape[0] == 0:
            raise ValueError("feature_matrix must contain at least one row")
        return features

