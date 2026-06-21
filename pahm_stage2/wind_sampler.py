# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Sampler interface for consuming the learned Stage 2 perturbation model."""

from __future__ import annotations

import numpy as np
import torch

from pahm_stage2.unsupervised_model import GMMWindModel


class WindSampler:
    """Sample latent perturbation feature vectors from a trained GMM checkpoint."""

    def __init__(self, model: GMMWindModel, random_state: int | None = None):
        self.model = model
        self.random_state = random_state

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        random_state: int | None = None,
    ) -> "WindSampler":
        """Build a sampler from the persisted unsupervised model artifact."""
        return cls(GMMWindModel.load(checkpoint_path), random_state=random_state)

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Return sampled perturbation context as a float32 torch tensor."""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.random_state is not None:
            np.random.seed(self.random_state)
        samples = self.model.sample(n_samples)
        return torch.as_tensor(samples, dtype=torch.float32)

