# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import numpy as np
import torch

from pahm_stage2.unsupervised_model import GMMWindModel
from pahm_stage2.wind_sampler import WindSampler


def _fit_and_save_model(tmp_path):
    rng = np.random.default_rng(21)
    features = np.vstack(
        [
            rng.normal(0.0, 0.01, size=(8, 6)),
            rng.normal(0.5, 0.01, size=(8, 6)),
        ]
    )
    model = GMMWindModel((2, 2), random_state=21)
    model.fit(features)
    checkpoint_path = tmp_path / "gmm_wind_model.pkl"
    model.save(str(checkpoint_path))
    return checkpoint_path


def test_sample_returns_correct_shape(tmp_path):
    checkpoint_path = _fit_and_save_model(tmp_path)
    sampler = WindSampler.from_checkpoint(str(checkpoint_path), random_state=3)

    sample = sampler.sample(5)

    assert sample.shape == (5, 6)


def test_sample_output_is_tensor(tmp_path):
    checkpoint_path = _fit_and_save_model(tmp_path)
    sampler = WindSampler.from_checkpoint(str(checkpoint_path), random_state=3)

    sample = sampler.sample(2)

    assert isinstance(sample, torch.Tensor)
    assert sample.dtype == torch.float32


def test_sampler_loads_from_checkpoint(tmp_path):
    checkpoint_path = _fit_and_save_model(tmp_path)

    sampler = WindSampler.from_checkpoint(str(checkpoint_path))

    assert sampler.model.selected_n_components == 2

