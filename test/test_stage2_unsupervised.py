"""Pruebas del contrato Etapa 1 -> Etapa 2 no supervisada."""

import json
import os
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _write_tau_w_fixture(root_dir) -> None:
    for split in ("train", "val", "test"):
        for idx in range(2):
            signal = np.linspace(-1.0, 1.0, 64, dtype=np.float32) + idx * 0.1
            np.save(root_dir / f"tau_w_{split}_{idx}.npy", signal)


def test_tau_w_dataloader_returns_only_unlabeled_windows(tmp_path) -> None:
    """Etapa 2 no debe recibir labels ni tuplas (entrada, etiqueta)."""
    from etapa2_unsupervised.tau_w_dataset import build_tau_w_dataloaders

    _write_tau_w_fixture(tmp_path)
    train_loader, val_loader, test_loader, norm = build_tau_w_dataloaders(
        tau_w_dir=tmp_path,
        window_size=16,
        stride=8,
        batch_size=4,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert isinstance(train_batch, torch.Tensor)
    assert isinstance(val_batch, torch.Tensor)
    assert isinstance(test_batch, torch.Tensor)
    assert train_batch.shape == (4, 16)
    assert val_batch.shape == (4, 16)
    assert test_batch.shape == (4, 16)
    assert norm.std > 0.0


def test_autoencoder_epoch_optimizes_reconstruction_only(tmp_path) -> None:
    """El autoencoder simple usa x como objetivo interno y MSE de reconstrucción."""
    from etapa2_unsupervised.autoencoder import TauWAutoencoder
    from etapa2_unsupervised.tau_w_dataset import build_tau_w_dataloaders
    from etapa2_unsupervised.train_unsupervised import run_epoch

    _write_tau_w_fixture(tmp_path)
    train_loader, _, _, _ = build_tau_w_dataloaders(
        tau_w_dir=tmp_path,
        window_size=16,
        stride=8,
        batch_size=8,
    )
    model = TauWAutoencoder(input_dim=16, hidden_dim=8, latent_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = run_epoch(model, train_loader, torch.device("cpu"), optimizer)

    assert loss > 0.0


def test_stage2_config_falls_back_to_stage1_tau_w_output_dir(tmp_path) -> None:
    """El punto de conexión por defecto es estimator_hyperparameters.tau_w_output_dir."""
    from etapa2_unsupervised.train_unsupervised import load_stage2_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "estimator_hyperparameters": {
                    "tau_w_output_dir": "outputs/custom_tau_w"
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_stage2_config(config_path)

    assert config["tau_w_input_dir"] == "outputs/custom_tau_w"
    assert config["model_type"] == "autoencoder"
