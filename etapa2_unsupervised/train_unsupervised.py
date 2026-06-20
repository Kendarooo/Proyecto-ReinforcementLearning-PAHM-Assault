"""Entrenamiento no supervisado de Etapa 2 conectado a outputs/tau_w."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from etapa2_unsupervised.autoencoder import TauWAutoencoder  # noqa: E402
from etapa2_unsupervised.tau_w_dataset import build_tau_w_dataloaders  # noqa: E402


DEFAULT_STAGE2_CONFIG = {
    "model_type": "autoencoder",
    "window_size": 128,
    "stride": 32,
    "hidden_dim": 64,
    "latent_dim": 8,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "random_seed": 42,
    "checkpoint_dir": "outputs/stage2_unsupervised",
    "wandb": {
        "enabled": False,
        "project": "etapa-2-unsupervised",
        "entity": None,
        "name": "tau-w-autoencoder",
        "mode": "online",
        "tags": ["stage-2", "unsupervised", "tau-w"],
    },
}


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else project_root / raw_path


def load_stage2_config(config_path: str | os.PathLike[str]) -> dict:
    """Carga config y conecta Etapa 2 al directorio tau_w de Etapa 1."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    stage2 = {**DEFAULT_STAGE2_CONFIG, **config.get("stage2_unsupervised", {})}
    estimator = config.get("estimator_hyperparameters", {})
    stage2["tau_w_input_dir"] = stage2.get(
        "tau_w_input_dir", estimator.get("tau_w_output_dir", "outputs/tau_w")
    )
    return stage2


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_wandb(config: dict):
    """Inicializa W&B si está habilitado para Etapa 2."""
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    return wandb.init(
        project=wandb_config.get("project", "etapa-2-unsupervised"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("name", "tau-w-autoencoder"),
        mode=wandb_config.get("mode", "online"),
        tags=wandb_config.get("tags"),
        config=config,
    )


def run_epoch(
    model: TauWAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Optimiza o evalúa solo MSE(x_recon, x); no hay labels."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x = batch.to(device)
        if is_training:
            optimizer.zero_grad()

        recon = model(x)
        loss = F.mse_loss(recon, x)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train(config: dict) -> Path:
    if config["model_type"] != "autoencoder":
        raise ValueError(
            "Etapa 2 implementa autoencoder simple; la pérdida permitida es "
            "solo reconstrucción."
        )

    set_reproducibility(int(config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = _init_wandb(config)

    tau_w_dir = _resolve_path(config["tau_w_input_dir"])
    checkpoint_dir = _resolve_path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, normalization = build_tau_w_dataloaders(
        tau_w_dir=tau_w_dir,
        window_size=int(config["window_size"]),
        stride=int(config["stride"]),
        batch_size=int(config["batch_size"]),
    )

    model = TauWAutoencoder(
        input_dim=int(config["window_size"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    best_val = float("inf")
    best_path = checkpoint_dir / "tau_w_autoencoder_best.pth"

    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, device)

            metrics = {
                "epoch": epoch,
                "train/reconstruction": train_loss,
                "val/reconstruction": val_loss,
            }
            if wandb_run is not None:
                wandb.log(metrics, step=epoch)

            print(
                f"Epoch {epoch:03d}/{config['epochs']} | "
                f"train_reconstruction={train_loss:.6f} | "
                f"val_reconstruction={val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_type": "autoencoder",
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "normalization": {
                            "mean": normalization.mean,
                            "std": normalization.std,
                        },
                        "loss": {
                            "validation_reconstruction": val_loss,
                        },
                    },
                    best_path,
                )

        with torch.no_grad():
            test_loss = run_epoch(model, test_loader, device)
        if wandb_run is not None:
            wandb.log({"test/reconstruction": test_loss}, step=int(config["epochs"]))
            wandb.save(str(best_path))
        print(f"test_reconstruction={test_loss:.6f}")
        print(f"checkpoint={best_path}")
        return best_path
    finally:
        if wandb_run is not None:
            wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena Etapa 2 de forma no supervisada sobre tau_w."
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "gym_wrapper/config.json"),
        help="Config compartida con Etapa 1.",
    )
    parser.add_argument(
        "--tau-w-dir",
        default=None,
        help="Sobrescribe el directorio outputs/tau_w si se necesita.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage2_config(args.config)
    if args.tau_w_dir is not None:
        config["tau_w_input_dir"] = args.tau_w_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train(config)


if __name__ == "__main__":
    main()
