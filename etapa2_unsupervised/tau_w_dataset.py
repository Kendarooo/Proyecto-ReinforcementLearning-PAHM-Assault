"""Carga no supervisada de las señales tau_w exportadas por la Etapa 1."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TauWNormalization:
    """Estadísticos calculados solo sobre entrenamiento."""

    mean: float
    std: float


def discover_tau_w_files(root_dir: str | os.PathLike[str], split: str) -> list[Path]:
    """Encuentra archivos tau_w_<split>_<idx>.npy producidos por Etapa 1."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Split inválido: {split}. Use uno de {VALID_SPLITS}.")

    root = Path(root_dir)
    paths = sorted(root.glob(f"tau_w_{split}_*.npy"))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron archivos tau_w_{split}_*.npy en {root}."
        )
    return paths


def _load_signal(path: Path) -> np.ndarray:
    signal = np.load(path).astype(np.float32)
    if signal.ndim != 1:
        raise ValueError(f"{path} debe contener una señal 1D; shape={signal.shape}.")
    return signal


def _window_signal(signal: np.ndarray, window_size: int, stride: int) -> Iterable[np.ndarray]:
    if signal.shape[0] < window_size:
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: signal.shape[0]] = signal
        yield padded
        return

    for start in range(0, signal.shape[0] - window_size + 1, stride):
        yield signal[start : start + window_size]


class TauWWindowDataset(Dataset):
    """Dataset no supervisado: cada item es solo una ventana de tau_w."""

    def __init__(
        self,
        file_paths: list[Path],
        window_size: int,
        stride: int,
        normalization: TauWNormalization | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size debe ser positivo.")
        if stride <= 0:
            raise ValueError("stride debe ser positivo.")

        self.file_paths = file_paths
        self.window_size = window_size
        self.stride = stride

        windows: list[np.ndarray] = []
        for path in file_paths:
            signal = _load_signal(path)
            windows.extend(_window_signal(signal, window_size, stride))

        if not windows:
            raise ValueError("No se pudo construir ninguna ventana de tau_w.")

        self.windows = np.stack(windows).astype(np.float32)
        self.normalization = normalization or TauWNormalization(
            mean=float(self.windows.mean()),
            std=float(self.windows.std() + 1e-8),
        )

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.windows[idx]
        window = (window - self.normalization.mean) / self.normalization.std
        return torch.from_numpy(window.astype(np.float32))


def build_tau_w_dataloaders(
    tau_w_dir: str | os.PathLike[str],
    window_size: int,
    stride: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, TauWNormalization]:
    """Crea dataloaders train/val/test sin etiquetas."""
    train_ds = TauWWindowDataset(
        discover_tau_w_files(tau_w_dir, "train"),
        window_size=window_size,
        stride=stride,
    )
    norm = train_ds.normalization
    val_ds = TauWWindowDataset(
        discover_tau_w_files(tau_w_dir, "val"),
        window_size=window_size,
        stride=stride,
        normalization=norm,
    )
    test_ds = TauWWindowDataset(
        discover_tau_w_files(tau_w_dir, "test"),
        window_size=window_size,
        stride=stride,
        normalization=norm,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, norm

