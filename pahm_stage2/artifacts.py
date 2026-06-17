# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Artifact-writing helpers for Stage 2 generated wind signals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_tau_signal(
    tau_w: np.ndarray,
    trajectory_id: str,
    outputs_config: dict,
) -> Path:
    """Save one tau_w(t) signal and return its output path."""
    file_format = outputs_config.get("file_format", "npy")
    if file_format != "npy":
        raise ValueError("Only outputs.file_format='npy' is currently supported")

    output_dir = Path(outputs_config["tau_w_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = outputs_config.get("file_prefix", "tau_w")
    output_path = output_dir / f"{file_prefix}_{trajectory_id}.npy"
    np.save(output_path, tau_w)
    return output_path


def write_manifest(manifest: dict, manifest_path: str) -> Path:
    """Write the manifest JSON for generated tau_w(t) signals."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
