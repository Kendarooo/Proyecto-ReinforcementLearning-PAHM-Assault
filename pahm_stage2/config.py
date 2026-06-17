# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Configuration loading for Stage 2 wind-signal generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_SECTIONS = ("data", "estimator", "outputs", "wandb")


def _seed_torch(seed: int) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def load_config(config_path: str) -> dict[str, Any]:
    """Load a JSON config, validate required sections, and apply seeds."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config: dict[str, Any] = json.load(config_file)

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        joined = ", ".join(missing_sections)
        raise ValueError(f"Missing required config section(s): {joined}")

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed)
    return config
