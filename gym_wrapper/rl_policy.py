"""Utilidades para cargar y ejecutar políticas RL en la demo PAHM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()) / path


def get_demo_model_path(
    config: dict[str, Any],
    *,
    model_type: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Obtiene la ruta de modelo RL seleccionada por la configuracion de demo."""
    demo_config = config.get("demo", {})
    selected_type = model_type or demo_config.get("rl_model_type", "robust")
    models = demo_config.get("models", {})
    if selected_type not in models:
        raise KeyError(f"RL model type not configured: {selected_type}")
    return _resolve_path(models[selected_type], Path(base_dir) if base_dir else None)


def load_rl_policy(model_path: str | Path, algorithm: str = "PPO"):
    """Carga una política Stable Baselines3 sin iniciar entrenamiento."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"RL policy model not found: {path}")

    try:
        from stable_baselines3 import A2C, PPO, SAC
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required to load RL policies. "
            "Install dependencies from requirements.txt."
        ) from exc

    algorithms = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
    name = algorithm.upper()
    if name not in algorithms:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    return algorithms[name].load(str(path))


def predict_rl_action(policy, observation, action_space, deterministic: bool = True):
    """Predice una acción y la adapta al espacio de acciones del entorno."""
    action, _ = policy.predict(observation, deterministic=deterministic)
    action = np.asarray(action, dtype=action_space.dtype).reshape(action_space.shape)
    action = np.clip(action, action_space.low, action_space.high)
    return action.astype(action_space.dtype)


def apply_rl_control_step(
    env,
    policy,
    observation,
    *,
    deterministic: bool = True,
    model_path: str | Path | None = None,
):
    """Ejecuta un paso de la demo usando una política RL ya cargada."""
    action = predict_rl_action(
        policy,
        observation,
        env.action_space,
        deterministic=deterministic,
    )
    next_obs, reward, terminated, truncated, info = env.step(action)
    info = dict(info)
    info.update(
        {
            "control_mode": "RL",
            "rl_model_path": str(model_path) if model_path is not None else None,
            "rl_action": float(np.asarray(action).reshape(-1)[0]),
        }
    )
    return next_obs, reward, terminated, truncated, info
