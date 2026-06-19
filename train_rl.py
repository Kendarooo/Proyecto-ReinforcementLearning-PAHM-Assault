"""Entrenamiento headless de agentes RL para el entorno PAHM de Etapa 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gym_wrapper.learned_pahm_ode import LearnedPAHMODE


DEFAULT_RL_TRAINING = {
    "enabled": True,
    "mode": "naive",
    "algorithm": "PPO",
    "total_timesteps": 10000,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "device": "auto",
    "render": False,
    "model_path": "pahm_model/pahm_fast_v2_best.pth",
    "reset_angle_deg": 720,
    "model_output_dir": "artifacts/stage3/models",
    "model_name": "pahm_ppo_naive",
    "checkpoint_freq": 5000,
    "log_dir": "artifacts/stage3/logs",
}


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Carga configuracion externa y normaliza la seccion `rl_training`."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    rl_config = {**DEFAULT_RL_TRAINING, **config.get("rl_training", {})}
    config["rl_training"] = rl_config
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def make_training_env(config: dict[str, Any]) -> LearnedPAHMODE:
    """Construye el entorno PAHM para entrenamiento sin renderizado."""
    rl_config = config["rl_training"]
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    mode = rl_config.get("mode", "naive")
    if mode not in {"naive", "robust"}:
        raise ValueError(f"Modo RL no soportado: {mode}")

    model_path = _resolve_path(rl_config["model_path"], config_dir)
    enable_wind = mode == "robust"
    randomize_wind_pattern = bool(
        rl_config.get("randomize_wind_pattern", enable_wind)
    )

    return LearnedPAHMODE(
        render_mode=None,
        model_path=str(model_path),
        reset_angle_deg=rl_config.get("reset_angle_deg", 720),
        enable_wind=enable_wind,
        wind_pattern=rl_config.get("wind_pattern"),
        wind_seed=rl_config.get("wind_seed"),
        randomize_wind_pattern=randomize_wind_pattern,
        config=config,
    )


def build_agent(algorithm_name: str, env, config: dict[str, Any]):
    """Construye el agente Stable Baselines3 indicado por configuracion."""
    try:
        from stable_baselines3 import A2C, PPO, SAC
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 es requerido para entrenar agentes RL. "
            "Instala las dependencias de requirements.txt."
        ) from exc

    algorithms = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
    }
    name = algorithm_name.upper()
    if name not in algorithms:
        raise ValueError(f"Algoritmo RL no soportado: {algorithm_name}")

    rl_config = config["rl_training"]
    kwargs = {
        "policy": rl_config.get("policy", "MlpPolicy"),
        "env": env,
        "learning_rate": rl_config["learning_rate"],
        "gamma": rl_config["gamma"],
        "device": rl_config["device"],
        "verbose": int(rl_config.get("verbose", 0)),
        "tensorboard_log": str(_resolve_path(rl_config["log_dir"], Path(config["_config_dir"]))),
    }
    if name in {"PPO", "A2C"}:
        kwargs["n_steps"] = int(rl_config["n_steps"])
    if name == "PPO":
        kwargs["batch_size"] = int(rl_config["batch_size"])

    return algorithms[name](**kwargs)


def _build_callbacks(config: dict[str, Any]):
    try:
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 es requerido para crear checkpoints RL."
        ) from exc

    rl_config = config["rl_training"]
    checkpoint_freq = int(rl_config.get("checkpoint_freq", 0))
    if checkpoint_freq <= 0:
        return None

    config_dir = Path(config["_config_dir"])
    checkpoint_dir = _resolve_path(
        rl_config.get("checkpoint_dir", rl_config["model_output_dir"]),
        config_dir,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return CallbackList(
        [
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=str(rl_config.get("model_name", "pahm_rl")),
            )
        ]
    )


def train_from_config(config_path: str | Path) -> str:
    """Entrena desde configuracion y retorna la ruta del modelo guardado."""
    config = load_config(config_path)
    rl_config = config["rl_training"]
    if not rl_config.get("enabled", True):
        raise ValueError("rl_training.enabled is false")

    env = make_training_env(config)
    try:
        env.reset(options={"randomize": True})
        model = build_agent(rl_config["algorithm"], env, config)
        model.learn(
            total_timesteps=int(rl_config["total_timesteps"]),
            callback=_build_callbacks(config),
        )

        output_dir = _resolve_path(rl_config["model_output_dir"], Path(config["_config_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / rl_config["model_name"]
        model.save(str(model_path))
        return str(model_path.with_suffix(".zip"))
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena agente RL headless para PAHM")
    parser.add_argument("--config", default="gym_wrapper/config.json")
    args = parser.parse_args()

    model_path = train_from_config(args.config)
    print(f"Modelo RL guardado en: {model_path}")


if __name__ == "__main__":
    main()
