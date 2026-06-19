"""Entrenamiento headless de agentes RL para el entorno PAHM de Etapa 3."""

from __future__ import annotations

import argparse
import copy
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

DEFAULT_EXPERIMENTS = {
    "modes": ["naive", "robust"],
    "naive": {
        "wind_enabled": False,
        "model_name": "pahm_ppo_naive",
    },
    "robust": {
        "wind_enabled": True,
        "model_name": "pahm_ppo_robust",
    },
}

DEFAULT_WANDB = {
    "enabled": False,
    "project": "pahm-rl-stage3",
    "entity": None,
    "mode": "disabled",
    "tags": ["stage3", "rl", "pahm"],
}


class _NullWandbRun:
    def log(self, data: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


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
    config["experiments"] = _merge_experiments(config.get("experiments", {}))
    config["wandb"] = {**DEFAULT_WANDB, **config.get("wandb", {})}
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def _merge_experiments(raw_experiments: dict[str, Any]) -> dict[str, Any]:
    experiments = copy.deepcopy(DEFAULT_EXPERIMENTS)
    for key, value in raw_experiments.items():
        if isinstance(value, dict) and isinstance(experiments.get(key), dict):
            experiments[key] = {**experiments[key], **value}
        else:
            experiments[key] = value
    return experiments


def _config_for_mode(config: dict[str, Any], mode: str | None) -> dict[str, Any]:
    """Aplica overrides de `experiments.<mode>` sobre `rl_training`."""
    selected_mode = mode or config["rl_training"].get("mode", "naive")
    experiments = config["experiments"]
    if selected_mode not in experiments.get("modes", []):
        raise ValueError(f"Modo RL no declarado en experiments.modes: {selected_mode}")

    mode_config = experiments.get(selected_mode, {})
    mode_specific_config = copy.deepcopy(config)
    mode_specific_config["rl_training"] = {
        **mode_specific_config["rl_training"],
        **mode_config,
        "mode": selected_mode,
    }
    return mode_specific_config


def make_training_env(config: dict[str, Any]) -> LearnedPAHMODE:
    """Construye el entorno PAHM para entrenamiento sin renderizado."""
    rl_config = config["rl_training"]
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    mode = rl_config.get("mode", "naive")
    if mode not in {"naive", "robust"}:
        raise ValueError(f"Modo RL no soportado: {mode}")

    model_path = _resolve_path(rl_config["model_path"], config_dir)
    enable_wind = bool(rl_config.get("wind_enabled", mode == "robust"))
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
    callbacks = []
    try:
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
    except ImportError as exc:
        checkpoint_freq = int(config["rl_training"].get("checkpoint_freq", 0))
        if checkpoint_freq <= 0:
            return None
        raise ImportError(
            "stable-baselines3 es requerido para callbacks RL."
        ) from exc

    rl_config = config["rl_training"]
    checkpoint_freq = int(rl_config.get("checkpoint_freq", 0))
    if checkpoint_freq > 0:
        config_dir = Path(config["_config_dir"])
        checkpoint_dir = _resolve_path(
            rl_config.get("checkpoint_dir", rl_config["model_output_dir"]),
            config_dir,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=str(rl_config.get("model_name", "pahm_rl")),
            )
        )

    wandb_run = config.get("_wandb_run")
    if wandb_run is not None and bool(config.get("wandb", {}).get("enabled", False)):
        callbacks.append(_WandbMetricsCallback(wandb_run, BaseCallback))

    if not callbacks:
        return None
    return CallbackList(callbacks)


def _WandbMetricsCallback(wandb_run, base_callback_cls):
    class WandbMetricsCallback(base_callback_cls):
        def _on_step(self) -> bool:
            payload = {
                "train/num_timesteps": self.num_timesteps,
                "train/mode": self.training_env.get_attr("enable_wind")[0]
                if self.training_env is not None
                else None,
            }
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    payload["train/episode_reward"] = info["episode"].get("r")
                    payload["train/episode_length"] = info["episode"].get("l")
                if "abs_tracking_error" in info:
                    payload["train/abs_tracking_error"] = info["abs_tracking_error"]
                if "wind_torque" in info:
                    payload["train/wind_torque"] = info["wind_torque"]
            wandb_run.log(payload)
            return True

    return WandbMetricsCallback()


def _start_wandb_run(config: dict[str, Any]):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return _NullWandbRun()

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb es requerido cuando wandb.enabled=true") from exc

    rl_config = config["rl_training"]
    return wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        mode=wandb_config.get("mode", "online"),
        tags=wandb_config.get("tags", []),
        name=f"{rl_config['model_name']}-{rl_config['mode']}",
        config={
            "rl_training": rl_config,
            "control": config.get("control", {}),
            "reward": config.get("reward", {}),
            "wind": config.get("wind", {}),
            "domain_randomization": config.get("domain_randomization", {}),
        },
    )


def train_from_config(config_path: str | Path, mode: str | None = None) -> str:
    """Entrena desde configuracion y retorna la ruta del modelo guardado."""
    config = _config_for_mode(load_config(config_path), mode)
    rl_config = config["rl_training"]
    if not rl_config.get("enabled", True):
        raise ValueError("rl_training.enabled is false")

    env = make_training_env(config)
    wandb_run = _start_wandb_run(config)
    config["_wandb_run"] = wandb_run
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
        saved_path = str(model_path.with_suffix(".zip"))
        wandb_run.log(
            {
                "model/path": saved_path,
                "train/mode": rl_config["mode"],
                "train/wind_enabled": bool(rl_config.get("wind_enabled", False)),
                "train/total_timesteps": int(rl_config["total_timesteps"]),
            }
        )
        return saved_path
    finally:
        wandb_run.finish()
        env.close()


def train_all_modes(config_path: str | Path) -> dict[str, str]:
    """Entrena todos los modos declarados en `experiments.modes`."""
    config = load_config(config_path)
    model_paths = {}
    for mode in config["experiments"].get("modes", []):
        model_paths[mode] = train_from_config(config_path, mode=mode)
    return model_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena agente RL headless para PAHM")
    parser.add_argument("--config", default="gym_wrapper/config.json")
    parser.add_argument(
        "--mode",
        choices=["naive", "robust", "all"],
        default=None,
        help="Modo a entrenar. Usa rl_training.mode si se omite.",
    )
    args = parser.parse_args()

    if args.mode == "all":
        model_paths = train_all_modes(args.config)
        print(f"Modelos RL guardados: {model_paths}")
    else:
        model_path = train_from_config(args.config, mode=args.mode)
        print(f"Modelo RL guardado en: {model_path}")


if __name__ == "__main__":
    main()
